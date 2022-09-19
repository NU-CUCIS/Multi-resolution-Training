'''
Copyright (C) 2020, Northwestern University and Lawrence Berkeley National Laboratory
See COPYRIGHT notice in top-level directory.
'''
import os
import time
import tensorflow as tf
import yaml
import numpy as np
import h5py
import math
from mpi4py import MPI
import multiprocessing as mp

class cosmoflow_sync_denser:
    def __init__ (self, yaml_file,
                  do_shuffle = 0,
                  batch_size = 4,
                  buffer_size = 128):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.read_index = 0
        self.rng = np.random.default_rng()
        self.do_shuffle = do_shuffle
        self.num_cached_train_batches = 0
        self.num_cached_valid_batches = 0
        self.num_cached_test_batches = 0
        self.train_file_index = 0
        self.valid_file_index = 0
        self.test_file_index = 0
        self.data_shape = (self.buffer_size, 128, 128, 128, 4)
        self.label_shape = (self.buffer_size, 4)
        self.file_index = 0

        # Parse the given yaml file and get the top dir and file names.
        with open (yaml_file, "r") as f:
            data = yaml.load(f, Loader = yaml.FullLoader)
            for key, value in data.items():
                if key == 'frameCnt':
                    self.samples_per_file = value
                    self.batches_per_file = int(value / self.batch_size)

                if key == 'numPar':
                    self.label_size = value

                if key == 'sourceDir':
                    self.prj = value['prj']
                    self.cfs = value['cfs']
                    
                if key == 'subDir_denser':
                    self.subdir = value

                if key == 'subDir_eval':
                    self.subdir_eval = value

                if key == 'splitIdx':
                    self.train_files = list(value['train'])
                    self.valid_files = list(value['val'])
                    self.test_files = list(value['test'])

                    self.train_files = [str(self.prj) + "/" + 
                                        str(self.subdir) + "/" +
                                        "PeterA_2019_05_4parE-rec" +
                                        str(file_name[1]) +
                                        ".h5" for file_name in enumerate(self.train_files)]
                    self.valid_files = [str(self.prj) + "/" +
                                        str(self.subdir) + "/" +
                                        #str(self.subdir) + str(self.subdir_eval) + "/" +
                                        "PeterA_2019_05_4parE-rec" +
                                        #"rec" +
                                        str(file_name[1]) +
                                        ".h5" for file_name in enumerate(self.valid_files)]
                    self.test_files = [str(self.prj) + "/" +
                                        str(self.subdir) + "/" +
                                        "PeterA_2019_05_4parE-rec" +
                                        str(file_name[1]) +
                                        ".h5" for file_name in enumerate(self.test_files)]

            print ("Number of samples per file: " + str(self.samples_per_file))
            print ("Label size: " + str(self.label_size))
            print ("sourceDir.prj: " + str(self.prj))
            print ("sourceDir.cfs: " + str(self.cfs))
            print ("subDir: " + str(self.subdir))
        print ("Buffer size: " + str(self.buffer_size) + " samples")

        self.num_train_files = len(self.train_files)
        self.offset = int(self.num_train_files / self.size) * self.rank

        # First, calculate the number of local files.
        common = int(self.num_train_files / self.size)
        remainder = self.num_train_files % self.size

        if self.rank < remainder:
            self.num_local_train_files = common + 1
        else:
            self.num_local_train_files = common

        self.num_train_batches = int(self.batches_per_file * self.num_local_train_files)

        # Count the number of local files for validaiton.
        num_local_valid_files = int(math.floor(len(self.valid_files) / self.size))
        local_valid_files_off = num_local_valid_files * self.rank
        if self.rank < (len(self.valid_files) % self.size):
            num_local_valid_files += 1
            local_valid_files_off += self.rank
        else:
            local_valid_files_off += (len(self.valid_files) % self.size)
        self.local_valid_files = self.valid_files[local_valid_files_off:
                                                  local_valid_files_off + num_local_valid_files]

        self.num_valid_batches = 0
        for file_path in self.local_valid_files:
            f = h5py.File(file_path, 'r')
            self.num_valid_batches += f['unitPar'].shape[0]
            f.close()
        self.num_valid_batches = int(math.floor(self.num_valid_batches / self.batch_size))

        # Calculate the number of local test files.
        num_local_test_files = int(math.floor(len(self.test_files) / self.size))
        local_test_files_off = num_local_test_files * self.rank
        if self.rank < (len(self.test_files) % self.size):
            num_local_test_files += 1
            local_test_files_off += self.rank
        else:
            local_test_files_off += (len(self.test_files) % self.size)
        self.local_test_files = self.test_files[local_test_files_off:
                                                  local_test_files_off + num_local_test_files]
                                                  
        self.num_test_batches = 0
        for file_path in self.local_test_files:
            f = h5py.File(file_path, 'r')
            self.num_test_batches += f['unitPar'].shape[0]
            f.close()
        self.num_test_batches = int(math.floor(self.num_test_batches / self.batch_size))

        self.shuffle()

    def shuffle (self):
        # Shuffle the file index.
        self.shuffled_file_index = np.arange(self.num_train_files)
        self.rng.shuffle(self.shuffled_file_index)
        self.comm.Bcast(self.shuffled_file_index, root = 0) 

        self.shuffled_sample_index = np.arange(128)
        self.rng.shuffle(self.shuffled_sample_index)

    '''
    Sample-based prefetch
    '''
    def read_train_sample (self, sample_id):
        # 1. Find a file.
        file_index = int(sample_id.numpy() / 128)
        file_index = self.shuffled_file_index[file_index + self.offset]
        f = h5py.File(self.train_files[file_index], 'r')

        # 2. Read a sample.
        sample_index = sample_id.numpy() % 128
        sample_index = self.shuffled_sample_index[sample_index]
        if sample_index >= f['unitPar'].shape[0]:
            sample_index %= f['unitPar'].shape[0]

        images = f['3Dmap'][sample_index]
        labels = f['unitPar'][sample_index]

        f.close()

        return images, labels

    def tf_read_train_sample (self, sample_id):
        images, labels = tf.py_function(self.read_train_sample, inp=[sample_id], Tout=[tf.float32, tf.float32])
        return images, labels

    def train_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_local_train_files * 128))
        dataset = dataset.map(self.tf_read_train_sample)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(4)
        #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset.__iter__()

    '''
    Batch-based prefetch
    '''
    def read_train_batch (self, batch_id):
        # 1. Find a file.
        file_index = int(batch_id.numpy() / self.batches_per_file)
        file_index = self.shuffled_file_index[file_index + self.offset]
        f = h5py.File(self.train_files[file_index], 'r')

        # 2. Read a batch.
        batch_index = batch_id.numpy() % self.batches_per_file
        batch_index = self.shuffled_batch_index[batch_index]
        if (batch_index * self.batch_size) >= f['unitPar'].shape[0]:
            batch_index %= int(f['unitPar'].shape[0] / self.batch_size)
        batch_index *= self.batch_size

        images = f['3Dmap'][batch_index: batch_index + self.batch_size]
        labels = f['unitPar'][batch_index: batch_index + self.batch_size]
        f.close()

        return images, labels

    def tf_read_train_batch (self, batch_id):
        images, labels = tf.py_function(self.read_train_batch, inp=[batch_id], Tout=[tf.float32, tf.float32])
        images.set_shape([self.batch_size, 128, 128, 128, 4])
        labels.set_shape([self.batch_size, 4])
        return images, labels

    def train_dataset_batch (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_train_batches))
        dataset = dataset.map(self.tf_read_train_batch)
        dataset = dataset.repeat()
        return dataset.__iter__()

    '''
    Functions for validation
    '''
    def read_valid_samples (self, batch_id):
        # Read a new file if there are no cached batches.
        if self.num_cached_valid_batches == 0:
            for i in range(len(self.local_valid_files)):
                f = h5py.File(self.local_valid_files[i], 'r')
                if i == 0:
                    self.images = f['3Dmap'][:]
                    self.labels = f['unitPar'][:]
                else:
                    self.images = np.append(self.images, f['3Dmap'][:], axis = 0)
                    self.labels = np.append(self.labels, f['unitPar'][:], axis = 0)
                f.close()

            self.num_cached_valid_batches = int(self.images.shape[0] / self.batch_size)

        # Get a mini-batch from the memory buffer.
        index = (self.num_cached_valid_batches - 1) * self.batch_size
        images = self.images[index : index + self.batch_size]
        labels = self.labels[index : index + self.batch_size]
        self.num_cached_valid_batches -= 1
        return images, labels

    def tf_read_valid_samples (self, batch_id):
        images, labels = tf.py_function(self.read_valid_samples, inp=[batch_id], Tout=[tf.float32, tf.float32])
        images.set_shape([self.batch_size, 128, 128, 128, 4])
        labels.set_shape([self.batch_size, 4])
        return images, labels

    def valid_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_batches))
        dataset = dataset.map(self.tf_read_valid_samples)
        dataset = dataset.repeat()
        return dataset.__iter__()

    '''
    test dataset
    '''
    def read_test_samples (self, batch_id):
        # Read a new file if there are no cached batches.
        if self.num_cached_test_batches == 0:
            if self.test_file_index == len(self.local_test_files):
                print ("batch_id: " + str(batch_id) + " Invalid test_file_index! " + str(self.test_file_index) + "/" + str(len(self.test_files)))
            f = h5py.File(self.local_test_files[self.test_file_index], 'r')
            self.test_file_index += 1
            self.images = f['3Dmap'][:,:,:,:,:]
            self.labels = f['unitPar'][:]
            f.close()
            self.num_cached_test_batches = int(self.images.shape[0] / self.batch_size)

        # Get a mini-batch from the memory buffer.
        index = (self.num_cached_test_batches - 1) * self.batch_size
        images = self.images[index : index + self.batch_size]
        labels = self.labels[index : index + self.batch_size]
        self.num_cached_test_batches -= 1
        return images, labels

    def tf_read_test_samples (self, batch_id):
        images, labels = tf.py_function(self.read_test_samples, inp=[batch_id], Tout=[tf.float32, tf.float32])
        images.set_shape([self.batch_size, 128,128,128,4])
        labels.set_shape([self.batch_size, 4])
        return images, labels

    def test_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_test_batches))
        dataset = dataset.map(self.tf_read_test_samples)
        dataset = dataset.repeat()
        return dataset.__iter__()
