'''
Copyright (C) 2020, Northwestern University and Lawrence Berkeley National Laboratory
See COPYRIGHT notice in top-level directory.
'''
import time
import h5py
import numpy as np
import multiprocessing
import horovod.tensorflow as hvd

class IOdaemon:
    def __init__ (self, dataset, do_shuffle = 0, buffer_size = 128):
        self.size = hvd.size()
        self.rank = hvd.rank()
        self.rng = np.random.default_rng()
        self.dataset = dataset
        self.shuffled_index = np.arange(self.dataset.num_train_groups)
        self.buffer_size = buffer_size
        self.do_shuffle = do_shuffle
        self.in_file_off = 0
        self.num_train_files = len(dataset.train_files)
        self.num_valid_files = len(dataset.valid_files)
        self.file_index = 0
        self.group_index = 0
        self.file_sizes = self.dataset.file_sizes

        self.data_shape = (self.buffer_size, 128, 128, 128, 4)
        self.label_shape = (self.buffer_size, 4)

        self.write_index = 0
        self.num_local_train_groups = self.dataset.num_local_train_groups
        self.local_train_group_offset = self.dataset.local_train_group_offset

        print ("R" + str(self.rank) + " will work on "  + str(self.dataset.num_local_train_groups) + " groups.")

    def run (self, lock, cv, finish,
             data, label, num_samples):
        num_buffers = len(data)

        self.shuffled_index[:] = self.dataset.shared_shuffled_index[:]
        print ("R" + str(self.rank) + " updated shuffled_index, [0] is : " + str(self.shuffled_index[0]))
        print ("Number of buffers: " + str(num_buffers))
        
        while 1:
            # Get the current write buffer index.
            write_index = self.write_index

            if finish.value == 1:
                break
            
            buf_off = 0
            if num_samples[write_index].value == 0:
                # Choose a group to read.
                group_index = self.shuffled_index[self.group_index + self.local_train_group_offset]
                sample_offset = group_index * self.buffer_size
                self.file_index = 0
                remain_offset = sample_offset
                for i in range(len(self.file_sizes)):
                    if remain_offset >= self.file_sizes[i]:
                        remain_offset -= self.file_sizes[i]
                    else:
                        self.file_index = i
                        break
                self.in_file_off = remain_offset
                while (buf_off < self.buffer_size):
                    # Open the target file.                 
                    start = time.time()
                    f = h5py.File(self.dataset.train_files[self.file_index], 'r')

                    file_len = f['3Dmap'].shape[0]
                    read_off = self.in_file_off
                    read_len = min (self.buffer_size - buf_off, file_len - read_off)

                    print("R" + str(self.rank) + " file_index: " + str(self.file_index) + " remain_offset: " + str(remain_offset))

                    # Read
                    data_np = np.frombuffer(data[write_index], dtype = np.uint16).reshape(self.data_shape)
                    np.copyto(data_np[buf_off:buf_off + read_len], f['3Dmap'][read_off:read_off + read_len])
                    label_np = np.frombuffer(label[write_index], dtype = np.float32).reshape(self.label_shape)
                    np.copyto(label_np[buf_off:buf_off + read_len], f['unitPar'][read_off:read_off + read_len])

                    f.close()
                    end = time.time()
                    print ("R" + str(self.rank) + " read " + str(read_len) +\
                           " samples from " + str(self.dataset.train_files[self.file_index]) +\
                           " and it took " + str(end - start) + " secs")

                    # Update the offsets.
                    buf_off += read_len
                    self.in_file_off += read_len
                    # If one file has been all consumed, go for the next local file.
                    if self.in_file_off == file_len:
                        self.in_file_off = 0
                        self.file_index += 1
                        
                    if buf_off == self.buffer_size:
                        self.group_index += 1
                        # If all the local groups have been traversed over,
                        # wrap around the index and get the shuffled index from the main thread.
                        if self.group_index == self.num_local_train_groups:
                            self.group_index = 0
                            if self.do_shuffle == 1:
                                self.shuffled_index[:] = self.dataset.shared_shuffled_index[:]
                                print ("R" + str(self.rank) + " updated shuffled_index, [0] is : " + str(self.shuffled_index[0]))

            lock.acquire()
            if buf_off > 0:
                num_samples[write_index].value = self.buffer_size
                self.write_index = (write_index + 1) % num_buffers
                cv.notify()

            while finish.value == 0 and num_samples[self.write_index].value > 0:
                cv.wait()
            lock.release()
