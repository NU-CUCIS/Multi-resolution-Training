'''
Copyright (C) 2020, Northwestern University and Lawrence Berkeley National Laboratory
See COPYRIGHT notice in top-level directory.
'''
import tensorflow as tf
import time
import argparse
import numpy as np
import multiprocessing as mp
import horovod.tensorflow as hvd
from model.model import model
from model.denser_model import denser_model
from train import Trainer
from IO.io_daemon import IOdaemon
from IO.feeder_sync import cosmoflow_sync
from IO.feeder_async import cosmoflow_async
from IO.feeder_sync_denser import cosmoflow_sync_denser

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type = int, default = 8,
                        help = "number of training samples for each mini-batch")
    parser.add_argument("-o", "--overlap", type = int, default = 1,
                        help = "0: do not overlap I/O with computation, 1: overlap I/O with computation")
    parser.add_argument("-p", "--checkpoint", type = int, default = 0,
                        help = "0: do not checkpoint the model, 1: checkpoint the model")
    parser.add_argument("-e", "--epochs", type = int, default = 1,
                        help = "number of epochs")
    parser.add_argument("-k", "--buffer_size", type = int, default = 128,
                        help = "buffer size with respect to the number of samples")
    parser.add_argument("-f", "--file_shuffle", type = int, default = 0,
                        help = "shuffle the files across the processes")
    parser.add_argument("-r", "--record_acc", type = int, default = 0,
                        help = "write the loss and accuracy into output files")
    parser.add_argument("-v", "--evaluate", type = int, default = 0,
                        help = "evaluate the model after every epoch")
    parser.add_argument("-y", "--config", default = "test.yaml",
                        help = "yaml file that describes the input data")
    parser.add_argument("-a", "--async_io", type = int, default = 1,
                        help = "asynchronous I/O module")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()

    # Initialize Horovod.tensorflow.
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        # On Summit, each resource set can view its own GPUs only.
        # So, the visible devices should be set to gpu:0 for every process.
        #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # multiprocessing synchronization
    lock = mp.Lock()
    cv = mp.Condition(lock = lock)
    finish = mp.Value('i')
    finish.value = 0
    data_buffer_size = args.buffer_size * 64 * 64 * 64 * 4
    label_buffer_size = args.buffer_size * 4

    # Initialize the shared memory space between TF process and I/O process.
    data = []
    label = []
    num_samples = []
    if args.overlap == 0:
        num_buffers = 1
    else:
        num_buffers = 2
    for i in range(num_buffers):
        data.append(mp.RawArray('H', data_buffer_size))
        label.append(mp.RawArray('f', label_buffer_size))
        num_samples.append(mp.Value('i'))
        num_samples[i].value = 0

    # Initialize model, dataset, and trainer.
    cosmo_model = model()
    denser_model = denser_model()
    if args.async_io == 1:
        dataset = cosmoflow_async(args.config, lock, cv,
                                  data, label, num_samples,
                                  batch_size = args.batch_size,
                                  buffer_size = args.buffer_size)
        # Initialize the I/O daemon.
        async_io_module = IOdaemon(dataset,
                                   args.file_shuffle,
                                   args.buffer_size)
    else:
        dataset = cosmoflow_sync(args.config,
                                 do_shuffle = args.file_shuffle,
                                 batch_size = args.batch_size,
                                 buffer_size = args.buffer_size)
        dataset_d = cosmoflow_sync_denser(args.config,
                                 do_shuffle = args.file_shuffle,
                                 batch_size = args.batch_size,
                                 buffer_size = args.buffer_size)

    trainer = Trainer(cosmo_model,
                      denser_model,
                      args.async_io,
                      dataset,
                      dataset_d,
                      do_shuffle = args.file_shuffle,
                      num_epochs = args.epochs,
                      do_checkpoint = args.checkpoint,
                      do_record_acc = args.record_acc,
                      do_evaluate = args.evaluate)

    if args.async_io == 1:
        io_process = mp.Process(target = async_io_module.run,
                                args = (lock, cv, finish,
                                        data, label, num_samples))
        io_process.start()

    # Start the training.
    trainer.train()

    # Kill the I/O process and finish the program.
    if args.async_io == 1:
        lock.acquire()
        finish.value = 1
        cv.notify()
        lock.release()
        io_process.join()
    print ("All done!")
