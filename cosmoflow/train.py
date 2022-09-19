'''
Copyright (C) 2020, Northwestern University and Lawrence Berkeley National Laboratory
See COPYRIGHT notice in top-level directory.
'''
import time
import tensorflow as tf
import multiprocessing as mp
from mpi4py import MPI
import horovod.tensorflow as hvd
from tqdm import tqdm
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

class Trainer:
    def __init__ (self, model, denser_model, async_io = 1, dataset = None, dataset_denser = None,
                  do_shuffle = 0, num_epochs = 1, checkpoint_dir = "/global/cscratch1/sd/kwf5687/cosmoflow_c1/cp",
                  do_checkpoint = 0, do_record_acc = 0, do_evaluate = 0):
        self.rank = hvd.rank()
        self.num_epochs = num_epochs
        self.async_io = async_io
        self.dataset = dataset
        self.dataset_denser = dataset_denser
        self.dataset_temp = dataset
        self.do_shuffle = do_shuffle
        self.do_record_acc = do_record_acc
        self.do_evaluate = do_evaluate
        model = model.build_model()
        self.model = model
        model.summary()
        self.checkpoint_dir = checkpoint_dir
        self.denser_model = denser_model.build_model()
        self.denser_model.summary()

        # This learning rate setting is for parallel training with a batch size of 256.
        lr = PiecewiseConstantDecay(boundaries = [1600, 2400],
                                    values = [2e-3, 2e-4, 2e-5])
        lr_d = PiecewiseConstantDecay(boundaries = [1600, 2400],
                                    values = [8e-4, 8e-5, 8e-6])
        self.loss = MeanSquaredError()
        opt = Adam(learning_rate = lr)
        opt_denser = Adam(learning_rate = lr_d)
        self.do_checkpoint = do_checkpoint
        self.checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0),
                                              model = model,
                                              optimizer = opt)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint,
                                                             directory = checkpoint_dir,
                                                             max_to_keep = 100)
        self.checkpoint_denser = tf.train.Checkpoint(epoch = tf.Variable(0),
                                                             model = self.denser_model,
                                                             optimizer = opt_denser)
        self.checkpoint_manager_denser = tf.train.CheckpointManager(checkpoint = self.checkpoint_denser,
                                                             directory = self.checkpoint_dir,
                                                             max_to_keep = 3)
        self.resume()

    def resume (self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print ("Model restored from checkpoint at epoch " + str(self.checkpoint.epoch.numpy()))

    @tf.function
    def train_step (self, data, label):
        with tf.GradientTape() as tape:
            prediction = self.checkpoint.model(data, training = True)
            loss = self.loss(label, prediction)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
        return loss

    @tf.function
    def train_step_denser (self, data, label):
        with tf.GradientTape() as tape:
            prediction = self.checkpoint_denser.model(data, training = True)
            loss = self.loss(label, prediction)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, self.checkpoint_denser.model.trainable_variables)
        self.checkpoint_denser.optimizer.apply_gradients(zip(gradients, self.checkpoint_denser.model.trainable_variables))
        return loss

    def train (self):
        train_dataset = self.dataset.train_dataset()
        valid_dataset = self.dataset.valid_dataset()

        first_epoch = self.checkpoint.epoch.numpy()
        for epoch_id in range(first_epoch, 50):
            print ("Epoch: " + str(epoch_id) + " lr: " + str(self.checkpoint.optimizer._decayed_lr('float32').numpy()))

            self.checkpoint.epoch.assign_add(1)
            self.dataset.train_file_index = 0
            self.dataset.waiting = 0
            loss_mean = Mean()
            self.start_time = time.perf_counter()

            if self.do_shuffle == 1:
                self.dataset.shuffle()

            # Train the model.
            for i in tqdm(range(self.dataset.num_train_batches)):
                if self.async_io == 1:
                    self.dataset.pre_batch()
                data, label = train_dataset.next()
                if self.async_io == 1:
                    self.dataset.post_batch()
                loss = self.train_step(data, label)
                loss_mean(loss)

                if epoch_id == 0 and i == 0:
                    hvd.broadcast_variables(self.checkpoint.model.variables, root_rank=0)
                    hvd.broadcast_variables(self.checkpoint.optimizer.variables(), root_rank=0)

            timing = time.perf_counter() - self.start_time
            train_loss = loss_mean.result()
            loss_mean.reset_states()

            if hvd.rank() == 0 and self.do_checkpoint == True and epoch_id % 5 == 0:
                self.checkpoint_manager.save()

            # Evaluate the current model using the validation data.
            if self.do_evaluate == 1:
                valid_loss = self.evaluate(valid_dataset, 0)
                valid_loss_np = valid_loss.numpy()
                average_loss = MPI.COMM_WORLD.allreduce(valid_loss_np, MPI.SUM) / MPI.COMM_WORLD.Get_size()

                print ("Epoch " + str(self.checkpoint.epoch.numpy()) +\
                       " training loss = " + str(train_loss.numpy()) +\
                       " validation loss = " + str(average_loss) +\
                       " training timing: " + str(timing) + " sec")
            else:
                print ("R " + str(self.rank) + "Epoch " + str(self.checkpoint.epoch.numpy()) +\
                       " waiting time = " + str(self.dataset.waiting) +\
                       " training loss = " + str(train_loss.numpy()) +\
                       " training timing: " + str(timing) + " sec")

            # Write the loss values to the output files.
            if self.rank == 0 and self.do_record_acc == 1:
                f = open("loss-train.txt", "a")
                f.write(str(train_loss.numpy()) + "\n")
                f.close()
                f = open("loss-valid.txt", "a")
                f.write(str(average_loss) + "\n")
                f.close()

        # copy the weights to the denser model
        print("R " + str(self.rank) + "copy model weights")
        for i in range(7):
            conv = self.model.get_layer(name = 'conv' + str(i + 1))
            self.denser_model.get_layer(name = 'conv' + str(i + 1)).set_weights(conv.get_weights())

        for i in range(8):
            bn = self.model.get_layer(name = 'bn' + str(i + 1))
            self.denser_model.get_layer(name = 'bn' + str(i + 1)).set_weights(bn.get_weights())

        train_dataset = self.dataset_denser.train_dataset()
        valid_dataset = self.dataset_denser.valid_dataset()
        self.dataset_temp = self.dataset_denser
        # evaluate before training with denser model
        first_epoch = self.checkpoint_denser.epoch.numpy()

        for epoch_id in range(first_epoch, self.num_epochs):
            print ("Epoch: " + str(epoch_id) + " lr: " + str(self.checkpoint_denser.optimizer._decayed_lr('float32').numpy()))

            self.checkpoint_denser.epoch.assign_add(1)
            self.dataset_denser.train_file_index = 0
            self.dataset_denser.waiting = 0
            loss_mean = Mean()
            self.start_time = time.perf_counter()

            if self.do_shuffle == 1:
                self.dataset_denser.shuffle()

            # Train the model.
            for i in tqdm(range(self.dataset_denser.num_train_batches)):
                if self.async_io == 1:
                    self.dataset_denser.pre_batch()
                data, label = train_dataset.next()
                if self.async_io == 1:
                    self.dataset_denser.post_batch()
                loss = self.train_step_denser(data, label)
                loss_mean(loss)

            if epoch_id == 0 and i == 0:
                    hvd.broadcast_variables(self.checkpoint_denser.model.variables, root_rank=0)
                    hvd.broadcast_variables(self.checkpoint_denser.optimizer.variables(), root_rank=0)

            timing = time.perf_counter() - self.start_time
            train_loss = loss_mean.result()
            loss_mean.reset_states()

            # Evaluate the current model using the validation data.
            if self.do_evaluate == 1:
                valid_loss = self.evaluate(valid_dataset, 1)
                valid_loss_np = valid_loss.numpy()
                average_loss = MPI.COMM_WORLD.allreduce(valid_loss_np, MPI.SUM) / MPI.COMM_WORLD.Get_size()
                
                print ("denser_model Epoch " + str(self.checkpoint_denser.epoch.numpy()) +\
                       " training loss = " + str(train_loss.numpy()) +\
                       " validation loss = " + str(average_loss) +\
                       " training timing: " + str(timing) + " sec")
            else:
                print ("denser_model R " + str(self.rank) + "Epoch " + str(self.checkpoint_denser.epoch.numpy()) +\
                       " waiting time = " + str(self.dataset_denser.waiting) +\
                       " training loss = " + str(train_loss.numpy()) +\
                       " training timing: " + str(timing) + " sec")
            if self.rank == 0 and self.do_record_acc == 1:
                f = open("loss-train.txt", "a")
                f.write(str(train_loss.numpy()) + "\n")
                f.close()
                f = open("loss-valid.txt", "a")
                f.write(str(average_loss) + "\n")
                f.close()

    def evaluate (self, dataset, model_type):
        self.dataset_temp.valid_file_index = 0
        loss_mean = Mean()
        for i in tqdm(range(self.dataset_temp.num_valid_batches)):
            data, label = dataset.next()
            if model_type == 0:
                prediction = self.checkpoint.model(data)
            else:
                prediction = self.checkpoint_denser.model(data)
            loss = self.loss(label, prediction)
            loss_mean(loss)
        return loss_mean.result()
