#! /usr/bin/env python
import sys

#SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM  #OPTION 0
from lstm_cnn import LSTM_CNN  #OPTION 1
from cnn import CNN  #OPTION 2 (Model by: Danny Britz)
from lstm import LSTM  #OPTION 3
from graph_res import plot_results
MODEL_TO_RUN = 1

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import batchgen
from tensorflow.contrib import learn
import statistics

from IPython import embed

# Parameters
# ==================================================

# Data loading params
dev_size = 0.1

# Model Hyperparameters
embedding_dim = 32  #128
max_seq_legth = 70
filter_sizes = [3, 4, 5]  #3
num_filters = 32
dropout_prob = 0.9  #0.5
l2_reg_lambda = 0.0
use_glove = False  #Do we use glove

# Training parameters
batch_size = 128
num_epochs = 10  #200
evaluate_every = 100  #100
checkpoint_every = 100000  #100
num_checkpoints = 0  #Checkpoints to store
MAX_TRAIN_SIZE = 20000

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

# Parameters for plt
training_steps = []
training_losses = []
training_accuracy = []

evaluation_steps = []
evaluation_losses = []
evaluation_accuracy = []

evaluation_losses_noncovid = []
evaluation_accuracy_noncovid = []

# Data Preparation
# ==================================================

filename = "../tweets.csv"
goodfile = "../data/good_tweets.csv"
badfile = "../data/bad_tweets.csv"
goodvalidfile = "../data/good_validation.csv"
badvalidfile = "../data/bad_validation.csv"

# Load data
print("Loading data...")
x_text, y = batchgen.get_dataset(goodfile, badfile,
                                 MAX_TRAIN_SIZE)  #TODO: MAX LENGTH
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
if (not use_glove):
    print("Not using GloVe")
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
else:
    print("Using GloVe")
    embedding_dim = 50
    filename = '../glove.6B.50d.txt'

    def loadGloVe(filename):
        vocab = []
        embd = []
        file = open(filename, 'r')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
        print('Loaded GloVe!')
        file.close()
        return vocab, embd

    vocab, embd = loadGloVe(filename)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)

    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False,
                    name="W")
    embedding_placeholder = tf.placeholder(tf.float32,
                                           [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)

    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
    sess = tf.Session(config=session_conf)
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    from tensorflow.contrib import learn
    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    x = np.array(list(vocab_processor.transform(x_text)))

    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    x = np.array(list(vocab_processor.transform(x_text)))

# Randomly shuffle data
np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(dev_size * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

x_covid, y_covid = batchgen.get_dataset(goodvalidfile, badvalidfile,
                                        MAX_TRAIN_SIZE)
x_covid = np.array(list(vocab_processor.fit_transform(x_covid)))

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

#embed()

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #embed()
        if (MODEL_TO_RUN == 0):
            model = CNN_LSTM(x_train.shape[1], y_train.shape[1],
                             len(vocab_processor.vocabulary_), embedding_dim,
                             filter_sizes, num_filters, l2_reg_lambda)
        elif (MODEL_TO_RUN == 1):
            model = LSTM_CNN(x_train.shape[1], y_train.shape[1],
                             len(vocab_processor.vocabulary_), embedding_dim,
                             filter_sizes, num_filters, l2_reg_lambda)
        elif (MODEL_TO_RUN == 2):
            model = CNN(x_train.shape[1], y_train.shape[1],
                        len(vocab_processor.vocabulary_), embedding_dim,
                        filter_sizes, num_filters, l2_reg_lambda)
        elif (MODEL_TO_RUN == 3):
            model = LSTM(x_train.shape[1], y_train.shape[1],
                         len(vocab_processor.vocabulary_), embedding_dim)
        else:
            print(
                "PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n"
            )
            exit()

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                     sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        #TRAINING STEP
        def train_step(x_batch, y_batch, save=False):
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: dropout_prob
            }
            _, step, summaries, loss, accuracy = sess.run([
                train_op, global_step, train_summary_op, model.loss,
                model.accuracy
            ], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))

            # Training step
            if step % 5 == 0:
                training_steps.append(step)
                training_losses.append(loss)
                training_accuracy.append(accuracy)

            if save:
                train_summary_writer.add_summary(summaries, step)

        #EVALUATE MODEL
        def dev_step(x_batch,
                     y_batch,
                     writer=None,
                     save=False,
                     evaluate_step_general_tweets=False):
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: 0.5
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))

            if not evaluate_step_general_tweets:
                # Evaluation step using covid tweetes

                evaluation_steps.append(step)
                evaluation_losses.append(loss)
                evaluation_accuracy.append(accuracy)
            else:
                # Using non-covid tweets

                evaluation_losses_noncovid.append(loss)
                evaluation_accuracy_noncovid.append(accuracy)

            if save:
                if writer:
                    writer.add_summary(summaries, step)

            return loss, accuracy

        #CREATE THE BATCHES GENERATOR
        batches = batchgen.gen_batch(list(zip(x_train, y_train)), batch_size,
                                     num_epochs)

        # RECORD BEST RESULT
        highest_acc = -1
        record_loss = 0
        record_step = 0

        highest_acc_noncovid = -1
        record_loss_noncovid = 0
        record_step_noncovid = 0

        #TRAIN FOR EACH BATCH
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation for general tweets:")
                step_loss, step_accuracy = dev_step(
                    x_dev,
                    y_dev,
                    writer=dev_summary_writer,
                    evaluate_step_general_tweets=True)

                if step_accuracy > highest_acc_noncovid:
                    highest_acc_noncovid = step_accuracy
                    record_loss_noncovid = step_loss
                    record_step_noncovid = current_step

                print("")
                print("\nEvaluation for the covid tweets:")
                step_loss, step_accuracy = dev_step(x_covid,
                                                    y_covid,
                                                    writer=dev_summary_writer)

                if step_accuracy > highest_acc:
                    highest_acc = step_accuracy
                    record_loss = step_loss
                    record_step = current_step

                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess,
                                  checkpoint_prefix,
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        dev_step(x_dev,
                 y_dev,
                 writer=dev_summary_writer,
                 evaluate_step_general_tweets=True)
        dev_step(x_covid,
                 y_covid,
                 writer=dev_summary_writer,
                 evaluate_step_general_tweets=False)

print("General tweets: step {}, loss {:g}, acc {:g}".format(
    record_step_noncovid, record_loss_noncovid, highest_acc_noncovid))
print("Covid-related tweets: step {}, loss {:g}, acc {:g}".format(
    record_step, record_loss, highest_acc))
print(f'min covid-loss {min(evaluation_losses)}')
print(f'min non-covid-loss {min(evaluation_losses_noncovid)}')

print(f'avg covid accuracy {statistics.mean(evaluation_accuracy)}')
print(
    f'avg non-covid accuracy {statistics.mean(evaluation_accuracy_noncovid)}')
plot_results(training_steps, training_losses, training_accuracy,
             evaluation_steps, evaluation_losses, evaluation_accuracy,
             evaluation_losses_noncovid, evaluation_accuracy_noncovid,
             record_step, highest_acc, record_step_noncovid,
             highest_acc_noncovid)
