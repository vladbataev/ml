from model import GatedCnn
from reader import build_vocab

import tensorflow as tf
import os
import time
import datetime
import tqdm
import numpy as np

# Data parameters
tf.flags.DEFINE_string("train_fname", "./train_data/train.txt", "Train file")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 256, "Dimensionality of word embedding (default: 128)")
tf.flags.DEFINE_integer("filter_size", 5, "Filter size")
tf.flags.DEFINE_integer("num_filters", 300, "Number of filters")
tf.flags.DEFINE_integer("num_layers", 20, "Number of convolutional laeyrs")
tf.flags.DEFINE_integer("max_seq_len", 100, "Maximum sequence length")
tf.flags.DEFINE_integer("block_size", 5, "Size of residual block")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("grad_clip", 0.1, "Maximum for norm of gradients")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def pad_sequence(sequence, max_seq_len):
    for i in range(max_seq_len - len(sequence)):
        sequence.append(0)
    return sequence[: max_seq_len]


def train():
    vocab = build_vocab(FLAGS.train_fname)
    train_X, train_y = [], []

    print("Reading training data ...")
    with open(FLAGS.train_fname) as fin:
        for line in tqdm.tqdm(fin):
            if line.strip():
                cur_seq = [vocab["<s>"]]
                for word in line.split():
                    if word in vocab:
                        cur_seq.append(vocab[word])
                if len(cur_seq) > 1:
                    cur_seq.append(vocab["</s>"])
                    train_X.append(np.asarray(pad_sequence(cur_seq[:-1], FLAGS.max_seq_len)))
                    train_y.append(np.asarray(pad_sequence(cur_seq[1:], FLAGS.max_seq_len)))
    print("Reading is done")
    print("Number of sentences: ", len(train_X))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = GatedCnn(embedding_size=FLAGS.embedding_size,
                             filter_size=FLAGS.filter_size,
                             max_seq_len=FLAGS.max_seq_len,
                             block_size=FLAGS.block_size,
                             vocab_size=len(vocab),
                             num_filters=FLAGS.num_filters,
                             num_layers=FLAGS.num_layers,
                             batch_size=FLAGS.batch_size)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer()
            grads_and_vars = optimizer.compute_gradients(model.loss)
            clipped_gradients = [(tf.clip_by_value(_[0], -FLAGS.grad_clip, FLAGS.grad_clip), _[1])
                                 for _ in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped_gradients, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("perplexity", model.perplexity)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            pass

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                }
                try:
                    _, step, summaries, loss, perplexity = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.perplexity],
                    feed_dict)
                except:
                    print(x_batch.shape)
                    print(y_batch.shape)
                    raise
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, perplexity {:g}".format(time_str, step, loss, perplexity))
                train_summary_writer.add_summary(summaries, step)

            # Training loop.
            for epoch_index in range(FLAGS.num_epochs):
                for i in range(0, len(train_X) - FLAGS.batch_size, FLAGS.batch_size):
                    batch_X = train_X[i: i + FLAGS.batch_size]
                    batch_y = train_y[i: i + FLAGS.batch_size]
                    train_step(batch_X, batch_y)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

if __name__ == "__main__":
    train()
