import tensorflow as tf
import argparse
import glob
import numpy as np
from . import utils
from .model import RnnClassifierModel


def main():
    parser = argparse.ArgumentParser(description='Rnn classifier arguments')
    parser.add_argument('train_data_dir', type=str, default='./data/',
                        help='Train data directory')
    parser.add_argument('emb_dim', type=int, default=256,
                        help='Dimension of word embedding')
    parser.add_argument('batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('num_hidden', type=int, default=2,
                        help='Size of hidden layers in rnn cell')
    parser.add_argument('dropout', type=float, default=0.5,
                        help='Dropout level')
    parser.add_argument('num_classes', type=int, default=2,
                        help='Number of classes')

    args = parser.parse_args()

    filenames = glob.glob(args.train_data_dir)
    dictionary = utils.build_dictionary(filenames)
    keys = tf.constant(list(dictionary.keys()))
    values = tf.constant(list(dictionary.values()), dtype=tf.int64)
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)

    inputs, target = utils.input_pipeline(filenames, batch_size=args.batch_size)

    num_objects = utils.count_num_objects(args.train_data_dir)

    with tf.device("/gpu:1"):
        model = RnnClassifierModel(inputs, target,
                                   vocab_size=len(dictionary),
                                   num_classes=args.num_classes)
        training_opt = model.optimize
        init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        table.init.run()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(args.num_epochs):
            epoch_error = []
            for _ in range(num_objects // args.batch_size + 1):
                batch_data = sess.run(inputs)
                batch_target = sess.run(target)
                model.data = batch_data
                model.target = batch_target
                sess.run(training_opt)
                batch_error = sess.run(model.error)
                epoch_error.append(batch_error)
            print("Epoch error: {}".format(np.mean(epoch_error)))
            print("Epoch {} is finished".format(i))
        coord.request_stop()
        coord.join(threads)
        save_path = saver.save(sess, "./tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main()
