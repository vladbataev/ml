import tensorflow as tf
import functools


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class RnnClassifierModel:
    def __init__(self, data, target, vocab_size, cell_type='gru',
                 dropout=0.5, num_hidden=100, num_layers=2, emb_dim=256, num_classes=2):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_layers = num_layers
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._emb_dim = emb_dim
        self._vocab_size = vocab_size
        self._num_classes = num_classes
        self._cell_type = cell_type

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _embedding_matrix(vocab_size, emb_dim):
        return tf.Variable(tf.random_uniform([vocab_size + 1, emb_dim]), name="embedding")

    @lazy_property
    def hidden_states(self):
        embedding_matrix = self._embedding_matrix(self._vocab_size, self._emb_dim)
        emb_vector = tf.nn.embedding_lookup(embedding_matrix, self.data)
        if self._cell_type == "gru":
            network = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        elif self._cell_type == "lstm":
            network = tf.nn.rnn_cell.LSTMCell(self._num_hidden)
        else:
            raise ValueError("Supported cell types: lstm or gru")
        network = tf.nn.rnn_cell.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.nn.rnn_cell.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(network, emb_vector, dtype=tf.float32, swap_memory=True)
        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        return output

    @lazy_property
    def prediction(self):
        output = self.hidden_states
        last = output[-1, :, :]
        # Softmax layer.
        softmax_weights, softmax_bias = self._weight_and_bias(self._num_hidden, self._num_classes)
        probs = tf.nn.softmax(tf.matmul(last, softmax_weights) + softmax_bias)
        return probs

    @lazy_property
    def cost(self):
        probs = self.prediction
        logits = tf.log(probs)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.target)

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.cast(self.target, tf.int64), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
