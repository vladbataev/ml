import tensorflow as tf


class GatedCnn:
    def __init__(self, embedding_size, max_seq_len, vocab_size, num_layers,
                 filter_size, num_filters, block_size, batch_size):
        self.input_x = tf.placeholder(tf.int32, [batch_size, max_seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [batch_size, max_seq_len], name="input_y")
        input_x = tf.one_hot(self.input_x, vocab_size, dtype=tf.int32)
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.embedding_weights = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                                              minval=-1.0, maxval=1.0), name="embedding")
            self.embedded_words = tf.nn.embedding_lookup(self.embedding_weights, self.input_x)

        #padding in the beginning to not see the future words
        padded_zeros = tf.zeros([batch_size, filter_size - 1, embedding_size], dtype=tf.float32)
        gate_block_input = tf.concat([padded_zeros, self.embedded_words], 1)
        gate_block_input = tf.expand_dims(gate_block_input, axis=-1)
        for i in range(num_layers):
            with tf.variable_scope("gate-block-scope" + str(i)):
                if i > 0:
                    gate_block_input = tf.squeeze(gate_block_input, axis=-2)
                    gate_block_input = tf.expand_dims(gate_block_input, axis=-1)
                    padded_zeros = tf.zeros([batch_size, filter_size - 1, gate_block_input.shape[-2], 1],
                                            dtype=tf.float32)
                    gate_block_input = tf.concat([padded_zeros, gate_block_input], 1)

                filter_shape = [filter_size, gate_block_input.shape[-2], 1, num_filters]
                linear_output = self.build_conv_block(gate_block_input, filter_shape,
                                                      num_filters, "linear", "VALID")
                gate_output = self.build_conv_block(gate_block_input, filter_shape,
                                                    num_filters, "gated", "VALID")
                gate_block_output = linear_output * tf.sigmoid(gate_output)
                #residual connection
                if i % block_size == 0:
                    if i > 0:
                        gate_block_output += res_input
                    res_input = gate_block_output
                gate_block_input = gate_block_output

        gate_block_output = tf.squeeze(gate_block_output, axis=-2)
        print(gate_block_output.shape)
        with tf.name_scope("output"):
            softmax_w = tf.get_variable("softmax_w", [num_filters, vocab_size], tf.float32,
                                        tf.random_normal_initializer(0.0, 0.1))
            softmax_b = tf.get_variable("softmax_b", [vocab_size], tf.float32, tf.constant_initializer(1.0))
            straight = tf.reshape(gate_block_output, [-1, num_filters])
            straight = tf.add(tf.matmul(straight, softmax_w), softmax_b)
            self.logits = tf.reshape(straight, [batch_size, max_seq_len, vocab_size], name="logits")
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            mask = tf.sign(self.input_y)
            mask = tf.cast(mask, tf.float32)
            cross_entropy *= mask
            cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
            cross_entropy /= tf.reduce_sum(mask, axis=1)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("perplexity"):
            self.perplexity = tf.exp(-self.loss, name="perplexity")

    def build_conv_block(self, block_input, filter_shape, num_filters, name, padding):
        W = tf.get_variable("%s_W" % name, filter_shape, tf.float32, tf.truncated_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b" % name, num_filters, tf.float32, tf.constant_initializer(1.0))
        return tf.add(tf.nn.conv2d(block_input, W, strides=[1, 1, 1, 1], padding=padding), b)

    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        return length
