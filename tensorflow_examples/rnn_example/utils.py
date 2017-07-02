import tensorflow as tf
import subprocess


def build_dictionary(filenames):
    index = 1
    dictionary = {}

    for filename in filenames:
        with open(filename) as in_file:
            for row in in_file:
                words = (row.split(',')[1]).split()
                for word in words:
                    if word not in dictionary:
                        dictionary[word] = index
                        index += 1
    return dictionary


def sentence2indexes(example, table):
    return table.lookup(tf.string_split(example).values)


def read_my_file_format(filename_queue, table):
    reader = tf.TextLineReader()
    key, record_string = reader.read(filename_queue)
    record_defaults = [[1], [''], [1]]
    dialog_id, dialog, label = tf.decode_csv(record_string, record_defaults=record_defaults)
    dialog = tf.pack([dialog])
    processed_example = sentence2indexes(dialog, table)
    return processed_example, label


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_my_file_format(filename_queue)
    capacity = 100
    example_batch, label_batch = tf.train.batch(
        [example, label],
        batch_size=batch_size,
        capacity=capacity,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
    )
    return example_batch, label_batch


def count_num_objects(train_dir):
    command = 'find ' + train_dir + ' -name "*.csv" | xargs wc -l'
    output, error = subprocess.Popen(
        command, universal_newlines=True, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    return int(output.split('\n')[-2].split()[0])
