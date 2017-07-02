from collections import defaultdict


def build_vocab(fname, vocab_size=10000):
    """
    Leave only most frequent words
    """
    freq = defaultdict(int)
    with open(fname) as fin:
        for line in fin:
            if line.strip():
                words = line.split()
                for word in words:
                    freq[word] += 1
    most_freq = sorted(freq.items(), key=lambda x: -x[1])[:vocab_size]
    vocab = {word[0]: i + 1 for i, word in enumerate(most_freq)}
    vocab["<s>"] = len(vocab)
    vocab["</s>"] = len(vocab)
    return vocab

#print(build_vocab("./train_data/train.txt"))