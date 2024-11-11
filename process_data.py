import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and splits into folds.
    """
    revs = []
    pos_file, neg_file = data_folder
    vocab = defaultdict(float)

    # Helper function to clean and process the reviews
    def process_file(file, label):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                rev = line.strip()
                orig_rev = clean_str(rev) if clean_string else rev.lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum = {"y": label,
                         "text": orig_rev,
                         "num_words": len(orig_rev.split()),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)

    # Process positive and negative reviews
    process_file(pos_file, 1)
    process_file(neg_file, 0)

    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = {}
    W = np.zeros((vocab_size + 1, k), dtype='float32')

    i = 1
    for word, vec in word_vecs.items():
        W[i] = vec
        word_idx_map[word] = i
        i += 1

    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vectors from Google (Mikolov) word2vec.
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = ''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch.decode())
            if word in vocab:
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)  # Skip non-matching words
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    Add random vectors for words that are not in word2vec.
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() if not TREC else string.strip()


if __name__ == "__main__":
    w2v_file = sys.argv[1]
    data_folder = ["rt-polarity.pos", "rt-polarity.neg"]

    print("Loading data...")
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)

    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print(f"Data loaded! Number of sentences: {len(revs)}, Vocab size: {len(vocab)}, Max sentence length: {max_l}")

    print("Loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, vocab)
    print(f"Word2vec loaded! Num words already in word2vec: {len(w2v)}")

    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    with open("mr.p", "wb") as f:
        pickle.dump([revs, W, W2, word_idx_map, vocab], f)

    print("Dataset created!")
