from d2l import torch as d2l
import random
import re
import torch

def read_war_of_the_worlds():
    """Load the time machine dataset into a list of text lines."""
    with open('./war-of-the-worlds.txt', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md
def load_corpus_war_of_the_worlds(max_tokens=-1):
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_war_of_the_worlds()
    tokens = d2l.tokenize(lines, 'char')
    vocab = d2l.Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# Defined in file: ./chapter_recurrent-neural-networks/language-models-and-dataset.md
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


# Defined in file: ./chapter_recurrent-neural-networks/language-models-and-dataset.md
class SeqDataLoader:
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_war_of_the_worlds(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# Defined in file: ./chapter_recurrent-neural-networks/language-models-and-dataset.md
def load_data_war_of_the_worlds(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


""" Sample Ouput:
martians mass the calling ow the huge gun in the vast pit
martians in iss veithe irrtha suif i fers on mitel chit t
martiansi under footii what we saw from the ruined housei
martians suem to have calluld ringi andy mate ingehiss th
martians they meroues it is ear on the bur that bight aro
martians s eme bee the hume bline which the dest hat an i
martians mass acrrof that ules thouste perave the minds u
martians tae storond of theurslama that ntwerng sttho das
martians saem to ingt ly a mearors of the ourempersucteen
martians suem to have calculated their descent with amazi
martians warred in the same spiritthe martians seem to ha
"""
