import os
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from Transformer import Transformer


class Corpus(Dataset):
    def __init__(self, path, phase, tokenize=True):
        super().__init__()
        self.dictionary = Dictionary()
        self.dict_path = './data/dict.json'
        self.data = os.path.join(path, f'{phase}.tsv')
        self.max_seq_length = 32

        self.init_dictionary(self.data)

        if tokenize:
            self.datas = self.tokenize(self.data)

    def init_dictionary(self, data_path):
        if os.path.exists(self.dict_path):
            print('loaded dict from local file: {0}'.format(self.dict_path))
            self.load_dictionary()
        else:
            print('build dict from training corpus file: {0}'.format(data_path))
            self.dictionary.add_word('<sos>')
            self.dictionary.add_word('<eos>')
            self.dictionary.add_word('<pad>')
            max_len = 0
            with open(data_path, 'r', encoding="utf8") as f:
                for line in f:
                    segs = line.strip().split('\t')
                    for seg in segs:
                        words = seg.split()
                        if len(words) > max_len:
                            max_len = len(words)
                        for word in words:
                            self.dictionary.add_word(word)
            print('length of dictionary: ', len(self.dictionary))
            print('max sentence length: ', max_len)

    def load_dictionary(self):
        with open(self.dict_path, 'r', encoding='utf-8') as rf:
            word_list = json.load(rf)
            self.dictionary = Dictionary()
            for word in word_list:
                self.dictionary.add_word(word)

    def decode(self, batch_ids):
        batch_results = []
        for batch in batch_ids:
            tokens = [self.dictionary.idx2word[i] for i in batch]
            batch_results.append(tokens)
        return batch_results

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            samples = []
            for line in f:
                line = line.strip()
                columns = line.split('\t')
                sample_ids = []
                for column in columns:
                    words = ['<sos>'] + column.split() + ['<eos>']
                    words.extend(['<pad>'] * (self.max_seq_length - len(words)))
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    sample_ids.append(ids)
                samples.append(sample_ids)
        data = torch.tensor(samples).type(torch.int64)
        return data
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        txt = self.datas[item][0]
        label = self.datas[item][1]
        return txt, label

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


###############################################################################
#  parameters
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/200k', help='location of the data corpus')
parser.add_argument('--d_model', type=int, default=256, help='size of word embeddings')
parser.add_argument('--dim_feedforward', type=int, default=1024, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--nhead', type=int, default=2, help='number of heads in the transformer encoder/decoder')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--save', type=str, default='simple_transformer_model.pt', help='path to save the model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dataset = Corpus(args.data, 'train')
vocab_size = len(train_dataset.dictionary)

model = Transformer(vocab_size,
                    args.d_model,
                    args.nhead,
                    args.dim_feedforward,
                    args.nlayers,
                    args.dropout).to(device)


# Load the best saved model.
model.load_state_dict(torch.load(args.save))

def inference(number_list):
    model.eval()
    batch_inputs = []
    batch_y_inputs = []

    for number in number_list:
        number_str = str(number)
        words = [i for i in number_str]
        words = ['<sos>'] + words + ['<eos>']
        words.extend(['<pad>'] * (train_dataset.max_seq_length - len(words)))
        ids = []
        for word in words:
            ids.append(train_dataset.dictionary.word2idx[word])
        batch_inputs.append(ids)
        batch_y_inputs.append([train_dataset.dictionary.word2idx['<sos>']])

    batch_inputs = torch.tensor(batch_inputs).to(device)
    batch_y_inputs = torch.tensor(batch_y_inputs).to(device)

    results = []
    with torch.no_grad():
        for i in range(0, train_dataset.max_seq_length):  # max_seq_length
            output, _, _, _ = model(batch_inputs, batch_y_inputs)
            output = output.view(batch_inputs.size(0), -1, output.size(-1))
            output = output[:, -1:, :]
            output_ids = torch.argmax(output, -1)

            batch_y_inputs = torch.cat([batch_y_inputs, output_ids], dim=1)

            output_texts = train_dataset.decode(output_ids)
            print(output_texts)
            results.append(output_texts)

    print('Result: ')
    for i in range(0, len(number_list)):
        word_list = [results[j][i][0] for j in range(0, len(results))]
        word_list = [i for i in word_list if i not in ['<eos>', '<sos>', '<pad>']]
        words = ' '.join(word_list)
        print(number_list[i], '-->', words)
    return results

inference([
    1235678,
    200.3236,
    -2000,
    10000001,
    66666666,
    -823982502.002,
    987654321.12,
    3295799.9873462
])