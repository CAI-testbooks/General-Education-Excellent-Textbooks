'''
Author: Chengyu Zheng
Date: 2024-11-15
Description: 
'''
import os
import time
import math
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from tqdm import tqdm

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


def evaluate(model, data_loader, batch_size):
    """
    evaluation function
    """
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        evaluate_iterator = tqdm(enumerate(data_loader), total=len(data_loader))
        for iteration, data in evaluate_iterator:
            txt = data[0].to(device)
            label = data[1].to(device)
            label_input = label[:, :-1]
            label_expected = label[:, 1:]

            model.zero_grad()

            output, _, _, _ = model(txt, label_input)

            loss = criterion(output, label_expected.reshape(-1))
            total_loss += loss.detach().item()*batch_size
    evaluate_loss = total_loss/((iteration+1)*batch_size)
    print('evaluation loss: {0}'.format(evaluate_loss))
    return evaluate_loss


###############################################################################
# training parameters
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/200k', help='location of the data corpus')
parser.add_argument('--d_model', type=int, default=256, help='size of word embeddings')
parser.add_argument('--dim_feedforward', type=int, default=1024, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--nhead', type=int, default=2, help='number of heads in the transformer encoder/decoder')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit4')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N', help='batch size128')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='num_workers')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--save', type=str, default='simple_transformer_model.pt', help='path to save the model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dataset = Corpus(args.data, 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valid_dataset = Corpus(args.data, 'valid')
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
vocab_size = len(train_dataset.dictionary)

model = Transformer(vocab_size,
                    args.d_model,
                    args.nhead,
                    args.dim_feedforward,
                    args.nlayers,
                    args.dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

try:
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader))
        for iteration, data in train_iterator:
            txt = data[0].to(device)
            label = data[1].to(device)

            optimizer.zero_grad()
            label_input = label[:, :-1]
            label_expected = label[:, 1:]

            output, _, _, _ = model(txt, label_input)

            loss = criterion(output, label_expected.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*args.batch_size
            message = f'Epoch: {epoch}/{args.epochs}, iter: {iteration+1}/{len(train_loader)}, lr: {lr_scheduler.get_last_lr()[0]}, loss: {total_loss/((iteration+1)*args.batch_size)}'
            train_iterator.set_description(message)
        evaluate(model, valid_loader, args.batch_size)    
        lr_scheduler.step()
        torch.save(model.state_dict(), args.save)
except KeyboardInterrupt:
    print('Exiting training!')
print

# Load the best saved model.
model.load_state_dict(torch.load(args.save))
test_loss = evaluate(model, valid_loader, args.batch_size)
print('\nTesting ',  '-' * 89)
print('Test loss {:5.2f}'.format(test_loss))
print('Testing ',  '-' * 89)