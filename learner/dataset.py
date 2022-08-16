import time
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader

from aae_utils.utils import CharVocab, string2tensor, property2tensor
from utils.filesystem import load_dataset
from .skipgram import Vocab


class DataCollator:
    def __init__(self, vocab):
        self.vocab = vocab

    def merge(self, sequences):
        '''
        Sentence to indices, pad with zeros
        '''
        # print('before:')
        # print(sequences)
        # sequences = sorted(sequences, key=len, reverse=True)# YL: choose not to sort it, because I need properties in same order
        # print('after:')
        # print(sequences)
        lengths = [len(seq) for seq in sequences]
        padded_seqs = np.full((len(sequences), max(lengths)), self.vocab.PAD)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return torch.LongTensor(padded_seqs), lengths

    def __call__(self, data):
        # seperate source and target sequences
        # if batch is 3
        # data: [([1, 6127, 2701], [6127, 2701, 2], array([0.8318088 , 2.02817336, 3.34902   ])),
        #       ([1, 31928, 269, 1036, 44], [31928, 269, 1036, 44, 2], array([0.86020293, 3.05513613, 3.9534    ])), 
        #       ([1, 850, 4212, 769], [850, 4212, 769, 2], array([0.51382926, 1.96542485, 2.44458   ]))]

        src_seqs, tgt_seqs, properties = zip(*data)  #

        # now src_seqs, tgt_seqs, properties are in separate lists.
        # print('data:')
        # print(data)
        # print('\n------------------------------------------------------------------\n')
        # print('src_seqs:')
        # print(src_seqs)
        # print('tgt_seqs:')
        # print(tgt_seqs)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_seqs, src_lengths = self.merge(src_seqs)
        tgt_seqs, tgt_lengths = self.merge(tgt_seqs)
        properties = torch.tensor(properties, dtype=torch.float)
        return src_seqs, tgt_seqs, src_lengths, properties


class FragmentDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, config, kind='train', data=None):
        """
        Reads source and target sequences from a csv file or given variable data_given.
        kind: string, from {'train', 'test', 'given'}
        data: pd array, having the following fields: smiles, fragments, n_fragments, C, F, N, O, Other, 
                                                     SINGLE, DOUBLE, TRIPLE, Tri, Quad, Pent, Hex,
                                                     logP, mr, qed, SAS

        """
        self.config = config

        if kind != 'given':
            data = load_dataset(config, kind=kind)
            # data = data[0:20000]

        # the following is for test data, because there are n_fragments=0 / 1 
        min_nfrag = 2
        data = data[data.n_fragments >= min_nfrag]

        self.data = data.reset_index(drop=True)
        self.size = self.data.shape[0]

        # print('data:',self.data.SAS)

        self.vocab = None

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        seq = self.data.fragments[index].split(" ")
        seq = self.vocab.append_delimiters(seq)
        src = self.vocab.translate(seq[:-1])  # include '<SOS>' but  do not include '<EOS>'
        tgt = self.vocab.translate(seq[1:])  # do not include '<SOS>', but do not include '<EOS>'

        properties = np.array([0, 0, 0], dtype=float)
        properties[0] = self.data.qed[index]
        properties[1] = self.data.SAS[index]
        properties[2] = self.data.logP[index]

        return src, tgt, properties

    def __len__(self):
        return self.size

    def get_loader(self):
        start = time.time()
        collator = DataCollator(self.vocab)
        loader = DataLoader(dataset=self,
                            collate_fn=collator,
                            batch_size=self.config.get('batch_size'),
                            num_workers=24,
                            shuffle=True)
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Data loaded. Size: {self.size}. '
              f'Time elapsed: {elapsed}.')
        return loader

    def get_vocab(self):
        start = time.time()
        if self.vocab is None:
            try:
                self.vocab = Vocab.load(self.config)
            except Exception:
                self.vocab = Vocab(self.config, self.data)

        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Vocab created/loaded. '
              f'Size: {self.vocab.get_size()}. '
              f'Effective size: {self.vocab.get_effective_size()}. '
              f'Time elapsed: {elapsed}.')

        return self.vocab

    def change_data(self, data):
        self.data = data
        self.size = self.data.shape[0]


class SMILESDataset(Dataset):
    def __init__(self, config, kind='train', data=None):
        self.config = config
        self.fieldnames = ['smiles', 'qed', 'SAS', 'logP']
        if kind != 'given':
            self.data = load_dataset(config, kind=kind)
        else:
            self.data = data
        '''Temp Implementation (A1)'''
        subsets = config.get('subsets')
        if subsets > 1:
            end = min((config.get('batch_size') * subsets), len(self.data))
            self.data = self.data.loc[:end]
        '''End Point'''
        self.data.reset_index(drop=True, inplace=True)
        self.data.drop(columns=['mr'], inplace=True)
        self.vocab = CharVocab(self.data['smiles'])

    def get_data(self):
        return self.data

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.data)

    def get_loader(self, shuffle=True):
        def aae_get_collate_fn():
            def collate(data):
                properties = torch.tensor([p[1:] for p in data], dtype=torch.float)
                tensors = [string2tensor(string[0], self.vocab) for string in data]
                lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long)
                encoder_inputs = pad_sequence(tensors,
                                              batch_first=True,
                                              padding_value=self.vocab.pad)
                encoder_input_lengths = lengths - 2

                decoder_inputs = pad_sequence([t[:-1] for t in tensors],
                                              batch_first=True,
                                              padding_value=self.vocab.pad)
                decoder_input_lengths = lengths - 1

                decoder_targets = pad_sequence([t[1:] for t in tensors],
                                               batch_first=True,
                                               padding_value=self.vocab.pad)
                decoder_target_lengths = lengths - 1

                return (encoder_inputs, encoder_input_lengths), \
                       (decoder_inputs, decoder_input_lengths), \
                       (decoder_targets, decoder_target_lengths), properties

            return collate

        start = time.time()
        collator = aae_get_collate_fn()
        loader_data = self.data.loc[:, self.fieldnames].to_numpy()
        loader = DataLoader(dataset=loader_data,
                            batch_size=self.config.get('batch_size'),
                            shuffle=shuffle,
                            collate_fn=collator)
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Data loaded. Size: {len(self.data)}. '
              f'Time elapsed: {elapsed}.')
        return loader

    def change_data(self, data):
        self.data = data
        self.vocab = CharVocab(self.data['smiles'])
