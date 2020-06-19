import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, ChainDataset
from torchtext.data.functional import generate_sp_model, load_sp_model

from transformers import AlbertTokenizer

class MultitaskDataset(ChainDataset):
    pass

class CMQA(torch.utils.data.Dataset):
    def __init__():
        pass

class BioASQ(torch.utils.data.Dataset):
    def __init__():
        pass

class MEDIQA(torch.utils.data.Dataset):
    def __init__():
        pass


class RandomDataset(Dataset):

    def __init__(self, shape):
        """Pass shape tuple in to be unpacked"""
        self.len = shape[0]
        self.data =  torch.empty(*shape, dtype=torch.long).random_(10)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class DataProcessor():

    def train_tokenizer(self, path):
        """Given path to dataset, tokenize the dataset"""
        sp = generate_sp_model(path, vocab_size=23456, model_prefix='spm_user')
        return sp

    def load_tokenizer(self, path):
        """Load the trained tokenizer"""
        if args.init_bert_weights:
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        else:
            tokenizer = load_sp_model(path)
            
        return tokenizer

    def get_data(self):
        pass
