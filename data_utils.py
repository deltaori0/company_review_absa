from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import nltk
import re
from nltk import pos_tag
from nltk import RegexpParser
import random
import numpy as np
from torch.utils.data import random_split
import os

class Category_Classification_Dataset(Dataset):
    def __init__(self, df, tokenizer, opt, pos_encoding=False,
            pos_idx_zero=['[UNK]', '[SEP]', '[CLS]', '[PAD]', ',', '.']):
        self.category = True if opt.dataset_series == 'company' else False
        print('category: {}'.format(self.category))
        self.tokenizer = tokenizer
        if self.category:
            self.df, _ = sum_category(df)
        else:
            self.df = df
            self.tokenizer = tokenizer
        self.dataset = list()
        self.pos_vocab = dict()
        self.pos_encoding = pos_encoding
        if pos_encoding:
            nltk.download('averaged_perceptron_tagger')
        list_sentence = df.sentence
        list_polarity = df.polarity
        list_category = df.category

        self.label_map = {'negative': 0, 'positive': 1, 'neutral': 2, 'conflict': 3}

        print('{:,} samples in this dataset'.format(len(df)))


        
        i = 1
        for sentence, category, polarity in zip(list_sentence, list_category, list_polarity):
            category_words = ' '.join(category.split('#'))
            encoded = self.tokenizer.encode_plus(
                text=sentence,              # the sentence to be encoded
                text_pair=category_words, 
                add_special_tokens=True,    # Add [CLS] and [SEP]
                padding='max_length', 
                max_length=opt.max_length,  # maximum length of a sentence
                pad_to_max_length=True,     # Add [PAD]s
                return_token_type_ids=True, 
                return_tensors='pt'         # ask the function to return PyTorch tensors
                )
            
            if pos_encoding:
                tag_pair = pos_tag(self.tokenizer.convert_ids_to_tokens(encoded['input_ids'].squeeze(0)))
                list_pos = list()
                for word, pos in tag_pair:
                    if word in pos_idx_zero:
                        self.pos_vocab[word] = 0
                        list_pos.append(word)
                    else:
                        if pos in self.pos_vocab.keys():
                            list_pos.append(pos)
                        else:
                            self.pos_vocab[pos] = i
                            i += 1
                            list_pos.append(pos)
                encoded_pos = [self.pos_vocab[tag] for tag in list_pos]
                tensor_pos = torch.tensor(encoded_pos, dtype=torch.int8)
                data = {'input_ids': encoded['input_ids'], 'attention_masks': encoded['attention_mask'], 'token_type_ids': encoded['token_type_ids'],
                    'labels': self.label_map[polarity], 'pos': tensor_pos} 
            
            else:
                data = {'input_ids': encoded['input_ids'], 'attention_masks': encoded['attention_mask'], 'token_type_ids': encoded['token_type_ids'],
                        'labels': self.label_map[polarity]}
            
            self.dataset.append(data)

    def get_sample(self, idx):
        print('Sentence: {}'.format(self.df.sentence[idx]))
        print('Aspect Category: {}'.format(self.df.category[idx]))
        print('Polarity: {}'.format(self.df.polarity[idx]))
        print('Input IDs: {}'.format(self.dataset[idx]['input_ids']))
        print('Token type IDs: {}'.format(self.dataset[idx]['token_type_ids']))
        print('Encoded Label: {}'.format(self.dataset[idx]['labels']))
        if self.pos_encoding:
            print('Pos_encoding: {}'.format(self.dataset[idx]['pos']))

    def get_pos_vocab(self):
        return self.pos_vocab

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

def sum_category(df):
    '''
    for sentihood dataset
    category: price, target: LOCATION1 to category: LOCATION1#price

    @args
    df: data frame to apply
    '''
    df_old = df.copy()
    new = [df['term'][i]+'#'+df['category'][i] for i in range(len(df))]
    df['aspect category'] = new
    return df, df_old

def relative_position_encoding(sentence, target, tokenizer, max_length):
    sent = tokenizer.tokenize(sentence)
    term = tokenizer.tokenize(target)
    sent_length = len(sent)
    term_length = len(term)

    sent_space = ' '.join(sent)
    term_space = ' '.join(term)
    temp_idx = sent_space.index(term_space)
    from_idx = len(sent_space[:temp_idx-1].split()) if temp_idx != 0 else 0
    to_idx = from_idx + term_length

    center = [0] * term_length
    first_cls = [max_length]
    left = list(range(from_idx, 0, -1)) # [5, 4, 3, 2, 1]
    right = list(range(1, sent_length-to_idx+1)) # [1, 2, 3, 4, 5, 6, ...]
    pad = [max_length] * (max_length-sent_length-1)
    return torch.tensor(first_cls + left + center + right + pad, dtype=torch.int16)

def reverse_pos(dataset, max_length):
    '''
    @params
    dataset: trainset or testset to change
    max_length: max sequence length, opt.max_length
    @return
    reversed dataset
    '''
    for data in dataset:
        data['pos'] = max_length - data['pos']
    return dataset

def custom_random_split(dataset, val_ratio, random_seed, testset):
    SEED = random_seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True # ?
    torch.backends.cudnn.benchmark = False # ?
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    num_val = int(len(dataset) * val_ratio)
    num_train = len(dataset) - num_val
    train_set, val_set = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(SEED))
    print('Ratio of datasets {} : {} : {}'.format(len(train_set), len(val_set), len(testset)))

    return train_set, val_set, testset

def preprocess(text):
    text = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', text)
    print('text', text)
    return text

def clean_sentence(df, clean_func):
    df['sentence'] = df['sentence'].apply(clean_func)
    return df