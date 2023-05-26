import numpy as np
import torch
from torchvision import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import preprocessing
from collections import Counter
import nltk
import numpy as np
nltk.download('punkt')
from datasets.parse_xml import parse_SemEval
from transformers import BertTokenizer
import random 

class Data_BERT:
        
    def __init__(self, handler, X_train, Y_train, X_test, Y_test):

        # initialize variables
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.max_len = 0
        
        sentences = np.concatenate([self.X_train, self.X_test])

        # For every sentence...
        for sent in sentences:

            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)

            # Update the maximum sentence length.
            self.max_len = max(self.max_len, len(input_ids))

        #print('Max sentence length: ', self.max_len)

        self.input_ids_train, self.attention_masks_train, self.labels_train = self.tokenize(X_train, Y_train)
        self.input_ids_test, self.attention_masks_test, self.labels_test = self.tokenize(X_test, Y_test)

    def tokenize(self, X, Y):

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in X: #sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
        
            encoded_dict = self.tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = self.max_len,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                            )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(Y)

        return input_ids, attention_masks, labels

    def initialize_labels(self, num, is_active_learning):
        # generate initial labeled pool
        if is_active_learning:
            tmp_idxs = np.arange(self.n_pool)
            np.random.shuffle(tmp_idxs)
            self.labeled_idxs[tmp_idxs[:num]] = True
        else:
            self.labeled_idxs[np.arange(self.n_pool)] = True
    
    def update_n_pool(self):
        self.n_pool = self.input_ids_train.size(dim=0) #len(self.X_train) 

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        training_idxs = labeled_idxs[:int(len(labeled_idxs)*0.8)]
 
        return labeled_idxs, self.input_ids_train[labeled_idxs], self.attention_masks_train[labeled_idxs], self.labels_train[labeled_idxs]
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.input_ids_train[unlabeled_idxs], self.attention_masks_train[unlabeled_idxs], self.labels_train[unlabeled_idxs]
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.input_ids_train, self.attention_masks_train, self.labels_train
        
    def get_test_data(self):
        return self.input_ids_test, self.attention_masks_test, self.labels_test
    
    def cal_test_acc(self, preds):
        return 1.0 * np.sum(self.Y_test == preds) / self.n_test
    
    def cal_test_precision(self, preds):
        return precision_score(self.Y_test, preds)
    
    def cal_test_recall(self, preds):
        return recall_score(self.Y_test, preds)
    
    def cal_test_f1(self, preds):
        return f1_score(self.Y_test, preds)


def get_SemEval_Restaurants(handler):

    train_data = pd.read_csv("datasets/SemEval/Restaurants_Train.csv")
    test_data = pd.read_csv("datasets/SemEval/Restaurants_Test.csv")

    train_data = train_data.dropna()
    test_data = test_data.dropna()
    train_data = train_data[train_data['sentiment'] != 'neutral']
    test_data = test_data[test_data['sentiment'] != 'neutral']

    train_data["sentence"] = train_data["sentence"] +" "+ train_data["category"]
    test_data["sentence"] = test_data["sentence"] +" "+ test_data["category"] 

    train_data['sentiment'] = train_data['sentiment'].str.lower()
    test_data['sentiment'] = test_data['sentiment'].str.lower()
    train_data['sentence'] = train_data['sentence'].str.lower()
    test_data['sentence'] = test_data['sentence'].str.lower()

    X_train = train_data['sentence'].values
    X_test = test_data['sentence'].values

    y_train = train_data['sentiment'].values
    y_test = test_data['sentiment'].values

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return Data_BERT(handler, X_train, y_train, X_test, y_test)

def get_SemEval_Laptops(handler):

    train_data = pd.read_csv("datasets/SemEval/Laptops_Train.csv")
    test_data = pd.read_csv("datasets/SemEval/Laptops_Test.csv")

    train_data = train_data.dropna()
    test_data = test_data.dropna()
    train_data = train_data[train_data['sentiment'] != 'neutral']
    test_data = test_data[test_data['sentiment'] != 'neutral']

    train_data["sentence"] = train_data["sentence"] +" "+ train_data["category"]
    test_data["sentence"] = test_data["sentence"] +" "+ test_data["category"] 

    train_data['sentiment'] = train_data['sentiment'].str.lower()
    test_data['sentiment'] = test_data['sentiment'].str.lower()
    train_data['sentence'] = train_data['sentence'].str.lower()
    test_data['sentence'] = test_data['sentence'].str.lower()

    X_train = train_data['sentence'].values
    X_test = test_data['sentence'].values

    y_train = train_data['sentiment'].values
    y_test = test_data['sentiment'].values

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return Data_BERT(handler, X_train, y_train, X_test, y_test)
