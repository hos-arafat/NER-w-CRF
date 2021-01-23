import os
import pickle
import sys
from argparse import ArgumentParser


import nltk
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("Found NLTK POS Tagger!")
except LookupError:
    print("Did not find NLTK POS Tagger, downloading it...")
    nltk.download('averaged_perceptron_tagger')
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

class PreProcessor():
    """Utility class to preprocess the train, dev, test files."""
    def __init__(self, mode):
        super(PreProcessor).__init__()
        """
        Args:
            mode: train, test, or dev or submit if we are running "implementation.py"
        """ 

        self.mode = mode

        if self.mode == "train":
            self.tsv_file = "train.tsv"

        if self.mode == "dev":
            self.tsv_file = "dev.tsv"
        
        if self.mode == "test":
            self.tsv_file = "test.tsv"
        
        # Path to the various input, POS, and label tokenizers
        self.word_to_int_path = os.path.join("./model/", "word_to_int_dictionary.pickle")
        self.int_to_word_path = os.path.join("./model/", "int_to_word_dictionary.pickle")

        self.pos_to_int_path = os.path.join("./model/", "pos_to_int_dictionary.pickle")
        self.int_to_pos_path = os.path.join("./model/", "int_to_pos_dictionary.pickle")

        self.label_to_int_path = os.path.join("./model/", "labels_to_int_dictionary.pickle")
        self.int_to_label_path = os.path.join("./model/", "int_to_labels_dictionary.pickle")

        self.training_tokens = []


    def read_data(self, parent):
        """
        Function that reads data file and returns list of lists 
        representing the input sentences and labels in the file

        Args: path to .tsv file
        
        Returns: None
                        
        """ 
        

        print("{:} Dataset will be built from the following files {:}...".format(self.mode, self.tsv_file))

        self.list_l_tokens = []
        self.list_l_labels = []


        all_labels = set() 

        # Read file content
        with open((os.path.join(parent, self.tsv_file)), 'r', encoding='utf-8') as f:
            f_content_1 = f.readlines()
            for line in f_content_1:

                # All sentences start with the '#' token  
                if line[0] == "#":

                    self.list_l_tokens.append(line[1:].split())
                    

                    # Labels for each line = number of words in the line
                    line_length = len(line[1:].split())


                    list_labels = []
                    list_tokens = []

                # Line for each label starts with a digit 
                # and has the following form '23	Mediterranean	LOC' 
                # Add all labels for sentence to list of labels
                elif line[0].isdigit():
                    num = int(line.split()[0])
                    
                    token = line.split()[1]


                    label = line.split()[-1]
                    all_labels.add(label)

                    # Add all words to a list to build input vocab
                    self.training_tokens.append(token) if self.mode == "train" else None
                    list_labels.append(label)

                    if num == (line_length-1):
                        self.list_l_labels.append(list_labels)

        return 


    def create_pos_data(self, test_data):
        """
        Function that reads data file and returns list of lists 
        representing the POS tags for every word in the file

        Args: 
            test_data: list of lists of words, if we are running on data from "implementation.py"
        
        Returns: 
            list_l_pos: list of lists of POS tags        
        """ 

        self.all_pos_tags = set()
        self.list_l_pos = list()

        prediction = list()

        # If we are running the implentation.py file, 
        # list of words will be supplied to this function,
        # otherwise, function returns POS tags of "self.list_l_tokens"
        if test_data is None:
            # If we already predicted the POS tags for this file before
            # load POS tags from file (faster)
            pos_data_path = "./data/pos_{}.utf8".format(self.mode)
            if os.path.exists(pos_data_path):
                print("found file ! Getting pos data from file")
                with open(pos_data_path, "r", encoding="utf8") as f:
                    f_content_1 = f.readlines()
                    for l in f_content_1:
                        # print(l)
                        pos_sentence = l.split()
                        self.list_l_pos.append(pos_sentence)
                        if self.mode == "train":
                            for pos in pos_sentence:
                                self.all_pos_tags.add(pos)

                        
            # If not, predict POS Tags using NLTK POS Tagger
            # and right them to file so we don't predict each time
            else:
                print("Did not find file with POS sentences")
                print("Predicting...")
                prediction = (nltk.pos_tag_sents(self.list_l_tokens)) 
                print("Done!")
                print("Writing to file")
                with open(pos_data_path, "w", encoding="utf8") as f:
                    for p in prediction:
                        # print("p is", p)
                        pos_list = []
                        for elem in p:
                            # print("elem is", elem)
                            pos = elem[1]
                            pos_list.append(pos)
                            self.all_pos_tags.add(pos) if self.mode == "train" else None
                            f.write(pos + " ")
                        self.list_l_pos.append(pos_list)
                        f.write("\n")
        else:
            # We are running implentation.py and have to predict tags
            # for the secrete dataset / tokens supplied 
            print("Predicting POS tags for test set...")
            prediction = (nltk.pos_tag_sents(test_data)) 
            print("Done!")

            for l in prediction:
                pos_list = []
                for tup in l:
                    pos_list.append(tup[1])

                self.list_l_pos.append(pos_list)

        return self.list_l_pos

    def save_tokenizer(self, tokenizer_type, token_to_int, int_to_token):
        """
        Function that saves the input OR pos OR label tokenizer

        Args: tokenizer_type: "input" OR "pos" OR "label"
        
        Returns: None        
        """ 
        if tokenizer_type == "input":
            token_to_int_path = self.word_to_int_path
            int_to_token_path = self.int_to_word_path
        elif tokenizer_type == "pos":
            token_to_int_path = self.pos_to_int_path
            int_to_token_path = self.int_to_pos_path
        elif tokenizer_type == "label":
            token_to_int_path = self.label_to_int_path
            int_to_token_path = self.int_to_label_path

        with open(token_to_int_path, "wb") as handle:
            pickle.dump(token_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(int_to_token_path, "wb") as handle:
            pickle.dump(int_to_token, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return

    def create_tokenizer(self, tokenizer_type):
        """
        Function that creates the input OR POS tags OR label tokenizer
        and saves them to disk

        Args: 
            tokenizer_type: "input" OR "pos" OR "label"
        
        Returns: 
            token_to_int: dict that maps from token to and integer        
            int_to_token: dict that maps from integer to token        
        """ 

        list_l_tokens = list()
        token_to_int = dict()

        if tokenizer_type == "input":
            list_l_tokens = self.training_tokens

            token_to_int = {ni: indi for indi, ni in enumerate(set(list_l_tokens), start=2)}
            token_to_int["<pad>"] = 0
            token_to_int["<unk>"] = 1

        elif tokenizer_type == "pos":
            
            token_to_int = {ni: indi for indi, ni in enumerate(self.all_pos_tags, start=2)}
            token_to_int["<pad>"] = 0
            token_to_int["<unk>"] = 1


        elif tokenizer_type == "label":
            token_to_int = {"PER": 1, "ORG": 2, "LOC": 3, "O": 4}
            token_to_int["<pad>"] = 0


        int_to_token = {v: k for k, v in token_to_int.items()}   

        self.save_tokenizer(tokenizer_type, token_to_int, int_to_token)

        return token_to_int, int_to_token

    def create_all_tokenizers(self):
        """
        Function that creates the input AND POS tags AND label tokenizers

        Args: None
        
        Returns: 
            token_to_int: dict that maps from 'token' to and integer        
            int_to_token: dict that maps from integer to 'token'      
            where 'token' is input words, POS tags, and labels  
        """ 
        return self.create_tokenizer("input"), self.create_tokenizer("pos"), self.create_tokenizer("label")


    def load_tokenizer(self, tokenizer_type):
        """
        Function that loads the input OR pos OR label tokenizer

        Args: 
            tokenizer_type: "input" OR "pos" OR "label"
        
        Returns: 
            token_to_int: dict that maps from token to and integer        
            int_to_token: dict that maps from integer to token        
        """ 
        if tokenizer_type == "input":
            token_to_int_path = self.word_to_int_path
            int_to_token_path = self.int_to_word_path
        elif tokenizer_type == "pos":
            token_to_int_path = self.pos_to_int_path
            int_to_token_path = self.int_to_pos_path
        elif tokenizer_type == "label":
            token_to_int_path = self.label_to_int_path
            int_to_token_path = self.int_to_label_path

        with open(token_to_int_path, 'rb') as handle:
            token_to_int = pickle.load(handle)
        with open(int_to_token_path, 'rb') as handle:
            int_to_token = pickle.load(handle)   
        
        return token_to_int, int_to_token

    def load_all_tokenizers(self):
        """
        Function that loads the input AND POS AND label tokenizers from disk

        Args: None
        
        Returns: 
            token_to_int: dict that maps from 'token' to and integer        
            int_to_token: dict that maps from integer to 'token'      
            where 'token' is input words, POS tags, and labels  
        """ 
        return self.load_tokenizer("input"), self.load_tokenizer("pos"), self.load_tokenizer("label")

    def encode_text(self, text_type, test_data):
        """
        Function that encods input words OR POS tags OR labels to torch.tensors

        Args: 
            text_type: "input" OR "pos" OR "label"
            test_data: list of lists of words, if we are running on data from "implementation.py"
        
        Returns: 
            data_idx: list of torchTensors representing the encoded data
        """ 
        
        token_to_int = dict()
        list_l_tokens = list()

        if text_type == "input":
            token_to_int, _ = self.load_tokenizer("input")
            if self.mode == "train" or self.mode == "dev" or self.mode == "test":
                list_l_tokens = self.list_l_tokens
            elif self.mode == "submit":
                list_l_tokens = test_data

        elif text_type == "pos":
            token_to_int, _ = self.load_tokenizer("pos")
            list_l_tokens = self.list_l_pos
        elif text_type == "label":
            token_to_int, _ = self.load_tokenizer("label")
            list_l_tokens = self.list_l_labels
                    
        # data is the text converted to indexes, as list of lists
        data = []
        # for each sentence
        for sentence in list_l_tokens:
            paragraph = []
            # for each token in the sentence, map it to and int 
            # using its corresponding input/pos/label tokenizer
            for i in sentence:
                id_ = token_to_int[i] if i in token_to_int else token_to_int["<unk>"]
                paragraph.append(id_)
            paragraph = torch.LongTensor(paragraph)
            data.append(paragraph)

        return data


    def encode_data(self, include_pos=False, test_data=None):
        """
        Function that encods input words AND POS tags AND labels to torch.tensors

        Args: 
            include_pos: whether or not to encode POS data as well
            test_data: list of lists of words, if we are running on data from "implementation.py"
        
        Returns: 
            tuple of lists of torchTensors representing the encoded data
        """ 
        
        # If we are NOT running "implementation.py"
        # and including or NOT including POS Tags data
        if self.mode == "train" or self.mode == "dev" or self.mode == "test":
            test_data = None
            if include_pos == True:
                return self.encode_text("input", test_data), self.encode_text("pos", test_data), self.encode_text("label", test_data) 
            elif include_pos == False:
                return self.encode_text("input", test_data), None, self.encode_text("label", test_data) 
        
        # If we are running "implementation.py", there are no labels
        # and again we can include or NOT include POS Tags data    
        # 
        # We return "None" in place of any non-existant POS tag or label data
        # this allows the dataloader class to create a batch that includes the write elements
        # (inputs only...inputs and pos only...inputs,pos, and labels, and so on)    
        elif self.mode == "submit":
            if include_pos == True:
                return self.encode_text("input", test_data), self.encode_text("pos", test_data), None 
            elif include_pos == False:
                return self.encode_text("input", test_data), None, None



class NERDataset(Dataset):
    """
    Class that creates a Named Entity Recognition Dataset
    """ 
    def __init__(self, **kwargs):
        """    
        Args: 
            x: list of torchTensors representing the input
            p: list of torchTensors representing the POS tags
            y: list of torchTensors representing the labels
        
        Returns: 
            tuple or dict of torchTensors
        """
        self.encoded_data = kwargs["x"]
        self.encoded_pos = kwargs["p"]
        self.encoded_labels = kwargs["y"]

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        t = []
        if self.encoded_data is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce index_dataset on this object
            before trying to retrieve elements. In case you want to retrieve raw
            elements, use the method get_raw_element(idx)""")

        if self.encoded_labels is None:
            # data we get in "implementation.py" has no labels
            if self.encoded_pos is None:
                # We experiment with and without feeding POS tags to model
                return ({"inputs": self.encoded_data[idx]})
            else:
                return ({"inputs":self.encoded_data[idx], "pos":self.encoded_pos[idx]})
        else:
            if self.encoded_pos is None:
                return (self.encoded_data[idx], self.encoded_labels[idx])
            else:
                return (self.encoded_data[idx], self.encoded_labels[idx], self.encoded_pos[idx])


def create_dataset(dataset_type, opts, test_data):
    """
    Function that creates train-dev-test NERDataset

    Args: 
        dataset_type: train or dev or test or submit if we are running in "implementation.py"
        opts: dictionary outlining various options including if we want POS tags or not
        test_data: list of lists of words, if we are running on data from "implementation.py"
    
    Returns: 
        dataset: NERDataset instance
        encoded_labels: list of torchTensors representing encoded POS tags
        encoded_labels: list of torchTensors representing encoded labels
    """ 

    include_pos = False

    parent = "./data"

    p = PreProcessor(dataset_type)

    # If we are NOT running "implementation.py", we read the data from file
    if dataset_type == "train" or dataset_type == "dev" or dataset_type == "test":
        p.read_data(parent)    

    # We can include or exclude POS Tags data
    if opts["use_pos_embeddings"] == True:
        include_pos = True
        p.create_pos_data(test_data=test_data)

    # Encode all the data to a list of torchTensors
    encoded_tokens, encoded_tokens_pos, encoded_labels = p.encode_data(include_pos=include_pos, test_data=test_data)

    # Build an NER Dataset
    dataset = NERDataset(x=encoded_tokens, p=encoded_tokens_pos, y=encoded_labels)
    print("{} dataset size is {}".format(dataset_type, len(dataset)))

    if dataset_type == "train" or dataset_type == "dev" or dataset_type == "test":
        return dataset, encoded_tokens_pos, encoded_labels
    elif dataset_type == "submit":
        return dataset
    