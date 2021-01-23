import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils import data

from model import Model
from stud.ner_models import NERModel, NERModel_CRF
from stud.preprocess_dataset import PreProcessor, create_dataset
from stud.train_utils import opts, HParams
from stud.utils import collate_test_batch

# Use this function in this file's main to pass data
# and try out this file locally first
# from implement_utils import read_data

np.random.seed(opts["random_seed"])
torch.manual_seed(opts["random_seed"])

def build_model(device: str) -> Model:
    # STUDENT: 
    return StudentModel(device)
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()

def reconstruct_network_output(list_predictions, original_len):
    print("Reconstructing the Network's Prediction....")
    reconst = []                    
    var = 0
    for c_idx in range(len(original_len)):
        reconst.append(list_predictions[var:var+original_len[c_idx]])
        var += original_len[c_idx]

    return reconst


class RandomBaseline(Model):

    options = [
        ('LOC', 98412),
        ('O', 2512990),
        ('ORG', 71633),
        ('PER', 115758)
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]

class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device):
        
        self.device = device
        self.crf = opts["use_crf"]

        self.parse = PreProcessor("submit")
        
        self.include_pos = False 
        if opts["use_pos_embeddings"] == True:
            self.include_pos = True

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:

        # parse original length of input tokens
        # to reconstruct them later
        original_len = np.asarray([len(x) for x in tokens])
        print("original len is", original_len)
        # original_len = np.insert(original_len, 0, 0)
        
        # create dataloader out of input tokens
        dataset = create_dataset("submit", opts, tokens)
        test_dataset = data.DataLoader(dataset, collate_fn=collate_test_batch, batch_size=32)

        all_predictions = list()     
        read_predictions = list()
        before_reconst_predictions = list()

        # Load input, pos, and label tokenizers
        (vocabulary, decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = self.parse.load_all_tokenizers()

        # Define model and hyperparameters
        load_params = HParams(vocabulary, pos_vocabulary, label_vocabulary, opts)

        if self.crf == True:
            ner_model = NERModel_CRF(load_params)
        else:
            ner_model = NERModel(load_params)
    
        save_path = opts["save_model_path"]
        epoch = opts["epochs"]

        print("Number of Epochs", epoch)
        print("ckpt to load", os.path.join(save_path, 'state_{}.pth'.format(epoch-1)))

        state_dict = torch.load(os.path.join(save_path, 'state_{}.pth'.format(epoch-1)), map_location=self.device)
        ner_model.load_state_dict(state_dict)
        ner_model.to(self.device)
        ner_model.eval()

        indexed_pos = None

        for indexed_elem in test_dataset:

            indexed_in = indexed_elem[0].to(self.device)
            print("shape of index_in is", indexed_in.shape)
            mask = indexed_elem[1].to(self.device)
            if self.include_pos == True:
                indexed_pos = indexed_elem[2].to(self.device)

            if self.crf == True:
                # CRF returns predictions as lists of list 
                # thanks to the "mask" it takes as input
                predictions = ner_model.decode_crf(indexed_in, indexed_pos, mask)

                # Simply convert those predictions from a list of list of ints
                # to a list of lists of readable labels
                mapper = lambda x: [int_to_label[w] for w in x]
                mapped_predictions = list(map(mapper, predictions))

                # Add them to the final list of readable predictions we will return
                read_predictions.extend(mapped_predictions)
        
                
            elif self.crf == False:
                # Model returns predictions as one long list
                predictions = ner_model(indexed_in, indexed_pos)
                predictions = torch.argmax(predictions, -1).view(-1)

                valid_indices = predictions != 0    

                valid_predictions = predictions[valid_indices]

                mapper = lambda x: int_to_label[x]
                mapped_predictions = list(map(mapper, valid_predictions.tolist()))

                before_reconst_predictions.extend(mapped_predictions)
        
        if self.crf == False:
            
            print("calling recon func")
            read_predictions = reconstruct_network_output(before_reconst_predictions, original_len)


        return read_predictions
