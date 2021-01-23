import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from ner_models import NERModel, NERModel_CRF
from preprocess_dataset import PreProcessor, create_dataset
from trainer import Trainer
from utils import collate_labelled_batch, compute_precision, plot_conf
from utils import parse_embeddings_text_file, create_pretrained_embeddings, load_pretrained_embeddings
from train_utils import opts, HParams

# Set the random seed to be able to reproduce results when needed
random.seed(opts['random_seed'])
np.random.seed(opts['random_seed'])
torch.manual_seed(opts['random_seed'])

p = PreProcessor("train")

# Load tokenizers for input, pos, and labels. If they don't exist, create them
if os.path.exists(p.word_to_int_path):
    (vocabulary, decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = p.load_all_tokenizers()
else:
    p.read_data("../../data")
    p.create_pos_data(test_data=None)
    (vocabulary, decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = p.create_all_tokenizers()

# Create train and dev datasets
train_dataset, _, _ = create_dataset(dataset_type="train", opts=opts, test_data=None)
train_dataloader = data.DataLoader(train_dataset, collate_fn=collate_labelled_batch, batch_size=256)


dev_dataset, _, encoded_dev_labels = create_dataset(dataset_type="dev", opts=opts, test_data=None)
dev_dataloader = data.DataLoader(dev_dataset, collate_fn=collate_labelled_batch, batch_size=256)

print("\nVocab size",len(vocabulary))
print("POS Vocab size",len(pos_vocabulary))
print("Label Vocab size",len(label_vocabulary))

# Define hyperparameters
params = HParams(vocabulary, pos_vocabulary, label_vocabulary, opts)

if opts['use_glove_embeddings'] == True:

    
    glove_txt_file = "./resources/glove.840B.300d/glove.840B.300d.txt"

    glove_word2embed_dict_path = "./model/glove.840B.300D.pickle"
    
    glove_embeddings_npy_path = "./model/glove_embeddings.npy"

    # If embeddings are saved as NPY files, load them. If not, create them
    if not os.path.exists(glove_embeddings_npy_path):
        glove_word2embed_dict = parse_embeddings_text_file(glove_txt_file, glove_word2embed_dict_path)

        pretrained_embeddings = create_pretrained_embeddings("glove", glove_word2embed_dict, vocabulary, decode, opts["glove_embedding_dim"])
    else:
        pretrained_embeddings = load_pretrained_embeddings(glove_embeddings_npy_path).to(opts["device"])

    # Set model embeddings equal to the pretrained embeddings
    params.embeddings    = pretrained_embeddings


if opts['use_pretrained_pos_embeddings'] == True:


    pos_txt_file = "./model/pos_embeddings"
    pos2embed_dict_path = "./model/pos.300D.pickle"


    pos_embeddings_npy_path = "./model/pos_embeddings.npy"

    # If POS embeddings are saved as NPY files, load them. If not, create them
    if not os.path.exists(pos_embeddings_npy_path):
        pos2embed_dict = parse_embeddings_text_file(pos_txt_file, pos2embed_dict_path)

        pretrained_pos_embeddings = load_pretrained_embeddings("pos", pos2embed_dict, pos_vocabulary, int_to_pos, opts["pos_embedding_dim"])
    else:
        pretrained_pos_embeddings = load_pretrained_embeddings(pos_embeddings_npy_path).to(opts["device"])

    # Set model embeddings equal to the pretrained embeddings
    params.pos_embeddings    = pretrained_pos_embeddings


# Create the model according to the architecture defined in "opts" dictionary
if opts['use_crf'] == True:

    print("\n\nBiLSTM-CRF")
    ner_model = NERModel_CRF(params).cuda()
    if opts['use_pos_embeddings'] == True:
        print("with POS embeddings")
    if opts['use_glove_embeddings'] == True:
        print("with GloVe embeddings")

elif opts['use_crf'] == False:

    print("\n\nBiLSTM")
    ner_model = NERModel(params).cuda()
    if opts['use_pos_embeddings'] == True:
        print("with POS embeddings")
    if opts['use_glove_embeddings'] == True:
        print("with GloVe embeddings")


# Create Trainer
trainer = Trainer(
    model         = ner_model,
    loss_function = nn.CrossEntropyLoss(ignore_index=label_vocabulary['<pad>']),
    optimizer     = optim.Adam(ner_model.parameters(), lr=opts['learning_rate']),
    label_vocab   = label_vocabulary,
    is_CRF        = opts['use_crf'],
    use_pos       = opts['use_pos_embeddings'], 
    clip_grads    = opts['grad_clipping'],
    early_stop    = opts['early_stopping'],
    device        = opts["device"]
)

# Create folder to save the model in
if not os.path.exists(opts["save_model_path"]):
    os.makedirs(opts["save_model_path"])


print("Number of Epochs", opts["epochs"])

trainer.train(train_dataloader, dev_dataloader, opts)

# Create file to write all the model hyperparameters and training cofiguration in
hyperParams_file = open("{}/hyper-parameters.txt".format(opts["save_model_path"]), "a", encoding="utf8")
for (option, value) in opts.items():
    hyperParams_file.write("{}: {}\n".format(option, value))

# Evaluate model by computing precision, confusion matrix and recall & F-score on dev set
precisions = compute_precision(ner_model, dev_dataloader, label_vocabulary, encoded_dev_labels, opts)
per_class_precision = precisions["per_class_precision"]
print("Micro Precision: {}\nMacro Precision: {}".format(precisions["micro_precision"], precisions["macro_precision"]))
print("Per class Precision:")
for idx_class, precision in sorted(enumerate(per_class_precision), key=lambda elem: -elem[1]):
    label = int_to_label[(idx_class)] if idx_class != 0 else int_to_label[(idx_class)]
    print(label, precision)
print("Rcall: {}\n\nF-1 score: {}\n\n".format(precisions["recall"], precisions["f1"]))

# Print, plot, and save the confusion matrix
print("Confusion matrix\n", precisions["confusion_matrix"])
plot_conf("DevSet_Confusion_Matrix", precisions["confusion_matrix"], opts["save_model_path"])

