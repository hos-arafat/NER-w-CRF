import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from ner_models import NERModel, NERModel_CRF
from plot_embeddings import visualize_model_embeddings
from preprocess_dataset import PreProcessor, create_dataset
from trainer import Trainer
from train_utils import opts, HParams
from utils import collate_labelled_batch, compute_precision, plot_conf
from utils import parse_embeddings_text_file, load_pretrained_embeddings


random.seed(opts['random_seed'])
np.random.seed(opts['random_seed'])
torch.manual_seed(opts['random_seed'])

# Load or create vocbulary from tokenizers
p = PreProcessor("train")

if os.path.exists(p.word_to_int_path):
    (vocabulary, decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = p.load_all_tokenizers()
else:
    (vocabulary, decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = p.create_all_tokenizers()

# Load train, dev, test datasets
train_dataset, _ = create_dataset(dataset_type="train", opts=opts, test_data=None)
train_dataloader = data.DataLoader(train_dataset, collate_fn=collate_labelled_batch, batch_size=256)


dev_dataset, encoded_dev_labels = create_dataset(dataset_type="dev", opts=opts, test_data=None)
dev_dataloader = data.DataLoader(dev_dataset, collate_fn=collate_labelled_batch, batch_size=256)

test_dataset, encoded_test_labels = create_dataset(dataset_type="test", opts=opts, test_data=None)
test_dataloader = data.DataLoader(test_dataset, collate_fn=collate_labelled_batch, batch_size=256)

print("\nVocab size",len(vocabulary))
print("POS Vocab size",len(pos_vocabulary))
print("Label Vocab size",len(label_vocabulary))

# Load Hyperparameters
params = HParams(vocabulary, pos_vocabulary, label_vocabulary, opts)

# Load Glove embeddings if specified in options dict
if opts['use_glove_embeddings'] == True:

    # print("Initializing word embeddings from GloVe")
    
    glove_txt_file = "./model/glove.840B.300D/glove.840B.300D.txt"

    glove_word2embed_dict_path = "./model/glove.840B.300D.pickle"
    

    glove_word2embed_dict = parse_embeddings_text_file(glove_txt_file, glove_word2embed_dict_path)

    pretrained_embeddings = load_pretrained_embeddings("glove", glove_word2embed_dict, vocabulary, decode, opts["glove_embedding_dim"])

    # print("Glove Pretrained embeddings shape", pretrained_embeddings.shape)

    params.embedding_dim = opts["glove_embedding_dim"]
    params.embeddings    = pretrained_embeddings
    params.vocab_size    = len(vocabulary)

# Load POS embeddings if specified in options dict
if opts['use_pretrained_pos_embeddings'] == True:

    # print("\nInitializing POS embeddings from pretrained")

    pos_txt_file = "./model/pos_embeddings"
    pos2embed_dict_path = "./model/pos.300D.pickle"

    # print("POS Embedding dimension is", opts["pos_embedding_dim"])

    pos2embed_dict = parse_embeddings_text_file(pos_txt_file, pos2embed_dict_path)

    pretrained_pos_embeddings = load_pretrained_embeddings("pos", pos2embed_dict, pos_vocabulary, int_to_pos, opts["pos_embedding_dim"])

    params.pos_embedding_dim = opts["pos_embedding_dim"]
    params.pos_embeddings    = pretrained_pos_embeddings
    params.pos_vocab_size    = len(pos_vocabulary)

# Dictionary to store F-score achieved by each configuration
F1_scores_dict =  dict()

# Grid search candidate values
epochs = [10, 20, 30]
learning_rates = [1e-2, 1e-3, 1e-4]
dropout_values = [0.3, 0.4, 0.5]
a = "bilstm-crf" # bilstm

# Grid search loops over the candidate values and trains a model 
# for each given configuration of candidate values
for e in epochs:
    opts['epochs'] = e
    print("E is", e)
    for learn_rate in learning_rates:
        opts['learning_rate'] = learn_rate
        for dropout in dropout_values:    
            opts['dropout'] = dropout

            # Folder to save this specific model
            opts['save_model_path'] = "./model/{}/{}_epochs_{}_LR_{}_DP".format(a, e, learn_rate, dropout)
            print(opts['save_model_path'])

            if not os.path.exists(opts["save_model_path"]):
                os.makedirs(opts["save_model_path"])

            ner_model = NERModel_CRF(params).cuda() # NERModel(params).cuda() if crf architecture is false

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

            # Save all model hyperparameters to file
            hyperParams_file = open("{}/hyper-parameters.txt".format(opts["save_model_path"]), "a", encoding="utf8")
            for (option, value) in opts.items():
                hyperParams_file.write("{}: {}\n".format(option, value))


            print("Number of Epochs", opts["epochs"])

            # Train model
            trainer.train(train_dataloader, dev_dataloader, opts)

            # Evaluate model by computing F-score on test set
            precisions = compute_precision(ner_model, test_dataloader, label_vocabulary, encoded_test_labels, opts)
            curr_f1 = precisions["f1"]
            print("\n\nF-1 score: {}\n\n".format(curr_f1))
            print("Confusion matrix\n", precisions["confusion_matrix"])
            
            # Print, plot, and save the confusion matrix
            file_name = "TestSet_Confusion_Matrix"
            plot_conf(file_name, precisions["confusion_matrix"], opts["save_model_path"])

            # Add F-score obtained by this model to dict storing all F-scores
            F1_scores_dict.update({opts["save_model_path"]: curr_f1})

# Find the model that achieved the highest F-score
best = max(F1_scores_dict.items(), key=lambda x: x[1])
print("Best model in {} scord F1 of {}".format(best[0], best[1]))






