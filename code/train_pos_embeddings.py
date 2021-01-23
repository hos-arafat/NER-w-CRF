import multiprocessing
from time import time
import logging
import os

import torch
import nltk
import numpy as np
from gensim.models import Word2Vec

from preprocess_dataset import PreProcessor, create_dataset



def train_pos2vec(training_data, w, embed_size, lr, epochs):
        """
        Trains CBOW Word2Vec model for one experiment with the best value of the hyper-parameters
        found after Gird Search (See Appendix A.2 in report.pdf)
        """
        # Specify formatting of logging the training progress
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        cores = multiprocessing.cpu_count() # Count the number of cores in a computer

        print()
        print("Training Word2Vec Model....")
        print("Number of cores is ", cores)

        # Initialize Word2Vec model
        w2v_model = Word2Vec(min_count=5, window=w, size=embed_size, sample=1e-3,
                     alpha=lr,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)
        # Claculate time taken to load vocab into model
        t = time()

        w2v_model.build_vocab(training_data, progress_per=10000)
        print(w2v_model)
        # words = list(w2v_model.wv.vocab)
        # print(words)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        # Claculate time taken to train the model once, for 30 epochs
        t = time()

        w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=epochs, report_delay=1)

        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

        return w2v_model

def save_pos_embeddings(w2v_model, path_to_save):
    """
    Save POS embeddings to (.vec) file
    """
    X = w2v_model[w2v_model.wv.vocab]

    print()
    print("Will save embeddings.vec to ", path_to_save)
    print("Embeddings Matrix type is ", type(X))
    print("Embeddings Matrix size ", X.shape)

    # Loop over the model's vocab
    pos_embeds = []
    all_tokens = w2v_model.wv.vocab
    for token in all_tokens:
        pos_embeds.append(token)

    print("Number of POS embeddings ", len(pos_embeds))

    # Write POS embeddings to file
    with open(path_to_save, "w") as f:
        f.write(str(len(pos_embeds)) + " " + str(X.shape[1]))
        f.write("\n")
    f.close()
    for key in pos_embeds:
        with open(path_to_save, "a") as f:
            f.write(key + " ")
            for element in w2v_model[key]:
                f.write(str(element) + " ")
            f.write("\n")
        f.close()

    print("Successfully Saved embeddings!")

if __name__ == "__main__":

    parent = "./data"

    p = PreProcessor("train")

    # Load training data as list of lists of input words
    p.read_data(parent)
    train_pos_data = p.create_pos_data(test_data=None)
    
    # Train Gensim Word2Vec model of POS Tags to obtain POS2Vec model
    model = train_pos2vec(train_pos_data, 10, 300, 1e-3, 30)
    save_pos_embeddings(model, "./model/pos_embeddings")