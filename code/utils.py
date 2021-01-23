import os
import pickle
from itertools import compress, takewhile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score


def collate_labelled_batch(batch):
    """
    Function that pads a batch which contains ground truth labels

    Args:
        batch: list of tuples or dicts containing the inputs, POS tags and labels
    Returns:
        padded_inputs: batch of torchTensors containing the input data
        padded_labels: batch of torchTensors containing the ground truth labels
        padded_pos: batch of torchTensors containing the POS tags
        mask: batch of torchTensors reflecting seq length and padding with 1s and 0s
    """

    unpadded_inputs = []
    unpadded_pos = []
    unpadded_labels = []

    # Detect if batch has POS tags or not
    pos_present = False
    
    for tup in batch:
        unpadded_inputs.append(tup[0])
        unpadded_labels.append(tup[1])
        # If tuple has 3rd POS tags element
        if len(tup) == 3:
            pos_present = True
            unpadded_pos.append(tup[2])

    # Pad inputs, POS tags, and labels per batch with a value of 0
    padded_inputs = torch.nn.utils.rnn.pad_sequence(unpadded_inputs, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(unpadded_labels, batch_first=True, padding_value=0)
    mask = (padded_inputs != 0)
    if pos_present:
        padded_pos = torch.nn.utils.rnn.pad_sequence(unpadded_pos, batch_first=True, padding_value=0)
        return padded_inputs, padded_labels, mask, padded_pos
    else:
        return padded_inputs, padded_labels, mask


def collate_test_batch(batch):
    """
    Function that pads a batch which DOES NOT contain ground truth labels
    
    Args:
        batch: list of tuples or dicts containing the inputs, POS tags
    Returns:
        padded_inputs: batch of torchTensors containing the input data
        padded_pos: batch of torchTensors containing the POS tags
        mask: batch of torchTensors reflecting seq length and padding with 1s and 0s
    """
    unpadded_inputs = []
    unpadded_pos = []

    # Detect if batch has POS tags or not
    pos_present = False

    for tup in batch: 
        unpadded_inputs.append(tup["inputs"])
        # If dict has 2nd POS tags element
        if len(tup) == 2:
            pos_present = True
            unpadded_pos.append(tup["pos"])

    # Pad inputs and POS tags per batch with a value of 0
    padded_inputs = torch.nn.utils.rnn.pad_sequence(unpadded_inputs, batch_first=True, padding_value=0)
    mask = (padded_inputs != 0)
    if pos_present:
        padded_pos = torch.nn.utils.rnn.pad_sequence(unpadded_pos, batch_first=True, padding_value=0)
        return padded_inputs, mask, padded_pos
    else:
        return padded_inputs, mask


def compute_precision(model:nn.Module, l_dataset, l_label_vocab, dev_d, opts, int_to_label, pre):
    """
    Function that Computes precision, recall, F-score, and confusion matrix
    
    Args:
        model: pytroch model to evaluate
        l_dataset: dataloader of a labelled dataset
        l_label_vocab: dictionary mapping labels to ints
        dev_d: list of TorchTensors contating ground truth labels
        opts: dictionary that specifies various training options and hyperparameters
    Returns:
        dictionary contatining precision, recall, F-score, and confusion matrix
    """

    all_predictions = list()
    all_labels = list()
    
    read_labels = list()
    read_predictions = list()

    indexed_pos = None

    model.eval()

    # Loop over dataset (dataloader)
    for indexed_elem in l_dataset:

        indexed_in = indexed_elem[0].to(opts["device"])
        indexed_labels = indexed_elem[1].to(opts["device"])
        mask = indexed_elem[2].to(opts["device"], dtype=torch.uint8)

        labels = indexed_labels.view(-1)
        valid_indices = labels != 0

        # If model uses POS embeddings include batch of encoded POS tags
        if opts["use_pos_embeddings"] == True:
            indexed_pos = indexed_elem[3].to(opts["device"])

        # If model uses CRF layer decode predictions
        if opts["use_crf"] == True:
            predictions = model.decode_crf(indexed_in, indexed_pos, mask)
            flattened_predictions = [w for l in predictions for w in l]
            all_predictions.extend(flattened_predictions)
            read_predictions.extend(predictions)
            
        elif opts["use_crf"] == False:
            predictions = model(indexed_in, indexed_pos)
            predictions = torch.argmax(predictions, -1).view(-1)
            valid_predictions = predictions[valid_indices]

            all_predictions.extend(valid_predictions.tolist())
        
        valid_labels = labels[valid_indices]
        all_labels.extend(valid_labels.tolist())
        
        read_labels.append(valid_labels.tolist())

    # Get index of a few labels and access the POS of those indeces
    select_labels_idx = [i for i, v in enumerate(all_labels) if v==1 or v==2 or v==3]
    all_pos = [y for x in pre.list_l_pos for y in x]
    
    pos_that_are_agent = [all_pos[ag_ix] for ag_ix in select_labels_idx]
    labels_that_are_agent = [int_to_label[all_labels[ag_ix]] for ag_ix in select_labels_idx]
    
    # Create a datafram of POS Tags and the labels associated with them
    df = pd.DataFrame(list(zip(pos_that_are_agent, labels_that_are_agent)), columns=["POS Tag", "Label"])
    
    # Plot the POS Tags of the Labels we selected in "select_labels_idx"
    df=df.groupby(['POS Tag','Label']).size()
    df=df.unstack()
    df.plot(kind='bar')
    plt.xticks(rotation=0)
    plt.show()

    # Compute precision, recall, F-score, and confusion matrix
    micro_precision = sk_precision(all_labels, all_predictions, average="micro", zero_division=0)
    macro_precision = sk_precision(all_labels, all_predictions, average="macro", zero_division=0)
    per_class_precision = sk_precision(all_labels, all_predictions, labels = list(range(len(l_label_vocab))), average=None, zero_division=0)
    confusion = confusion_matrix(all_labels, all_predictions, normalize="true")
    
    r = recall_score(all_labels, all_predictions, average='macro')
    f = f1_score(all_labels, all_predictions, average='macro')


    # Write the ground truth label and prediction of every sentence to file
    # for debugging purposes
    eval_file = "{}/Prediction_vs_GroundTruth.txt".format(opts["save_model_path"])
    s_op_file = open(eval_file, "a", encoding="utf8")


    for sentence, prediction in zip(dev_d, read_predictions):
        original_sentence = sentence.tolist()

        s_op_file.write("org:")
        for word in original_sentence:
            s_op_file.write(str(word) + " ")
        s_op_file.write("\n")
        s_op_file.write("pre:")
        for p in prediction:
            s_op_file.write(str(p) + " ")
        s_op_file.write("\n")
        s_op_file.write("\n")
        assert len(sentence) == len(prediction)
  
    # Write the F-score and Recall the model acheived to file      
    s_op_file.write("Rcall: {}\n\nF-1 score: {}\n\n".format(r, f))

    return {"micro_precision":micro_precision,
            "macro_precision":macro_precision, 
            "recall":r, 
            "f1":f, 
            "per_class_precision":per_class_precision,
            "confusion_matrix":confusion}


def plot_conf(f_name, matrix, save_path):
    """
    Function that confusion matrix as a heatmap
    
    Args:
        f_name: name to give to confusion matrix when saving it
        matrix: confusion matrix as returned by the "compute_precision" function
        save_path: path to save the confusion matrix heatmap figure
    """
    
    # Plot and save the confusion matrix
    fig = plt.figure(figsize=(10, 7))
    axes = fig.add_subplot(111)
    sn.set(font_scale=1.5)
    sn.heatmap(matrix, annot=True, annot_kws={"size": 15}, ax=axes)
    axes.set_xlabel('Predicted labels')
    axes.set_ylabel('True labels')
    axes.set_title('Confusion Matrix')
    axes.xaxis.set_ticklabels(['PER', 'ORG', 'LOC', 'O'])
    axes.yaxis.set_ticklabels(['PER', 'ORG', 'LOC', 'O'])
    # plt.show()
    plt.savefig("{:}/{}.png".format(save_path, f_name))
    return

def parse_embeddings_text_file(embedding_txt_file_path, token2embedding_dict_path):
    """
    Function that parses text file contatining lines as such:
    tokens [ floats representing the embeddings ]
    
    Args:
        embedding_txt_file_path: path to embeddings text file
        token2embedding_dict_path: path to save the "int:[embedding]" dictionary as pickle file
    Returns:
        index_to_vec: "int:[embedding]" dictionary representing storing each token as an int
        and its corresponding embedding vector
    """
    
    # If file does not exist
    if not os.path.exists(token2embedding_dict_path): 

        words = []
        idx = 0
        word2idx = {}
        vectors = []

        # Read file contents
        with open(embedding_txt_file_path, 'r') as f:
            f_content = f.readlines()
            for idx, l in enumerate(f_content):
                # GloVe embeddings file often starts with a blank space
                # we need to check if this is a space there by mistake
                # or if it is an embedding for the space or newline characters
                first_char = l[0]
                if first_char.isspace() == True:
                    print("\nFirst char is a space !!")
                    line = l.split()
                    if line[0][-1].isdigit() == True:
                        print("When we split, the first element is a float")
                        print("This means the first word was a space !")
                        whitespace = list(takewhile(str.isspace, l))
                        word = whitespace[0]
                        print("Whitespae word is `{}`".format(word))

                        words.append(word)
                        word2idx[word] = idx
                        idx += 1

                        vect = np.array(line).astype(np.float32)
                        print("Shape of vector for space elem is", vect.shape)
                        # print("vector is", vect)

                else:
                    # If character is not a space
                    line = l.split()
                    word = line[0]

                    words.append(word)
                    word2idx[word] = idx
                    idx += 1

                    digit_mask = [element[-1].isdigit() for element in line[1:]]

                    digits = list(compress(line[1:], digit_mask))
                    vect = np.array(digits).astype(np.float32)
                    print("Shape of vector for non space elem is", vect.shape)

                vectors.append(vect)

        vectors = np.asarray(vectors)
        print("Embedding Vectors shape", vectors.shape)

        # create integer:embedding vector dictionary 
        index_to_vec = {w: vectors[word2idx[w]] for w in words}
        
        # save dictionary to disk
        with open(token2embedding_dict_path, "wb") as handle:
            pickle.dump(index_to_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # Load dictionary from disk if we find it
        print("Loading Token2Embedding pickle file..")
        with open(token2embedding_dict_path, 'rb') as handle:
            index_to_vec = pickle.load(handle)

    return index_to_vec

def create_pretrained_embeddings(mode, token2vec_dict, vocabulary, decode, dim):
    """
    Function that creates pre-trained embeddings as a numpy array [vocab_size, embedding_dimension]
    
    Args:
        token2vec_dict: "int:[embedding]" dictionary representing storing each token as an int
        and its corresponding embedding vector, as returned by the "parse_embeddings_text_file" function
        vocabulary: dictionary mapping token to ints
        decode: dictionary ints to tokens
        dim: embedding dimension
    Returns:
        pretrained_embeddings: a numpy array representing embeddings [vocab_size, embedding_dimension]
    """
    
    if mode == "glove":
        print("Loading GloVe embeddings from disk...")
        embeddings_npy_path = "./model/glove_embeddings.npy"
    elif mode == "pos":
        print("Loading POS embeddings from disk...")
        embeddings_npy_path = "./model/pos_embeddings.npy"

    # Create torchTensor with size [vocab size, embedding dimension]
    pretrained_embeddings = torch.randn(len(vocabulary), dim)
    initialised = 0

    # loop over int-to-token dictionary
    for (i, w) in (decode.items()):
        i = int(i)
        w = str(w)
        if w in token2vec_dict:
            # Set the i-th element in embedding layer to be 
            # equal to the embedding of the i-th word in our vocab
            initialised += 1
            vec = token2vec_dict[w]
            pretrained_embeddings[i] = torch.FloatTensor(vec)

    # Set 0th vector in embedding layer to be all 0s
    pretrained_embeddings[vocabulary["<pad>"]] = torch.zeros(dim)

    # Save embeddings as NPY for faster loading in the future
    # using the "load_pretrained_embeddings" function below
    np.save(embeddings_npy_path, pretrained_embeddings)
    
    print("Done! \nInitialised {} embeddings {}".format(mode, initialised))
    print("Random initialised embeddings {} ".format(len(vocabulary) - initialised))

    return pretrained_embeddings


def load_pretrained_embeddings(embeddings_npy_path):
    """
    Function that loads pre-trained embeddings as a numpy array [vocab_size, embedding_dimension]
    
    Args:
        token2vec_dict: "int:[embedding]" dictionary representing storing each token as an int
        and its corresponding embedding vector, as returned by the "parse_embeddings_text_file" function
        vocabulary: dictionary mapping token to ints
        decode: dictionary ints to tokens
        dim: embedding dimension
    Returns:
        pretrained_embeddings: a numpy array representing embeddings [vocab_size, embedding_dimension]
    """
    print("\nLoading Embeddings from NPY file")
    pretrained_embeddings = torch.LongTensor(np.load(embeddings_npy_path))
    print("Embeddings from NPY file shape", pretrained_embeddings.shape)

    return pretrained_embeddings
