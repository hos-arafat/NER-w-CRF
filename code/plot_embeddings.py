import os
import re
from argparse import ArgumentParser

from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


# Specify command line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-e", "--embed_path", required=False, default="../resources/embeddings.vec", help="Path to the sense embeddings")

    return parser.parse_args()


def load_gensim_embeddings(path_to_load):
    """
    Loads the '.vec' file from path specified as argument on the command line

    returns: loaded Word2Vec model
    """
    print()
    print("Loading embeddings from ", path_to_load)
    model = KeyedVectors.load_word2vec_format(path_to_load, binary=False)
    print("Successfully loaded embeddings!")
    return model


def visualize_gensim_embeddings(path_to_load):
    """
    Plots the embeddings (from the ".vec" file) for a few select 'representative' senses
    """

    # Ensure we can load the embeddings from the specified path
    if os.path.exists(path_to_load):
        print("\nFound the embeddings!")
        w2v_model = load_gensim_embeddings(path_to_load)
    else:
        print("\nUnable to find embeddings! Please verify path is correct or train the network to obtain them...")
        return

    # Retrive the model's Vocabulary
    words = list(w2v_model.wv.vocab)
    # Create list of words to be plotted
    list_of_words = ['NNP', 'VBD', 'VBN', 'DT', 'NN', 'CC', 'PRP$', 'MD', 'VB', 
    'RB', 'NNS', 'WP', 'JJ', 'PRP', 'VBZ', 'JJS', 'JJR', 'POS', 
    'RBR', 'VBG', 'VBP', 'WDT', 'NNPS', 'WRB', 'FW', 'RBS', 'PDT', 'EX', 'WP$']
    # list_of_words = words

    # Print the senses found
    print("Plotting the embeddings for the following senses {:}".format(list_of_words))

    # Get vectors all the senses we are interested in plotting
    X = w2v_model[list_of_words]
    
    # Preform PCA dimensionality reduction on the sense embedding vectors
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    # Create scatter plot of sense embeddings
    plt.scatter(result[:, 0], result[:, 1])

    # Loop over each vector / scatter point and annotate it with the sense
    for i, word in enumerate(list_of_words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    # Plot the figure
    plt.title("Embeddings")
    # Save the figure
    plt.savefig('subset_of_trained_embeddings.png', dpi=None, facecolor='w', edgecolor='b',
    orientation='portrait', papertype=None, format=None,
    transparent=False, bbox_inches="tight", pad_inches=0.1,
    frameon=None, metadata=None)
    plt.show()


def visualize_model_embeddings(mode, embeds, token_to_int, opts):
    """
    Plots the embeddings (from the ".vec" file) for a few select 'representative' senses
    """

    if mode == "word":
        if opts["use_glove_embeddings"] == True:   
            fig_name = "GloVe Embeddings"
        elif opts["use_glove_embeddings"] == False:   
            fig_name = "Model Embeddings"
    
        # pick some words to visualise
        words = ['dog', 'horse', 'animal','Tesla', 'Einstein', 'Hitler', 'Merkel', 'Sergio','Roberto', 'France',
        'Italy', 'Spain', 'Donald', 'Trump', 'Education', 'Harvard', 'MIT', 'York', 'Boston',
        "Deutsche", "Times", "Bank", 'Obama', 'Bush', "the", 'parents', 'sometimes']

        # # pick some words to visualise
        # words = ['dog', 'horse', 'animal', 'Einstein', 'Merkel', 'Sergio','Roberto', 'France',
        # 'Italy', 'Spain', 'Donald', 'Harvard', 'MIT', 'York', 'Boston', 'Orleans',
        # "Deutsche", "Times", 'Obama', 'Bush', "the", 'sometimes']

        # pick some words to visualise
        # words = ['dog', 'horse', 'animal', 'Einstein', 'Merkel', 'Sergio','Roberto', 'France',
        # 'Italy', 'Spain', 'United', 'Donald', "Deutsche", 'Obama', 'Bush',
        # "the",'sometimes']


    elif mode == "pos":
        if opts["use_pos_embeddings"] == True:
            fig_name = "Model POS Embeddings"
            if opts["use_pretrained_pos_embeddings"] == True:
                fig_name = "Pre-Trained POS Embeddings"
        # pick some POS tags to visualise
        words = ["WRB", "MD", "WP", "WRB", "PRP", "PRP$", "NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "RB", "RBR", "RBS", "DT"] #"Paris", "France", "Europe", "united_states_of_america", "country", "city"]

    
    # perform PCA to reduce our 300d embeddings to 2d points that can be plotted
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeds.detach().cpu())

    indexes = [token_to_int[x] for x in words]
    points = [pca_result[i] for i in indexes]
    for i,(x,y) in enumerate(points):
        plt.plot(x, y, 'ro')
        plt.text(x, y, words[i], fontsize=12) # add a point label, shifted wrt to the point
    plt.title(fig_name)
    plt.savefig("{}/{}.png".format(opts["save_model_path"], fig_name))
    plt.show()


    return


if __name__ == '__main__':

    # Parse the command line arguments
    args = parse_args()
    
    # Visualize POS embeddings
    visualize_gensim_embeddings(args.embed_path)
