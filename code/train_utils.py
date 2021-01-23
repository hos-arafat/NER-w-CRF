# This file contains a dictionary that specifies the model's hyperparameters

opts = {}


opts['random_seed'] = 0

# Type of architecture to use

# opts['architecture'] = "lstm" 
opts['architecture'] = "bilstm-crf" # "bilstm", "lstm"...
# opts['bidirectional'] = False
# opts['use_crf'] = False

# Bidirectional LSTM or not
if opts['architecture'] == "bilstm" or opts['architecture'] == "bilstm-crf":
    opts['bidirectional'] = True
elif opts['architecture'] == "lstm":
    opts['bidirectional'] = False

# Use a CRF layer or not
if opts['architecture'] == "bilstm" or opts['architecture'] == "lstm":
    opts['use_crf'] = False
elif opts['architecture'] == "bilstm-crf":
    opts['use_crf'] = True

# Number of (Bi)LSTM layers
opts['lstm_layers'] = 1

# (Bi)LSTM hidden dimension
opts['hidden_dim'] = 256

opts['epochs'] = 2
opts['learning_rate'] = 1e-2

opts['dropout'] = 0.5

# Wether or not to use pretrained GloVe embeddings
opts['use_glove_embeddings'] = True 

# Wether or not to use POS embeddings
opts['use_pos_embeddings'] = True
# Wether or not to use pretrained POS embeddings
opts['use_pretrained_pos_embeddings'] = True

opts['glove_embedding_dim'] = 300
opts['pos_embedding_dim'] = 300

# Wether or not to use Gradient clipping
opts['grad_clipping'] = True

# Wether or not to use Early stopping
opts['early_stopping'] = True

opts['device'] = "cuda"

opts["save_model_path"] = "./model/10_epochs_0.01_LR_0.5_DP"

class HParams():
    """Class that specifies the model's hyperparameters."""
    def __init__(self, vocabulary, pos_vocabulary, label_vocabulary, opts):
        """
        Args:
            vocabulary: dictionary mapping input words to ints
            pos_vocabulary: dictionary mapping POS tags to ints
            label_vocabulary: dictionary mapping ground truth labels to ints
            opts: dictionary (above) that specifies various training options and hyperparameters
        """ 

        self.vocab_size = len(vocabulary)
        self.label_vocabulary = len(label_vocabulary)
        self.pos_vocab_size = len(pos_vocabulary)
        self.num_classes = len(label_vocabulary)

        self.hidden_dim = opts["hidden_dim"]
        self.embedding_dim = opts['glove_embedding_dim']
        self.pos_embedding_dim = opts['pos_embedding_dim']

        self.use_pos_embeddings = opts['use_pos_embeddings']

        self.bidirectional = opts['bidirectional']
        self.num_layers = opts['lstm_layers']
        self.dropout = opts['dropout']

        self.embeddings = None
        self.pos_embeddings = None