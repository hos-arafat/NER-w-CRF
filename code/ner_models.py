import torch
import torch.nn as nn
from torchcrf import CRF


class NERModel(nn.Module):
    """Class the implementing BiLSTM model"""
    def __init__(self, hparams):
        super(NERModel, self).__init__()
        """
        Args:
            hparams: Class containing hyperparameters
        Returns:
            o: output of the model
        """

        self.hparams = hparams

        # input to lstm is either word embeddings or (word + pos) embeddings
        lstm_input_dim = hparams.embedding_dim       


        # Word Embedding layer: a matrix [wors vocab size, embedding_dim]
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings is not None:
            print("\nModel Initializing word embeddings from pretrained GloVe")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        # POS Embedding layer: a matrix [POS tags vocab size, embedding_dim]
        if self.hparams.use_pos_embeddings == True:
            self.pos_embedding = nn.Embedding(self.hparams.pos_vocab_size, self.hparams.pos_embedding_dim)
            if self.hparams.pos_embeddings is not None:
                print("Model Initializing POS embeddings from pretrained")
                self.pos_embedding.weight.data.copy_(self.hparams.pos_embeddings)
            
            lstm_input_dim = hparams.embedding_dim + hparams.pos_embedding_dim        

        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with word embeddings) from left to right and outputs 
        # a new **contextual** representation of each word that depend
        # on the preciding words.
        self.lstm = nn.LSTM(lstm_input_dim, hparams.hidden_dim, 
                            batch_first=True,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0)
        # Hidden layer: transforms the input value/scalar into a hidden vector representation.
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        # Dropout and final fully connected layer
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)
    
    def forward(self, x, p):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        # If we are using POS embeddings
        if p is not None:
            pos_embeddings = self.pos_embedding(p)
            pos_embeddings = self.dropout(pos_embeddings)
            concat_embeddings = torch.cat((embeddings, pos_embeddings), dim=-1)
            o, (h, c) = self.lstm(concat_embeddings)
        else:
            o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output

class NERModel_CRF(nn.Module):
    """Class the implementing BiLSTM model with a CRF layer"""
    def __init__(self, hparams):
        super(NERModel_CRF, self).__init__()
        """
        Args:
            hparams: Class containing hyperparameters
        Returns:
            o: output of the model
        """

        self.hparams = hparams

        # input to lstm is either word embeddings or (word + pos) embeddings
        lstm_input_dim = hparams.embedding_dim

        # Word Embedding layer: a matrix [wors vocab size, embedding_dim]
        self.word_embedding = nn.Embedding(self.hparams.vocab_size, self.hparams.embedding_dim)
        if self.hparams.embeddings is not None:
            print("\nModel Initializing embeddings from pretrained GloVe")
            self.word_embedding.weight.data.copy_(self.hparams.embeddings)
            print("GloVe Done!")

        # POS Embedding layer: a matrix [POS tags vocab size, embedding_dim]
        if self.hparams.use_pos_embeddings == True:
            self.pos_embedding = nn.Embedding(self.hparams.pos_vocab_size, self.hparams.pos_embedding_dim)
            if self.hparams.pos_embeddings is not None:
                print("Model Initializing POS embeddings from pretrained")
                self.pos_embedding.weight.data.copy_(self.hparams.pos_embeddings)
                print("POS Done!")

            
            lstm_input_dim = hparams.embedding_dim + hparams.pos_embedding_dim        

        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with word embeddings) from left to right and outputs 
        # a new **contextual** representation of each word that depend
        # on the preciding words.           
        self.lstm = nn.LSTM(lstm_input_dim, self.hparams.hidden_dim, 
                                batch_first=True,
                                bidirectional=self.hparams.bidirectional,
                                num_layers=self.hparams.num_layers, 
                                dropout = self.hparams.dropout if self.hparams.num_layers > 1 else 0)

        # Hidden layer: transforms the input value/scalar into a hidden vector representation.
        lstm_output_dim = self.hparams.hidden_dim if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2

        # Dropout and final fully connected layer
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, self.hparams.num_classes)

        # Conditional Random Field layer
        self.crf = CRF(self.hparams.num_classes, batch_first=True)

    
    def forward(self, x, p):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        # If we are using POS embeddings
        if p is not None:
            pos_embeddings = self.pos_embedding(p)
            pos_embeddings = self.dropout(pos_embeddings)
            concat_embeddings = torch.cat((embeddings, pos_embeddings), dim=-1)
            o, (h, c) = self.lstm(concat_embeddings)
        else:
            o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output
    
    def log_probs(self, x, pos, tags, batch_mask):
        """
        Function that calculates the CRF loss
        Args:
            x: batch of torchTensor containing the inputs
            pos: batch of torchTensor containing the POS tags if they are used, else None
            tags: batch of torchTensor containing the ground truth labels
            batch_mask: batch of torchTensor containing a mask to reflect sequence length and padding
        Returns:
            o: CRF loss
        """
        emissions = self(x, pos)
        return self.crf(emissions, tags, mask=batch_mask)

    def decode_crf(self, x, pos, batch_mask):
        """
        Function that calculates the CRF loss
        Args:
            x: batch of torchTensor containing the inputs
            pos: batch of torchTensor containing the POS tags if they are used, else None
            tags: batch of torchTensor containing the ground truth labels
            batch_mask: batch of torchTensor containing a mask to reflect sequence length and padding
        Returns:
            o: decpded output of the CRF layer
        """
        emissions = self(x, pos)
        return self.crf.decode(emissions, mask=batch_mask)
