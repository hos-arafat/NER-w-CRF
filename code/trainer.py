import os

import torch
import torch.nn as nn


class Trainer():
    """Utility class to train and evaluate a model."""

    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer,
        label_vocab,
        is_CRF:bool=False,
        use_pos:bool=False,
        clip_grads:bool=False,
        early_stop:bool=False,
        log_steps:int=128,
        log_level:int=2,
        device:str="cuda"):
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
            is_CRF: whether or not the model has a CRF layer.
            use_pos: whether or not the model has a POS embedding layer.
            clip_grads: whether or not we want to use gradient clipping.
            early_stop: whether or not we want to use early stopping.
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.label_vocab = label_vocab

        self.device = device


        self.crf = is_CRF
        self.pos_embed = use_pos

        self.clip = clip_grads
        self.es = early_stop


    def train(self, train_dataset, 
              valid_dataset,
              opts):
        """
        Args:
            train_dataset: a Dataset or DatasetLoader instance containing
                the training instances.
            valid_dataset: a Dataset or DatasetLoader instance used to evaluate
                learning progress.
            opts: dictionary that specifies various training options and hyperparameters

        Returns:
            avg_train_loss: the average training loss on train_dataset over
                epochs.
        """
        if self.log_level > 0:
            print('Training ...')
        
        train_loss = 0.0

        epochs = opts["epochs"]
        save_folder = opts["save_model_path"]
        # Wether or not and early stop even happened
        early_stop_event = False

        prev_loss = list()
        pos = None


        for epoch in range(epochs):
            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            epoch_acc = 0.0
            self.model.train()

            for step, sample in enumerate(train_dataset):

                inputs = sample[0].to(self.device)
                labels = sample[1].to(self.device, dtype=torch.int64)
                mask = sample[2].to(self.device, dtype=torch.uint8)
                if self.pos_embed == True:
                    pos = sample[3].to(self.device)
                

                self.optimizer.zero_grad()

                if self.crf == False:
                    predictions = self.model(inputs, pos)
                    print("Predictions are", predictions.shape)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    print("Predictions become", predictions.shape)
                    labels = labels.view(-1)
                    print("Labels become", labels.shape)
                    sample_loss = self.loss_function(predictions, labels)

                # If model has CRF layer 
                elif self.crf == True:
                    sample_loss = -self.model.log_probs(inputs, pos, labels, mask)                  
                    predictions_list = self.model.decode_crf(inputs, pos, mask)

                    unpadded_predictions = [torch.LongTensor(x) for x in predictions_list]
                    predictions = torch.nn.utils.rnn.pad_sequence(unpadded_predictions, batch_first=True, padding_value=0)
                    predictions.to(self.device, dtype=torch.int64)               

                sample_loss.backward()

                if self.clip == True:
                    # Apply Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0) 
                
                self.optimizer.step()

                epoch_loss += sample_loss.tolist()

                # sample_acc = self.calculate_accuracy(predictions, labels)
                # epoch_acc += sample_acc

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, epoch_loss / (step + 1)))
            
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_loss = self.evaluate(valid_dataset)
            prev_loss.append(valid_loss)

            if self.log_level > 0:
                print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))

            # If we want to use early stopping
            if self.es == True:
                # print list of all val losses for debugging purposes
                # print("list of val losses", prev_loss)
                if epoch > 0:
                    if  valid_loss > prev_loss[epoch-1]:
                        print("Validation loss increased ! Stopping training...")
                        early_stop_event = True
                        break

            # Save the model If early stopping is off 
            # or the val loss did NOT increase / Early Stop even did NOT happen
            if early_stop_event == False:
                print("Saving model")
                torch.save(self.model.state_dict(), os.path.join(save_folder, 'state_{}.pth'.format(epoch))) # save the model state
                torch.save(self.model, os.path.join(save_folder, 'checkpoint_{}.pt'.format(epoch))) # save the model state
            else:
                print("Early stop event triggered, not saving the model for this epoch")


        if self.log_level > 0:
            print('... Done!')
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    

    def evaluate(self, valid_dataset):
        """
        Args:
            valid_dataset: the dataset to use to evaluate the model.

        Returns:
            avg_valid_loss: the average validation loss over valid_dataset.
        """
        valid_loss = 0.0
        
        pos = None
        
        # set dropout to . Needed when we are in inference mode.
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:

                inputs = sample[0].to(self.device)
                labels = sample[1].to(self.device, dtype=torch.int64)
                mask = sample[2].to(self.device, dtype=torch.uint8)
                if self.pos_embed == True:
                    pos = sample[3].to(self.device)


                if self.crf == False:

                    predictions = self.model(inputs, pos)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    labels = labels.view(-1)
                    sample_loss = self.loss_function(predictions, labels)
                    
                # If model has CRF layer
                elif self.crf == True:
                    sample_loss = -self.model.log_probs(inputs, pos, labels, mask)
                    predictions_list = self.model.decode_crf(inputs, pos, mask)

                    unpadded_predictions = [torch.LongTensor(x) for x in predictions_list]
                    predictions = torch.nn.utils.rnn.pad_sequence(unpadded_predictions, batch_first=True, padding_value=0)
                    predictions.to(self.device, dtype=torch.int64)

                valid_loss += sample_loss.tolist()

                labels_list = labels.tolist()
                predictions_list = (torch.argmax(predictions, -1)).tolist()
        
        return valid_loss / len(valid_dataset)


