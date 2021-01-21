import torch
from torch import nn
import torch.optim as optim
import numpy as np
import time


class RNN(nn.Module):
    """
    Recurrent Neural Network module, implementing bLSTM and cbLSTM.
    """
    def __init__(self, input_dimension, embedding_dimension, num_of_layers, output_dimension,
                 hidden_dim=128, dropout=0.2, padding_idx=1, input_rep=0):
        """
        Args:
            input_dimension: size of vocabulary
            embedding_dimension: size of a word embedding
            num_of_layers: num. of LSTM layers to stack on top of one another
            output_dimension: num. of classes
            hidden_dim: size of the LSTM layer
            dropout: to be applied to embeddings and LSTM output
            padding_idx: index of the padding token
            input_rep: 0 = vanilla, 1 = case based
        """
        super().__init__()
        self.input_rep = input_rep
        self.embedding = nn.Embedding(input_dimension, embedding_dimension, padding_idx=padding_idx)
        lstm_input_dim = embedding_dimension
        if input_rep == 0:
            self.forward = self.forward_rep_0
        else:
            self.forward = self.forward_rep_1
            lstm_input_dim += 3
            case_vectors = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.case_embedding = nn.Embedding.from_pretrained(case_vectors, freeze=True)
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=num_of_layers,
                            dropout=dropout, bidirectional=True)
        self.dense = nn.Linear(hidden_dim * 2, output_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward_rep_0(self, x):
        """
        Forward function for the vanilla RNN.
        Args:
            x: input tensor

        Returns:
            predicted classes tensor
        """
        lstm_output, _ = self.lstm(self.dropout(self.embedding(x)))
        y_pred = self.dense(self.dropout(lstm_output))
        return y_pred

    def forward_rep_1(self, x):
        """
        Forward function for the case-based RNN.
        Args:
            x: list = [input embeddings tensor, input case type tensor]

        Returns:
            predicted classes tensor
        """
        x_embedding, x_case = x[0], x[1]
        x1 = self.dropout(self.embedding(x_embedding))
        x2 = self.case_embedding(x_case)
        lstm_input = torch.cat((x1, x2), 2)
        lstm_output, _ = self.lstm(lstm_input)
        y_pred = self.dense(self.dropout(lstm_output))
        return y_pred

    def init_weights(self, embeddings, padding_idx):
        """
        Initialize the network weights, excluding the case-based 3-dim vectors.
        Args:
            embeddings: pre-trained word embeddings
            padding_idx: index of padding token
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param.data, mean=0, std=0.1)
        self.embedding.weight.data.copy_(embeddings)
        self.embedding.weight.data[padding_idx] = torch.zeros(embeddings.shape[1])

    def fit(self, train_iterator, criterion, epochs, padding_tag_idx, val_iterator=None, save_best=True, verbose=True):
        """
        Fit the model on a dataset.
        Args:
            train_iterator (torch.data.BucketIterator): data to train on
            criterion: (torch.nn.CrossEntropyLoss): loss type
            epochs: num. of epochs to train
            padding_tag_idx: index of padding tag
            val_iterator: data to validate in (defaults to None)
            save_best: boolean to save the best performing model on the validation set
            verbose: to print progress (loss and acc per episode)
        """
        optimizer = optim.Adam([p for p in self.parameters() if p.requires_grad])
        best_val_acc = 0
        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(train_iterator, optimizer, criterion, padding_tag_idx)
            ep_time = int(time.time() - start_time)
            val_str = ''
            if val_iterator:
                val_loss, val_acc = self.evaluate(val_iterator, criterion, padding_tag_idx)
                best_str = ''
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_str = ' NEW BEST'
                    if save_best:
                        torch.save(self.state_dict(), f'model_input_rep_{self.input_rep}.sav')
                val_str = f' val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}{best_str}'
            if verbose:
                print(f'epoch:{epoch + 1} time:{ep_time}(secs) loss:{train_loss:.4f} acc:{train_acc:.4f}{val_str}')
        if not val_iterator and save_best:  # save after last epoch, if no validation set
            torch.save(self.state_dict(), f'model_input_rep_{self.input_rep}.sav')

    def train_epoch(self, train_iterator, optimizer, criterion, padding_tag_idx):
        """
        Perform one training epoch
        Args:
            train_iterator: (torch.data.BucketIterator): data to train on
            optimizer: (torch.optim.Adam): optimizer from pytorch
            criterion: (torch.nn.CrossEntropyLoss): loss type
            padding_tag_idx: index of padding tag

        Returns:

        """
        self.train()
        losses, accs = [], []
        for batch in train_iterator:
            x = batch.text if self.input_rep == 0 else [batch.text, batch.case]
            y = batch.tags.view(-1)

            optimizer.zero_grad()
            y_pred = self(x)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            acc = get_acc(y_pred, y, padding_tag_idx)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            accs.append(acc.item())
        return np.mean(losses), np.mean(accs)

    def evaluate(self, dataset_iterator, criterion, padding_tag_idx):
        """
        Evaluate the model on some dataset
        Args:
            dataset_iterator (torch.data.BucketIterator): data to test on
            criterion: (torch.nn.CrossEntropyLoss): loss type
            padding_tag_idx: index of padding tag

        Returns:
            loss and accuracy
        """
        losses, accs = [], []
        self.eval()
        with torch.no_grad():
            for batch in dataset_iterator:
                x = batch.text if self.input_rep == 0 else [batch.text, batch.case]
                y = batch.tags.view(-1)

                y_pred = self(x)
                y_pred = y_pred.view(-1, y_pred.shape[-1])

                loss = criterion(y_pred, y)
                acc = get_acc(y_pred, y, padding_tag_idx)
                losses.append(loss.item())
                accs.append(acc.item())

        return np.mean(losses), np.mean(accs)


def get_acc(y_pred, y, padding_tag_idx):
    """
    Helped function to compute the accuracy using tensors
    Args:
        y_pred: predicted label distributions
        y: true labels
        padding_tag_idx: index of padding tag

    Returns:
        accuracy of predictions
    """
    not_padding = y != padding_tag_idx
    y = y[not_padding]
    y_pred = y_pred.argmax(dim=1, keepdim=True)[not_padding]
    return y_pred.squeeze(1).eq(y).sum() / torch.FloatTensor([y.shape[0]])
