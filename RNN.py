import torch
from torch import nn
import torch.optim as optim
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
            lstm_input_dim += 3
            self.forward = self.forward_rep_1
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

    def fit(self, dataset_iterator, criterion, epochs, padding_tag_idx, verbose=True):
        """
        Fit the model on a dataset.
        Args:
            dataset_iterator (torch.data.BucketIterator): data to train on
            criterion: (torch.nn.CrossEntropyLoss): loss type
            epochs: num. of epochs to train
            padding_tag_idx: index of padding tag
            verbose: to print progress (loss and acc per episode)
        """
        optimizer = optim.Adam([p for p in self.parameters() if p.requires_grad])
        for epoch in range(epochs):
            start_time = time.time()
            self.train()
            epoch_losses, epoch_accs = [], []
            for batch in dataset_iterator:
                x = batch.text if self.input_rep == 0 else [batch.text, batch.case]
                y = batch.tags.view(-1)

                optimizer.zero_grad()
                y_pred = self(x)
                y_pred = y_pred.view(-1, y_pred.shape[-1])

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                acc = get_acc(y_pred, y, padding_tag_idx)
                epoch_losses.append(loss.item())
                epoch_accs.append(acc.item())

            if verbose:
                ep_time = int(time.time() - start_time)
                total_loss, total_acc = sum(epoch_losses) / len(epoch_losses), sum(epoch_accs) / len(epoch_accs)
                print(f'epoch:{epoch + 1} time:{ep_time}(secs) loss:{total_loss:.4f} acc:{total_acc:.4f}')

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

        return sum(losses) / len(losses), sum(accs) / len(accs)


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

