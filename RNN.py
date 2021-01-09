import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_dimension, embedding_dimension, num_of_layers, output_dimension,
                 hidden_dim=128, dropout=0.2, padding_idx=1, input_rep=0):
        super().__init__()
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
        lstm_output, _ = self.lstm(self.dropout(self.embedding(x)))
        y_pred = self.dense(self.dropout(lstm_output))
        return y_pred

    def forward_rep_1(self, x):
        x_embedding, x_case = x[0], x[1]
        x1 = self.dropout(self.embedding(x_embedding))
        x2 = self.case_embedding(x_case)
        lstm_input = torch.cat((x1, x2), 2)
        lstm_output, _ = self.lstm(lstm_input)
        y_pred = self.dense(self.dropout(lstm_output))
        return y_pred