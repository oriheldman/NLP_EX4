import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class RNN(nn.Module):
    def __init__(self, input_dimension, embedding_dimension, num_of_layers, output_dimension,
                 hidden_dim=128, dropout=0.2, padding_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dimension, embedding_dimension, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dimension, hidden_dim, num_layers=num_of_layers,
                            dropout=dropout, bidirectional=True)
        self.dense = nn.Linear(hidden_dim * 2, output_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        lstm_output, _ = self.lstm(self.dropout(self.embedding(text)))
        y_pred = self.dense(self.dropout(lstm_output))
        return y_pred
