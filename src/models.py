import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import utils


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size, batch_size, dropout_rate, type='Regression'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.type = type

        self.fc_before_gru = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, hidden_size))

        # CRU , FC
        self.gru = nn.GRU(hidden_size, hidden_size, num_layer, dropout=dropout_rate)

        self.fc_after_gru = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, hidden_size), nn.ReLU())

        if self.type == 'Regression':
            self.fc = nn.Linear(hidden_size, output_size)
        else:
            self.fc_class1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 7))
            self.fc_class2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 5))

    def forward(self, X, seq_len, device):
        h_0 = self.init_hidden().to(device)

        X = self.fc_before_gru(X)

        packed = rnn_utils.pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=True)
        packed = packed.float().to(device)
        output, _ = self.gru(packed, h_0)
        output, _ = rnn_utils.pad_packed_sequence(output)

        output = self.fc_after_gru(output)
        if self.type == 'Regression':
            output = self.fc(output) # (seq_len, bath_num, output_size)
            return output
        else:
            output1, output2 = self.fc_class1(output), self.fc_class2(output)
            return output1, output2

    def init_hidden(self):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        return hidden