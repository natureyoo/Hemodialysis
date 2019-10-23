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

class MLP_large(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.bn1(self.relu(self.fc1(x)))
        out = self.bn2(self.relu(self.fc2(out)))
        out = self.fc3(out)
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

        # CRU , FC
        self.gru = nn.GRU(input_size, hidden_size, num_layer, dropout=dropout_rate)
        if self.type == 'Regression':
            self.fc = nn.Linear(hidden_size, output_size)
        else:
            self.fc_class1 = nn.Linear(hidden_size, 7)
            self.fc_class2 = nn.Linear(hidden_size, 5)

    def forward(self, X, seq_len, device):
        h_0 = self.init_hidden().to(device)
        packed = rnn_utils.pack_padded_sequence(X, seq_len, batch_first=False, enforce_sorted=False)
        packed = packed.float().to(device)
        output, _ = self.gru(packed, h_0)
        unpacked, unpacked_len = rnn_utils.pad_packed_sequence(output)
        if self.type == 'Regression':
            output = self.fc(unpacked) # (seq_len, bath_num, output_size)
            return output
        else:
            output1, output2 = self.fc_class1(unpacked), self.fc_class2(unpacked)
            return output1, output2

    def init_hidden(self):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        return hidden