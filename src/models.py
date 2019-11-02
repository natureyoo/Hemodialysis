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

# hj
import numpy as np
class MLP_HJ(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP_HJ, self).__init__()
        self.input_BN = nn.BatchNorm1d(input_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
            # nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
            # nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
            # nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
            # nn.Dropout(),
            # nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, 32), nn.ReLU(), \
            nn.Linear(32, num_classes)
        )
        # self.fc3 = nn.Linear(hidden_size, num_classes)
        for m in self.net :
            # for m in self.fc:
                if isinstance(m, nn.BatchNorm2d):
                    if m.weight is not None:
                        m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.BatchNorm1d):
                    if m.weight is not None:
                        m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    # m.weight.data.normal_(0.0, 100.0)
                    # get the number of the inputs
                    # n = m.in_features
                    # y = 1.0/np.sqrt(n)
                    # m.weight.data.uniform_(-y, y)
                    # m.bias.data.fill_(0)
                    # nn.init.orthogonal_(m.weight.data)
                    # m.bias.data.fill_(0)
                    # m.weight.data.fill_(0)
                    nn.init.kaiming_normal_(m.weight.data)
                else:
                    pass
    def forward(self, x):
        x = self.input_BN(x)
        out = self.net(x)
        return out

#############################################################################################

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