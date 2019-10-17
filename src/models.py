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
    def __init__(self, input_size, hidden_size, output_size, num_layer, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch_size = batch_size

        # LSTM , FC
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)

    def forward(self,X, seq_len, device):
        # print(X.shape)
        h_0 = torch.zeros(self.num_layer, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layer, self.batch_size, self.hidden_size).to(device)

        packed = rnn_utils.pack_padded_sequence(X,seq_len, batch_first=False, enforce_sorted=False)
        packed = packed.float().to(device)

        output, _ = self.lstm(packed, (h_0,c_0))
        # print(output.data.size())
        unpacked, unpacked_len = rnn_utils.pad_packed_sequence(output)

        # print('Shape of unpacked', unpacked.size())
        output = unpacked.permute(1, 0, 2) # (batch, seq_len, hidden_size)
        output = F.relu6(self.fc1(output))
        output = F.relu6(self.fc2(output))
        output = self.fc3(output)
        # print("Shape of Output: ", output.shape)
        return output


