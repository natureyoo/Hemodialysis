import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import utils
import torch.nn.init as init


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

        self.fc_before_rnn = nn.Sequential(nn.Linear(input_size, input_size))
        self.input_BN = nn.BatchNorm1d(input_size)
        self.BN_after_gru = nn.BatchNorm1d(hidden_size)

        # CRU , FC
        self.gru = nn.GRU(input_size, hidden_size, num_layer, dropout=dropout_rate)
        # self.gru_first = nn.GRU(input_size, hidden_size, num_layer)
        if self.type == 'Regression':
            self.fc = nn.Linear(hidden_size, output_size)
        else:
            self.fc_class1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 7))
            self.fc_class2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), \
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 5))
            # self.fc_class2 = nn.Linear(hidden_size, 5)
            
        for m in [self.fc_class1, self.fc_class2, self.input_BN, self.gru] :
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
                    n = m.in_features
                    y = 1.0/np.sqrt(n)
                    m.weight.data.uniform_(-y, y)
                    m.bias.data.fill_(0)
                    # nn.init.orthogonal_(m.weight.data)
                    # m.bias.data.fill_(0)
                    # m.weight.data.fill_(0)
                    # nn.init.kaiming_normal_(m.weight.data)
                elif isinstance(m, nn.GRU):
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            print('GRU weight init. --> orthogonal_')
                            init.orthogonal_(param.data)
                        # else:
                        #     init.normal_(param.data)
                else:
                    pass

    def forward(self, X, seq_len, device):
        h_0 = self.init_hidden().to(device)
        X = X.float()
        # X = X.permute(1, 2, 0)
        # X = self.input_BN(X.float())
        # X = X.permute(2,0,1)

        X = self.batchnorm_by_using_first_seq(X)
        X = self.fc_before_rnn(X)

        packed = rnn_utils.pack_padded_sequence(X, seq_len, batch_first=False, enforce_sorted=False)
        packed = packed.float().to(device)
        # _, h = self.gru_first(packed, h_0)
        output, _ = self.gru(packed, h_0)

        # output, _ = self.gru(packed, h_0)
        unpacked, unpacked_len = rnn_utils.pad_packed_sequence(output)

        # unpacked = unpacked.permute(1,2,0)
        # unpacked = self.BN_after_gru(unpacked)
        # unpacked = unpacked.permute(2,0,1)
        # print(X.size(), unpacked.size())
        if self.type == 'Regression':
            output = self.fc(unpacked) # (seq_len, bath_num, output_size)
            return output
        else:
            output1, output2 = self.fc_class1(unpacked), self.fc_class2(unpacked)
            return output1, output2

    def init_hidden(self):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        return hidden
    def batchnorm_by_using_first_seq(self, feature):
        mean = torch.mean(feature[0].detach(), dim=0)
        std = torch.std(feature[0].detach(), dim=0) + 1e-5
        feature = torch.div((feature-mean), std)
        return feature
    


class RNN_V2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size, batch_size, dropout_rate, sbp_num_class=7, map_num_class=5, another_class=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate


        # self.fc_before_rnn = nn.Sequential(nn.Linear(input_size, input_size), nn.Tanh(), nn.Linear(input_size, input_size))
        self.fc_before_rnn = nn.Sequential(nn.Linear(input_size, input_size))
        self.BN_before_fc = nn.BatchNorm1d(input_size)

        self.gru_module_1 = nn.GRUCell(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.gru_module_2 = nn.GRUCell(hidden_size, hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.gru_module_3 = nn.GRUCell(hidden_size, hidden_size)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)

        

        self.inter_fc_1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.inter_fc_2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.inter_fc_3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        self.gru_module_4 = nn.GRUCell(input_size, hidden_size)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.inter_fc_4 = nn.Sequential(nn.Linear(hidden_size,input_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(input_size, input_size))

        self.fc_class1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), 
            nn.Linear(hidden_size, sbp_num_class))
        self.fc_class2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), 
            nn.Linear(hidden_size, map_num_class))
            
        for m in [self.fc_class1, self.fc_class2, self.inter_fc_1, self.inter_fc_2, self.inter_fc_3, self.fc_before_rnn] :
            if isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                # n = m.in_features
                # y = 1.0/np.sqrt(n)
                # m.weight.data.uniform_(-y, y)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        # print('GRU weight init. --> orthogonal_')
                        # init.orthogonal_(param.data)
                        print('GRU weight init. --> xavier_normal_')
                        init.xavier_normal_(param.data)
            else:
                pass

    def forward(self, X, seq_len, device):
        h1, h2, h3, h4 = None, None, None, None
        X = X.float()
        X = X.permute(1, 2, 0)
        X = self.BN_before_fc(X)
        X = X.permute(2,0,1)
        seq_length = X.size(0)

        # X = self.batchnorm_by_using_first_seq(X).float()
        outputs = list()
        for i in range(seq_length):
            h1 = self.gru_module_1(X[i], h1)
            h_prop = self.inter_fc_1(h1)
            h_prop = self.layer_norm(h_prop)
            h_prop = F.relu(h_prop)
            h2 = self.gru_module_2(h_prop, h2)
            h_prop = self.inter_fc_2(h2)
            h_prop = self.layer_norm_2(h_prop)
            h_prop = F.relu(h_prop)
            h3 = self.gru_module_3(h_prop, h3)
            h_prop = self.inter_fc_3(h3)
            h_prop = self.layer_norm_3(h_prop)

            h_prop = self.inter_fc_4(h_prop)

            h4 = self.gru_module_4((h_prop+X[i]), h4)
            h_prop = h4


            h_prop = h_prop.float()
            outputs.append(h_prop)
            
        outputs = torch.stack(outputs)
        output1, output2 = self.fc_class1(outputs), self.fc_class2(outputs)
        return output1, output2

    def init_hidden(self):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        return hidden

    def batchnorm_by_using_first_seq(self, feature):
        mean = torch.mean(feature[0].detach(), dim=0)
        std = torch.std(feature[0].detach(), dim=0) + 1e-5
        feature = torch.div((feature-mean), std)
        return feature



class RNN_V3(nn.Module):
    def __init__(self, input_fix_size, input_seq_size, hidden_size, num_layer, output_size, batch_size, dropout_rate, sbp_num_class=7, map_num_class=5, another_class=0):
        super().__init__()
        self.input_fix_size = input_fix_size
        self.input_seq_size = input_seq_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.temp_logit = None
        self.temp_logit2 = None

        self.fc_fix = nn.Sequential(nn.Linear(input_fix_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.BN_before_fc_fix = nn.BatchNorm1d(input_fix_size)
        self.BN_before_fc_seq = nn.BatchNorm1d(input_seq_size)

        self.gru_module_1 = nn.GRUCell(input_seq_size, hidden_size)
        self.layer_norm = nn.LayerNorm(input_seq_size)
        self.gru_module_2 = nn.GRUCell(hidden_size, hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.gru_module_3 = nn.GRUCell(hidden_size, hidden_size)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)

        self.inter_fc_1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(hidden_size, input_seq_size))
        # self.inter_fc_1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(hidden_size, input_seq_size))
        self.inter_fc_2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.inter_fc_3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        self.gru_module_4 = nn.GRUCell(input_seq_size, hidden_size)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.inter_fc_4 = nn.Sequential(nn.Linear(hidden_size,input_seq_size), nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(input_seq_size, input_seq_size))

        self.merge_fc = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))


        self.fc_class1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(), 
            # nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Linear(hidden_size, 1))

        self.fc_class2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Linear(hidden_size, 1))

        self.fc_class3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Linear(hidden_size, 1))

        self.fc_class4 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Linear(hidden_size, 1))

        self.fc_class5 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), \
            nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Linear(hidden_size, 1))
            
        for m in [self.inter_fc_1, self.inter_fc_2, self.inter_fc_3, self.fc_fix] :
        # for m in [self.inter_fc_1, self.fc_fix] :
            if isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                # n = m.in_features
                # y = 1.0/np.sqrt(n)
                # m.weight.data.uniform_(-y, y)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        # print('GRU weight init. --> orthogonal_')
                        # init.orthogonal_(param.data)
                        print('GRU weight init. --> xavier_normal_')
                        init.xavier_normal_(param.data)
            else:
                pass

    def forward(self, X_fix, X_seq, seq_len, device):
        '''
        X : torch.Tensor
        X shape : (seq, batch size, feature size)
        seq_len : list --> e.g. [7,4,5,6,7,11]
        len(seq_len) == batch size

        output shape : (seq, batch size, 5)
        '''
        h1, h2, h3, h4 = None, None, None, None
        X_fix = X_fix.float()
        X_fix = self.BN_before_fc_fix(X_fix)
        X_seq = X_seq.float()
        X_seq = X_seq.permute(1, 2, 0)
        X_seq = self.BN_before_fc_seq(X_seq)
        X_seq = X_seq.permute(2,0,1)
        seq_length = X_seq.size(0)

        # fix input
        h_fix = self.fc_fix(X_fix)

        outputs = list()
        for i in range(seq_length):
            h1 = self.gru_module_1(X_seq[i], h1)
            h_prop = self.inter_fc_1(h1)
            h_prop = self.layer_norm(h_prop)
            h_prop = F.relu(h_prop)
            # h2 = self.gru_module_2(h_prop, h2)
            # reset_gate, input_gate = self.get_gate_value(self.gru_module_2, X_seq[i], torch.zeros(64, 256).float().to(device))
            # print('{}th reset_gate...{}, mean : {}'.format(i, reset_gate.shape, torch.mean(reset_gate)))
            # print(reset_gate)
            # print('{}th input_gate...{}, mean : {}'.format(i, input_gate.shape, torch.mean(input_gate)))
            # print(input_gate)
            # h_prop = self.inter_fc_2(h2)
            # h_prop = self.layer_norm_2(h_prop)
            # h_prop = F.relu(h_prop)
            # h3 = self.gru_module_3(h_prop, h3)
            # h_prop = self.inter_fc_3(h3)
            # h_prop = self.layer_norm_3(h_prop)
            # h_prop = self.inter_fc_4(h_prop)
            h4 = self.gru_module_4((h_prop+X_seq[i]), h4)
            h_prop = h4
            h_prop = self.merge_fc(torch.cat((h_fix, h_prop), 1))
            h_prop = h_prop.float()
            outputs.append(h_prop)

        outputs = torch.stack(outputs)  # list 를 torch.Tensor로 만들기 위해

        output1 = self.fc_class1(outputs)
        output2 = self.fc_class2(outputs)
        output3 = self.fc_class3(outputs)
        output4 = self.fc_class4(outputs)
        output5 = self.fc_class5(outputs)
        return torch.cat([output1, output2, output3, output4, output5], dim=-1)
        # return torch.cat([output1, output2, output3], dim=-1)

    def init_hidden(self):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        return hidden

    def batchnorm_by_using_first_seq(self, feature):
        mean = torch.mean(feature[0].detach(), dim=0)
        std = torch.std(feature[0].detach(), dim=0) + 1e-5
        feature = torch.div((feature-mean), std)
        return feature