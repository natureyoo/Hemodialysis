from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import torch

def pad_collate(batch, val=False):
    input_fix = [i[0][0] for i in batch]
    input_seq = [i[0][1] for i in batch]
    inter_target = [i[1][0] for i in batch]
    real_target = [i[1][1] for i in batch]
    batch_seq_len = [i[2] for i in batch]
    # if val: mask=[i[3] for i in batch]

    input_fix = torch.tensor(input_fix)
    input_seq = rnn_utils.pad_sequence([torch.tensor(x) for x in input_seq])
    inter_target = rnn_utils.pad_sequence([torch.tensor(x) for x in inter_target])
    real_target = rnn_utils.pad_sequence([torch.tensor(x) for x in real_target])
    # if val: mask = rnn_utils.pad_sequence([torch.tensor(x) for x in mask])
    #
    # if val:
    #     return ((input_fix, input_seq), (inter_target, real_target), batch_seq_len, mask)
    # else:
    #     return ((input_fix, input_seq), (inter_target, real_target), batch_seq_len)
    return ((input_fix, input_seq), (inter_target, real_target), batch_seq_len)

class MLP_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.dataset = data

    def __getitem__(self, idx):
        target = self.dataset[idx, -9:]
        x, y = self.dataset[idx, :-9], target[4:]
        return (x, y)

    def __len__(self):
        return len(self.dataset)

class RNN_Dataset(Dataset):
    """
    idx : target
    ------------------------------------------------------------------------------------------
    0~3 : Regression, classification (SBP, DBP)
    4~8 : 5 individual target (Initial SBP, Initial MAP, Under90, Current SBP, Current MAP)
    9~10 : 2 composite target (Initial, Current)
    ------------------------------------------------------------------------------------------
    """
    def __init__(self, data, type, ntime=None):
        self.ntime = ntime
        self.dataset = data[0]   # (fixed data, sequence data)
        self.seq_len = data[1]
        self.type = type

    def __getitem__(self, idx):
        if self.type == 'Regression':
            # Not implemented
            x, y = self.dataset[idx][:,idx,:], self.dataset[:,idx,:2]
        else:
            # target = self.dataset[idx][1][:,-14:]
            # x, y = (self.dataset[idx][0], self.dataset[idx][1][:,:-14]), (target[:,[4,5,6,10,11]], target[:,[7,8,9,12,13]])
            target = self.dataset[idx][1][:,-9:]
            x, y = (self.dataset[idx][0], self.dataset[idx][1][:,:-9]), (target[:,4:], target[:,4:])
        batch_seq_len = self.seq_len[idx]
        return (x, y, batch_seq_len)

    def __len__(self):
        return len(self.dataset)
