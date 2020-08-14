from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import torch

def pad_collate(batch, val=False):
    input_fix = [i[0][0] for i in batch]
    input_seq = [i[0][1] for i in batch]
    inter_target = [i[1] for i in batch]
    batch_seq_len = [i[2] for i in batch]

    input_fix = torch.tensor(input_fix)
    input_seq = rnn_utils.pad_sequence([torch.tensor(x) for x in input_seq])
    inter_target = rnn_utils.pad_sequence([torch.tensor(x) for x in inter_target])
    return ((input_fix, input_seq), (inter_target), batch_seq_len)

class HD_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.input = data[0]
        self.target = data[1]

    def __getitem__(self, idx):
        x, y = self.input[idx], self.target[idx]
        return (x, y)

    def __len__(self):
        return len(self.input)

class RNN_Dataset(Dataset):
    """
    idx : target
    ------------------------------------------------------------------------------------------
    -5 : IDH-1
    -4 : IDH-2
    -3 : IDH-3
    -2 : IDH-4
    -1 : IDH-5
    ------------------------------------------------------------------------------------------
    """
    def __init__(self, data, ntime=None):
        self.ntime = ntime
        self.dataset = data[0]   # (fixed data, sequence data)
        self.seq_len = data[1]

    def __getitem__(self, idx):
        target = self.dataset[idx][1][:,-5:]
        x, y = (self.dataset[idx][0], self.dataset[idx][1][:,:-5]), (target)
        batch_seq_len = self.seq_len[idx]
        return (x, y, batch_seq_len)

    def __len__(self):
        return len(self.dataset)