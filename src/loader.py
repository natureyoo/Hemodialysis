from torch.utils.data import Dataset


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
    0~5 : Regression & 5/7-way classification
    6~14: 60 minute target
        sbp, real "  : 4, 7
        map, real " : 5, 8
        under90, real " : 6, 9
        sbp from current, real " = 10, 12
        map from current, real " = 11, 13
    """
    def __init__(self, data, type, ntime=None):
        super().__init__()
        self.ntime = ntime
        if ntime is None:
            self.input = data[0][:,:,:-4]
            self.target = data[0][:,:,-4:]
        else :
            self.input = data[0][:,:,:-14]
            self.target = data[0][:,:,-14:]
        self.seq_len = data[1]
        self.type = type

    def __getitem__(self, idx):
        if self.type == 'Regression':
            x, y = self.input[:,idx,:], self.target[:,idx,:2]
        else:
            if self.ntime is None:
                x, y = self.input[:,idx,:], self.target[:,idx,2:]
            else:
                x, y = self.input[:,idx,:], (self.target[:,idx,[4,5,6,10,11]], self.target[:,idx,[7,8,9,12,13]])
        batch_seq_len = self.seq_len[idx]
        return (x, y, batch_seq_len)

    def __len__(self):
        return len(self.input[0])

class RNN_Val_Dataset(Dataset):
    def __init__(self, data, type, ntime=None):
        super().__init__()
        self.ntime = ntime
        if ntime is None:
            self.input = data[0][:,:,1:-4]
            self.target = data[0][:,:,-4:]
            self.mask = data[0][:,:,0]
        else :
            self.input = data[0][:,:,1:-14]
            self.target = data[0][:,:,-14:]
            self.mask = data[0][:,:,0]
        self.seq_len = data[1]
        self.type = type

    def __getitem__(self, idx):
        if self.type == 'Regression':
            x, y = self.input[:,idx,:], self.target[:,idx,:2]
        else:
            if self.ntime is None:
                x, y = self.input[:,idx,:], self.target[:,idx,2:]
                mask = self.mask[:,idx]
            else:
                x, y = self.input[:,idx,:], (self.target[:,idx,[4,5,6,10,11]], self.target[:,idx,[7,8,9,12,13]])
                mask = self.mask[:,idx]
        batch_seq_len = self.seq_len[idx]
        return (x, y, batch_seq_len, mask)

    def __len__(self):
        return len(self.input[0])