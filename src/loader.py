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
    def __init__(self, data, type):
        super().__init__()
        # self.input = data[0][:,:,:-4]
        # self.target = data[0][:,:,-4:]
        self.data = data[0]
        self.seq_len = data[1]
        self.type = type

    def __getitem__(self, idx):
        if self.type == 'Regression':
            x, y = self.input[:,idx,:], self.target[:,idx,:2]
        else:
            # x, y = self.input[:,idx,:], self.target[:,idx,2:]
            x, y = self.data[idx][:,:-4], self.data[idx][:,-2:]
        batch_seq_len = self.seq_len[idx]
        return (x, y, batch_seq_len)

    def __len__(self):
        return len(self.data[0])