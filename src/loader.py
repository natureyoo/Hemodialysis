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
    def __init__(self, data, seq_len):
        super().__init__()
        self.input = data[0]
        self.target = data[1]
        self.seq_len = seq_len

    def __getitem__(self, idx):
        x, y = self.input[:,idx,:], self.target[:,idx]
        batch_seq_len = self.seq_len[idx]
        return (x, y, batch_seq_len)

    def __len__(self):
        return len(self.input[0])

class ToyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.input = data[0]
        self.target = data[1]
        self.seq_len = data[2]

    def __getitem__(self, idx):
        x, y = self.input[idx], self.target[idx]
        seq = self.seq_len[idx]
        return (x,y,seq)

    def __len__(self):
        return len(self.input)