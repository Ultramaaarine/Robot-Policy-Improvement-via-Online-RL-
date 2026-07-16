import torch
from torch.utils.data import Dataset


class OnlineDataset(Dataset):
    def __init__(self,seq,batch_size):
        super().__init__()
        self.seq = seq
        self.batch_size = batch_size


    def __len__(self):
        return self.seq.shape[0] 
    
    def __getitem__(self,idx):
        batch = self.seq[idx]
        return batch