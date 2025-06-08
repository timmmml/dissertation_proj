from torch.utils.data import Dataset
class PredictorDataset(Dataset):
    def __init__(self, z0, I, q):
        self.z0 = z0
        self.I = I
        self.q = q

    def to(self, device):
        self.z0 = self.z0.to(device)
        self.I = self.I.to(device) if self.I is not None else None
        self.q = self.q.to(device)
        return self

    def __len__(self):
        return len(self.z0)

    def __getitem__(self, idx):
        return self.z0[idx], self.I[idx], self.q[idx]
