from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, input_ids, tag_ids, attention_masks):
        self.input_ids = input_ids
        self.tag_ids = tag_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'tag_ids': self.tag_ids[idx],
            'attention_mask': self.attention_masks[idx]
        }