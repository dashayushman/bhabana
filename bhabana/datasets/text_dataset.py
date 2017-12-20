import os

from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, root_dir, fields, line_processor):
        self.data_files = os.listdir(root_dir)
        self.fields = fields
        self.line_processor = line_processor
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.data_files[idx])
        with open(file_name, "r") as df:
            line = df.read().strip()
            sample = {'text': line, 'label': 1}
        return sample
