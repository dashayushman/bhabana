import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TextDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_files = os.listdir(root_dir)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.data_files[idx])
        with open(file_name, "r") as df:
            line = df.read().strip()
            sample = {'text': line, 'label': 1}
        return sample


if __name__ == '__main__':
    ds = TextDataset(root_dir='/home/dash/.bhabana/datasets/imdb/train/pos')
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=4, )
    for i_batch, sample_batched in enumerate(dl):
        print(i_batch, len(sample_batched['text']),
              len(sample_batched['label']))
        print(sample_batched['text'][0])
        print(sample_batched['text'][1])
        break