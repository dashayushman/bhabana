import os
import codecs
from functools import reduce

from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, root_dir, fields, line_processor):
        self.data_files = os.listdir(root_dir)
        self.fields = fields
        self.line_processor = line_processor
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_files)

    def update_sample(self, sample, payload):
        sample.update(payload)
        return sample

    def __getitem__(self, idx):
        processed_fields = []
        file_name = os.path.join(self.root_dir, self.data_files[idx])
        with codecs.open(file_name, "r", "utf-8") as df:
            line = df.read().strip()
            data_in_fields = self.line_processor(line)
            for field_data, field in zip(self.fields, data_in_fields):
                if hasattr(field["processors"], "__iter__"):
                    for processor in field["processors"]:
                        if processor.add_to_output:
                            processed_fields.append(processor(field_data))
        sample = reduce(self.update_sample, tuple(processed_fields), {})
        return sample
