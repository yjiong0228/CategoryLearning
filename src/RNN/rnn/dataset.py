from contextlib import contextmanager
from torch.utils.data import Dataset
import csv

from tqdm.auto import tqdm 



class CLDataset(Dataset):
    def __init__(self, sub_dict):
        self.sub_dict = sub_dict


    @staticmethod
    @contextmanager
    def get_reader(dataset_path: str):
        f = open(dataset_path, encoding="GBK", errors="ignore", mode="r", newline="")
        reader = csv.DictReader(f)
        try:
            yield reader
        finally:
            f.close()

    @classmethod
    def get_sub_dict(cls, dataset_path: str):
        sub_dict = {}      
        def extract_sub_dict(row, sub_dict):
            if row['iSub'] not in sub_dict:
                sub_dict[row['iSub']] = {}
                sub_dict[row['iSub']]['index'] = row['iSub']
                sub_dict[row['iSub']]['session'] = row['iSession']
                sub_dict[row['iSub']]['version'] = row['version']
                sub_dict[row['iSub']]['condition'] = row['condition']
                sub_dict[row['iSub']]['exp'] = []
            exp_list = sub_dict[row['iSub']]['exp']
            trial = row['iTrial']
            feature = [row['feature1'], row['feature2'], row['feature3'], row['feature4']]
            text = row['text']        
            category = row['category']
            choice = row['choice']
            feadback = row['feedback']
            exp_list.append({'trial': trial, 'feature': feature, 'text': text, 'category': category, 'choice': choice, 'feedback': feadback})
        with CLDataset.get_reader(dataset_path) as reader:
            for row in tqdm(reader):
                extract_sub_dict(row, sub_dict)
        return cls(sub_dict)

        
    
    def get_text_list(self):
        text_list = []
        for sub in self.sub_dict:
            for exp in self.sub_dict[sub]['exp']:
                text_list.append(exp['text'])
        return text_list