import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

PROCESSED_DATA_PATH = './data/processed/Task2_processed.csv'

@dataclass
class TrialData:
    features : list[float] # features
    choice : int # category subjects chose
    feedback : float # feedback subjects got

@dataclass
class TrialNumpyData(TrialData):
    features : np.ndarray[np.float32]
    choice : np.int32
    feedback : np.float32

class SubjectDataset(object):
    def __init__(self, data: List[TrialData], condition: int):
        self.data = data
        self.condition = condition

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        def to_numpy(data: TrialData) -> TrialNumpyData:
            return TrialNumpyData(
                features = np.array(data.features, dtype=np.float32),
                choice = np.int32(data.choice - 1),
                feedback = np.float32(data.feedback)
            )                                  
        if isinstance(idx, slice):            
            return [to_numpy(data) for data in self.data[idx]]
        return to_numpy(self.data[idx])

class CategoryDataset(object):

    def __init__(self):
        self.data : Dict[int, List[TrialData]] = {} # key: subject_id, value: list of TrialData
        self.condition : Dict[int, int] = {} # key: subject_id, value: condition

    def get_subject(self, iSub: int) -> SubjectDataset:
        return SubjectDataset(self.data[iSub], self.condition[iSub])

    @staticmethod
    def load_data_from_file(file_path: str = PROCESSED_DATA_PATH) -> 'CategoryDataset':
        dataset = CategoryDataset()
        df = pd.read_csv(file_path, keep_default_na=False)
        for iSub, sub_data in df.groupby('iSub'):
            dataset.data[iSub] = [TrialData(features = row[['feature1', 'feature2', 'feature3', 'feature4']].values.tolist(),
                                            choice = row['choice'],
                                            feedback = row['feedback']) for idx, row in sub_data.iterrows()]
            dataset.condition[iSub] = sub_data['condition'].iloc[0]
        return dataset
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test the data loading function')
    args = parser.parse_args()

    if args.test:
        dataset = CategoryDataset.load_data_from_file()
        # test number of subjects
        assert len(dataset.data) == 24
        # test number of trials for the first subject
        assert len(dataset.get_subject(1)) == 128
        # test the first trial data for the first subject
        assert dataset.get_subject(1)[0].features.tolist() == [0.09672186523675919, 0.4840755760669708, 0.08017640560865402, 0.47426190972328186]
        assert dataset.get_subject(1)[0].choice == 1
        assert dataset.get_subject(1)[0].feedback == 1.0
        print('All tests passed!')
    


