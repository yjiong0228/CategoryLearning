import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import os

# 避免huggingface 的 tokenizer 出现 fork 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

class CategoryDataset(Dataset):
    def __init__(self, text, stimuli, choice, feedback, sub_exp_start, split_len=10):
        self.all_data = []
        for i in range(len(sub_exp_start)-1):
            start = sub_exp_start[i]
            end = sub_exp_start[i+1]
            for j in range(start, end-split_len+1): #TDOD: check the range
                self.all_data.append((torch.stack([stimuli[k] for k in range(j, j+split_len)]),
                                      torch.stack([text[k] for k in range(j, j+split_len)]),
                                      torch.stack([choice[k] for k in range(j, j+split_len)]),
                                      torch.stack([feedback[k] for k in range(j, j+split_len)])))                                     
                                      


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]   
    
    @staticmethod
    def load_data(sub_train_id, split_len, mode='train'):
        if mode == 'train':
            if os.path.exists(CACHE_PATH + '/sub_train_id.pt'):
                if (torch.load(CACHE_PATH + '/sub_train_id.pt',weights_only=True).tolist() == sub_train_id):
                    logger.info('Loading data from cache')
                    all_text_vector = torch.load(CACHE_PATH + '/text.pt',weights_only=True)
                    all_stimuli = torch.load(CACHE_PATH + '/stimuli.pt',weights_only=True)
                    all_choice = torch.load(CACHE_PATH + '/choice.pt',weights_only=True)
                    all_feedback = torch.load(CACHE_PATH + '/feedback.pt',weights_only=True)
                    sub_exp_start = torch.load(CACHE_PATH + '/sub_exp_start.pt',weights_only=True)
                    return CategoryDataset(all_text_vector, all_stimuli, all_choice, all_feedback, sub_exp_start, split_len)
        logger.info(f'Loading data from {sub_train_id} subjects')
        df = pd.read_csv(PROJECT_PATH + '/dataset/Task2_aud.csv', keep_default_na=False, encoding='utf-8')
        texts = []
        for sub_id in tqdm(sub_train_id, desc='Loading texts'):
            text = df[df['iSub'] == sub_id]['text'].tolist()
            texts.append(text)

        df = pd.read_csv(PROJECT_PATH + '/dataset/Task2_processed.csv', keep_default_na=False)
        stimulis = []
        choices = []
        feedbacks = []
        for sub_id in tqdm(sub_train_id, desc='Loading stimulis, choices and feedbacks'):
            stimuli = df.loc[df['iSub'] == sub_id, ['feature1', 'feature2', 'feature3', 'feature4']].values.tolist()
            choice = df[df['iSub'] == sub_id]['choice'].to_list()
            feedback = df[df['iSub'] == sub_id]['feedback'].to_list()
            stimulis.append(stimuli)
            choices.append(choice)
            feedbacks.append(feedback)

        from transformers import AutoTokenizer, BertModel
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        model = BertModel.from_pretrained("bert-base-chinese")
        all_text = []
        for text in texts:
            all_text.extend(text)

        with torch.no_grad():
            inputs = tokenizer(all_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            all_text_vector = outputs.pooler_output
        logger.info('Texts loaded')

        all_stimuli = []
        for stimuli in stimulis:
            all_stimuli.extend(stimuli)
        all_stimuli = torch.tensor(all_stimuli, dtype=torch.float)
        logger.info('stimulis loaded')

        all_choice = []
        for choice in choices:
            all_choice.extend(choice)
        all_choice = torch.tensor(all_choice, dtype=torch.long) - 1
        logger.info('Choices loaded')

        all_feedback = []
        for feedback in feedbacks:
            all_feedback.extend(feedback)
        all_feedback = torch.tensor(all_feedback, dtype=torch.long)
        logger.info('Feedbacks loaded')

        sub_exp_start = [l:=0] + [l := l + len(text) for text in texts]

        if mode == 'train':
            if not os.path.exists(CACHE_PATH):
                os.makedirs(CACHE_PATH)
            torch.save(torch.tensor(sub_train_id), CACHE_PATH + '/sub_train_id.pt')
            torch.save(all_text_vector, CACHE_PATH + '/text.pt')
            torch.save(all_choice, CACHE_PATH + '/choice.pt')
            torch.save(all_feedback, CACHE_PATH + '/feedback.pt')
            torch.save(all_stimuli, CACHE_PATH + '/stimuli.pt')
            torch.save(sub_exp_start, CACHE_PATH + '/sub_exp_start.pt')

        return CategoryDataset(all_text_vector, all_stimuli, all_choice, all_feedback, sub_exp_start, split_len)






        