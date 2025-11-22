import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertModel
import os

# 避免huggingface 的 tokenizer 出现 fork 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CategoryDataset(Dataset):
    def __init__(self, text, choice, feedback, sub_exp_start, split_len=10):
        self.all_data = []
        for i in range(len(sub_exp_start)-1):
            start = sub_exp_start[i]
            end = sub_exp_start[i+1]
            for j in range(start, end-(split_len+1)):
                self.all_data.append((torch.stack([text[k] for k in range(j, j+split_len)]),
                                      torch.stack([choice[k] for k in range(j, j+split_len)]),
                                      torch.stack([feedback[k] for k in range(j, j+split_len)])))
                                      


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]   
    
    @staticmethod
    def load_data(sub_train_id, split_len, mode='train'):
        if mode == 'train':
            if os.path.exists('./cache/sub_train_id.pt'):
                if (torch.load('./cache/sub_train_id.pt',weights_only=True).tolist() == sub_train_id):
                    logger.info('Loading data from cache')
                    all_text_vector = torch.load('./cache/text3_0.pt',weights_only=True)
                    all_choice = torch.load('./cache/choice3_0.pt',weights_only=True)
                    all_feedback = torch.load('./cache/feedback3_0.pt',weights_only=True)
                    sub_exp_start = torch.load('./cache/sub_exp_start3_0.pt',weights_only=True)
                    return CategoryDataset(all_text_vector, all_choice, all_feedback, sub_exp_start, split_len)
        logger.info(f'Loading data from {sub_train_id} subjects')
        df = pd.read_csv('../dataset/Task2_aud.csv', keep_default_na=False, encoding='utf-8')
        texts = []
        for sub_id in tqdm(sub_train_id, desc='Loading texts'):
            text = df[df['iSub'] == sub_id]['text'].tolist()
            texts.append(text)

        df = pd.read_csv('../dataset/Task2_processed.csv', keep_default_na=False)
        choices = []
        feedbacks = []
        for sub_id in tqdm(sub_train_id, desc='Loading choices and feedbacks'):
            choice = df[df['iSub'] == sub_id]['choice'].tolist()
            feedback = df[df['iSub'] == sub_id]['feedback'].tolist()
            choices.append(choice)
            feedbacks.append(feedback)

        
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

        all_choice = []
        for choice in choices:
            all_choice.extend([int(c) for c in choice])
        all_choice = torch.tensor(all_choice) - 1
        logger.info('Choices loaded')

        all_feedback = []
        for feedback in feedbacks:
            all_feedback.extend([int(f) for f in feedback])
        all_feedback = torch.tensor(all_feedback)
        logger.info('Feedbacks loaded')

        sub_exp_start = [l:=0] + [l := l + len(text) for text in texts]

        if mode == 'train':
            if not os.path.exists('./cache'):
                os.makedirs('./cache')
            torch.save(torch.tensor(sub_train_id), './cache/sub_train_id.pt')
            torch.save(all_text_vector, './cache/text3_0.pt')
            torch.save(all_choice, './cache/choice3_0.pt')
            torch.save(all_feedback, './cache/feedback3_0.pt')
            torch.save(sub_exp_start, './cache/sub_exp_start3_0.pt')

        return CategoryDataset(all_text_vector, all_choice, all_feedback, sub_exp_start, split_len)






        