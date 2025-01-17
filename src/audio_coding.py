"""
对Task2的口头汇报数据进行文本分析
"""

import os
import pandas as pd
import re

class Processor:
    # def process(self, processed_data):

    # Define feature synonyms
    feature_synonyms = {
        'neck': ['脖子'],
        'head': ['头'],
        'leg': ['腿'],
        'tail': ['尾巴']
    }

    # Define adjective categories
    adjective_synonyms = {
        0.25: ['短'],
        0.75: ['长'],
        0.5: ['正常', '中等', '适中']
    }

    def interpret(description):
        # Initialize the feature dictionary with 'NA'
        features = {f'{feature}_value': 'NA' for feature in feature_synonyms.keys()}
        direct = {f'{feature}_direct': 'NA' for feature in feature_synonyms.keys()}
        indirect = {f'{feature}_indirect': 'NA' for feature in feature_synonyms.keys()}
        
        # Interpret each feature based on its synonyms and adjective categories
        for feature, terms in feature_synonyms.items():
            for term in terms:
                for value, adjectives in adjective_synonyms.items():
                    if any(re.search(f'{term}.{{0,2}}{adj}', description) for adj in adjectives):
                        features[f'{feature}_value'] = value
                        direct[f'{feature}_direct'] = 1
        
        # Apply the "比" logic if the description contains "比"
        if '比' in description:
            for feature, terms in feature_synonyms.items():
                for term in terms:
                    if any(re.search(f'比{term}.{{0,2}}{adj}', description) for adj in adjective_synonyms[1]):
                        features[f'{feature}_value'] = 3
                        direct[f'{feature}_direct'] = 'NA'
                        indirect[f'{feature}_indirect'] = 1                       
                    elif any(re.search(f'比{term}.{{0,2}}{adj}', description) for adj in adjective_synonyms[3]):
                        features[f'{feature}_value'] = 1
                        direct[f'{feature}_direct'] = 'NA'
                        indirect[f'{feature}_indirect'] = 1

        return {**features, **direct, **indirect}