"""
对Task2的口头汇报数据进行文本分析
"""

import pandas as pd
import re

class Processor:
    # def process(self, processed_data):


    # Define body parts and their corresponding column names
    BODY_PARTS = {
        '脖子': 'neck_value',
        '头': 'head_value',
        '腿': 'leg_value',
        '尾巴': 'tail_value'
    }

    # Define description keywords and their corresponding values
    DESCRIPTIONS = {
        '长': 3,
        '短': 1,
        '中等': 2,
        '适中': 2
    }

    # Define possible modifiers between body parts and descriptions in "比" pattern
    MODIFIERS = ['比较', '很', '等']

    def extract_values(self, text):
        # Initialize the result dictionary with None
        result = {
            'invalid': 0,
            'noinfo': 0,
            'neck_value': None,
            'head_value': None,
            'leg_value': None,
            'tail_value': None
        }
        
        # Check if text is NaN or empty after stripping
        if pd.isna(text) or str(text).strip() == '':
            result['invalid'] = 1
            return result
        
        # Split the text by Chinese comma and remove any trailing punctuation
        items = re.split(r'[，,]', text)
        
        for item in items:
            item = item.strip('。.？?！!、')
            if not item:
                continue
            
            # Check if there's a comparison with "比" (but not "比较")
            has_comparison = "比" in item and "比较" not in item
            
            # Find all body parts mentioned in the item
            mentioned_parts = [part for part in BODY_PARTS.keys() if part in item]
            
            if not mentioned_parts:
                continue
                
            # Find description keywords, but ensure "长" is not part of another word
            descriptions_found = []
            for desc in DESCRIPTIONS.keys():
                if desc == '长':
                    # Check if '长' exists but not as part of '长度'
                    if '长' in item and '长度' not in item:
                        descriptions_found.append(desc)
                else:
                    if desc in item:
                        descriptions_found.append(desc)
            
            if len(descriptions_found) >= 1:
                desc_value = DESCRIPTIONS[descriptions_found[0]]
                
                if has_comparison:
                    # Find the parts before and after "比"
                    parts_before = [part for part in mentioned_parts 
                                if item.find(part) < item.find('比')]
                    parts_after = [part for part in mentioned_parts 
                                if item.find(part) > item.find('比')]
                    
                    # Assign opposite values for parts before "比"
                    for part in parts_before:
                        result[BODY_PARTS[part]] = desc_value
                    
                    # Assign normal values for parts after "比"
                    for part in parts_after:
                        result[BODY_PARTS[part]] = 4 - desc_value
                else:
                    # No comparison, assign same value to all parts
                    for part in mentioned_parts:
                        result[BODY_PARTS[part]] = desc_value

        # New Logic: Handle "其他" or "其余" with a description adjective
        for item in items:
            if any(keyword in item for keyword in ['其他', '其余']):
                # Find description adjectives in the item
                descriptions_found = []
                for desc in DESCRIPTIONS.keys():
                    if desc == '长':
                        # Ensure '长' is not part of '长度'
                        if '长' in item and '长度' not in item:
                            descriptions_found.append(desc)
                    else:
                        if desc in item:
                            descriptions_found.append(desc)
                
                if descriptions_found:
                    # Use the first found description
                    desc_value = DESCRIPTIONS[descriptions_found[0]]
                    
                    # Assign to all body parts that are still None
                    for part, col in BODY_PARTS.items():
                        if result[col] is None:
                            result[col] = desc_value
                    break  # Assuming only one "其他" or "其余" per text

        # Check if all body part values are still None
        body_values = [result[col] for col in BODY_PARTS.values()]
        if all(v is None for v in body_values):
            result['noinfo'] = 1
        else:
            result['noinfo'] = 0
        
        return result

    def process(self, input_file, output_file):
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Initialize new columns
        df['invalid'] = 0
        df['noinfo'] = 0
        df['neck_value'] = None
        df['head_value'] = None
        df['leg_value'] = None
        df['tail_value'] = None
        
        # Apply the extraction function to each row
        extracted_data = df['text'].apply(extract_values)
        
        # Populate the new columns based on the extracted data
        df['invalid'] = extracted_data.apply(lambda x: x['invalid'])
        df['noinfo'] = extracted_data.apply(lambda x: x['noinfo'])
        df['neck_value'] = extracted_data.apply(lambda x: x['neck_value'])
        df['head_value'] = extracted_data.apply(lambda x: x['head_value'])
        df['leg_value'] = extracted_data.apply(lambda x: x['leg_value'])
        df['tail_value'] = extracted_data.apply(lambda x: x['tail_value'])
        
        # Save the processed DataFrame to a new CSV file
        df.to_csv(output_file, index=False, encoding='utf-8-sig')