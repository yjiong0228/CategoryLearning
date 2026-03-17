import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import yaml
import pandas as pd
from tqdm.auto import tqdm
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
from pytorch_lightning.tuner import Tuner


from .dataset import CategoryDataset
from .data_module import CategoryDataModule
from .core import CategoryTransformerDecoder
from .config import train_config


import structlog
logger = structlog.get_logger()

from enum import Enum
class Group(Enum):
    FAMILY = 1
    SPECIES = 2
    BOTH = 3

LOG_DIR = os.path.dirname(os.path.abspath(__file__))

def train(group: Group, split_len: int):
    sub_id = [1,  2,  3,  7,  8,  9, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]
    if group == Group.FAMILY:        
        sub_family_id = [i for i in sub_id if i % 3 == 1]
        sub_train_id = sub_family_id[:-1]
        sub_test_id = sub_family_id[-1]
        choice_size = 2
    elif group == Group.SPECIES:
        sub_species_id = [i for i in sub_id if i % 3 == 2]
        sub_train_id = sub_species_id[:-1]
        sub_test_id = sub_species_id[-1]
        choice_size = 4
    elif group == Group.BOTH:
        sub_both_id = [i for i in sub_id if i % 3 == 0]
        sub_train_id = sub_both_id[:-1]
        sub_test_id = sub_both_id[-1]
        choice_size = 4
    else:
        raise ValueError('Invalid group')

    logger.info(f'sub_train_id: {sub_train_id}, sub_test_id: {sub_test_id}')

    dataset = CategoryDataset.load_data(sub_train_id, split_len)
    data_module = CategoryDataModule(dataset)
    logger.info(f'Data loaded, len: {len(dataset)}, split_len: {split_len}')

    # choice_size = 2 #TODO: family: 2, both: 4, species: 4
    model = CategoryTransformerDecoder(
        stimuli_size=4,
        text_size=768,
        choice_size=choice_size,
        feedback_size=2,
        config=train_config,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(LOG_DIR, 'checkpoints', group.name),
        filename='gpt1.0.4-'+str(split_len)+'-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        default_root_dir=LOG_DIR,
    )
    tuner = Tuner(trainer)
    tuner.lr_find(model, data_module, min_lr=1e-8, max_lr=0.01, num_training=1000)
    trainer.fit(model, data_module)
    with open(os.path.join(trainer.logger.log_dir, 'hparams.yaml'), 'w') as f:
        yaml.dump(model.hparams, f)
    logger.info('best_model_path : %s', checkpoint_callback.best_model_path)
    trainer.test(model, data_module.test_dataloader())


    
