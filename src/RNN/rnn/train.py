import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import yaml
import pandas as pd
from tqdm.auto import tqdm
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
from pytorch_lightning.tuner import Tuner


from data import CategoryDataset
from data_module import CategoryDataModule
from core import CategoryRNN


import structlog
logger = structlog.get_logger()

sub_id = [1,  2,  3,  7,  8,  9, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]
# sub_species_id = [i for i in sub_id if i % 3 == 2]
# sub_family_id = [i for i in sub_id if i % 3 == 1]
sub_both_id = [i for i in sub_id if i % 3 == 0]
# sub_train_id = sub_family_id[:-1]
# sub_test_id = sub_family_id[-1]
sub_train_id = sub_both_id[:-1]
sub_test_id = sub_both_id[-1]
# sub_train_id = sub_species_id[:-1]
# sub_test_id = sub_species_id[-1]
logger.info(f'sub_train_id: {sub_train_id}, sub_test_id: {sub_test_id}')
split_len = 128

dataset = CategoryDataset.load_data(sub_train_id, split_len)
data_module = CategoryDataModule(dataset)
logger.info(f'Data loaded, len: {len(dataset)}, split_len: {split_len}')

choice_size = 4 #TODO: family: 2, both: 4, species: 4
model = CategoryRNN(768, 768, choice_size, 2, lr = 0.003)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkpoints',
    filename='rnn3.1.1-both-'+str(split_len)+'-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[checkpoint_callback],
)
tuner = Tuner(trainer)
tuner.lr_find(model, data_module, min_lr=1e-8, max_lr=0.002, num_training=1000)
trainer.fit(model, data_module)
with open(os.path.join(trainer.logger.log_dir, 'hparams.yaml'), 'w') as f:
    yaml.dump(model.hparams, f)
logger.info('best_model_path : %s', checkpoint_callback.best_model_path)
trainer.test(model, data_module.test_dataloader())


    
