import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Transformer

from .utils.transformer import (
    ExtendedTransformerDecoder,
    ExtendedTransformerDecoderLayer,
)

class CategoryTransformerDecoder(pl.LightningModule):
    def __init__(self, stimuli_size, text_size, choice_size, feedback_size, config, lr = 0.01):
        super(CategoryTransformerDecoder, self).__init__()
        self.save_hyperparameters()
        d_model = text_size + stimuli_size + choice_size + feedback_size
        self.hparams.config.d_model = d_model
        decoder_layer = ExtendedTransformerDecoderLayer(d_model, self.hparams.config.nhead, batch_first=True)
        self.decoder = ExtendedTransformerDecoder(decoder_layer, num_layers=self.hparams.config.num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + text_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, choice_size)
        )

    def forward(self, data):
        """
        data: tuple of (stimuli, text, choice, feedback)
        stimuli: (batch, seq_len, stimuli_size)
        text: (batch, seq_len, text_size)
        choice: (batch, seq_len)
        feedback: (batch, seq_len)

        return: (batch, 4)
        """
        stimuli, text, choice, feedback = data
        choice = F.one_hot(choice.to(torch.int64), num_classes=self.hparams.choice_size).float()
        feedback = F.one_hot(feedback.to(torch.int64), num_classes=self.hparams.feedback_size).float()
        x = torch.cat((stimuli[:, :-1], text[:, :-1], choice[:, :-1], feedback[:, :-1]), dim=2) # (batch, seq_len-1, d_model)
        tgt_mask = Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.decoder.forward(x, x, tgt_mask) # (batch, seq_len-1, d_model)
        x = torch.cat((x, text[:, 1:]), dim=2) # (batch, seq_len-1, d_model + text_size)        
        logits = self.mlp(x) # (batch, seq_len-1, choice_size)
        return logits


    def cal_loss(self, data):
        stimuli, text, choice, feedback = data
        output = self(data) # (batch-1, seq_len-1, choice_size)

        return F.cross_entropy(output.reshape(-1, self.hparams.choice_size), choice[:, 1:].reshape(-1))
    
    def training_step(self, batch, batch_idx):
        loss = self.cal_loss(batch)
        self.log('train_loss', loss, sync_dist=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.cal_loss(batch)
        self.log('val_loss', loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.cal_loss(batch)
        self.log('test_loss', loss, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        logits = self(batch)
        return logits
        
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    