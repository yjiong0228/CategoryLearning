import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

class CategoryRNN(pl.LightningModule):
    def __init__(self, text_size, hidden_size, choice_size, feedback_size,*, lr=0.0001):
        super(CategoryRNN, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.text_size = text_size
        self.choice_size = choice_size
        self.feedback_size = feedback_size
        self.rnn = nn.RNN(text_size + choice_size + feedback_size, hidden_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + text_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, choice_size)
        )

    def forward(self, data):
        """
        data: tuple of (text, choice, feedback)
        text: (batch, 768)
        choice: (batch, )
        feedback: (batch, )

        return: (batch, 4)
        """
        text, choice, feedback = data
        choice = F.one_hot(choice, num_classes=self.choice_size).float()
        # choice = torch.cat((torch.zeros(1, choice.size(1)).to(choice.device), choice), dim=0)[:-1]
        feedback = F.one_hot(feedback, num_classes=self.feedback_size).float()
        # feedback = torch.cat((torch.zeros(1, feedback.size(1)).to(feedback.device), feedback), dim=0)[:-1]
        x = torch.cat((text[:-1], choice[:-1], feedback[:-1]), dim=1) # (batch-1, 768 + 4 + 2)
        y, H = self.rnn(x) # (batch-1, hidden_size)， (1, hidden_size)
        # 3.0
        # x = torch.cat((H, text[-1].reshape(1,-1)), dim=1)
        # logits = self.mlp(x)
        # logits = logits.squeeze(0)
        # return logits, y
        # 3.1
        x = torch.cat((y, text[1:]), dim=1) # (batch-1, hidden_size + 768)
        logits = self.mlp(x)
        return logits


    def cal_loss(self, data):
        text, choice, feedback= data[0]
        output = self(data[0])

        return F.cross_entropy(output, choice[1:]) # + 0.01 * torch.sum(self.linear.weight[:, -6:-2] ** 2)
    
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
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    