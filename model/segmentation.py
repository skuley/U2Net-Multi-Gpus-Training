from typing import Dict, Any

from model.u2net import U2NET, U2NETP
# import lightning as pl
import pytorch_lightning as pl
import torch
import torch.nn as nn
from collections import OrderedDict as OD


bce_loss = nn.BCELoss()



class U2NetPL(pl.LightningModule):
    def __init__(self, pretrained: str = None, lr: float = 0.001, epsilon: float = 1e-08) -> object:
        super(U2NetPL, self).__init__()
        self.lr = lr
        self.epsilon = epsilon

        self.u2net = U2NET(3,1)
        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            self.u2net.load_state_dict(state_dict)
            print('pretrained loaded')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.u2net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=self.epsilon, weight_decay=0)
        return optimizer

    def forward(self, x):
        return self.u2net(x)

    def _common_step(self, batch, batch_idx, stage: str):
        image, gt = batch['image'], batch['gt']
        outputs = self.u2net(image)
        loss = self.loss(outputs, gt)
        self.log(f"{stage}_loss", loss, on_epoch=True)
        return loss

    def loss(self, outputs, target):
        total_loss = 0.0
        for output in outputs:
            loss = bce_loss(output, target)
            total_loss += loss
        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "train")
        # self.log_dict({'loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "val")
        # self.log_dict({'val_loss': loss})
        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint['state_dict']
        key_word = 'u2net.'
        new_sd = OD()
        for key, value in state_dict.items():
            if key_word in key:
                key = key.replace(key_word, '')
            new_sd[key] = value
        checkpoint['state_dict'] = new_sd


if __name__ == '__main__':
    u2net = U2NetPL()
    print(u2net)
