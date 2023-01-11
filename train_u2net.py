import os
import os.path as osp
import torch
import lightning as pl
from utils.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
import albumentations as A

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from model.segmentation import U2NetPL
import argparse

def load_dataloader(args):
    tr_transform = A.Compose([
        A.Resize(width=320, height=320),
        A.RandomCrop(width=288, height=288)
    ])

    vd_transform = A.Compose([
        A.Resize(width=320, height=320)
    ])
    
    tr_ds = Dataset(im_root=args.tr_im_path, gt_root=args.tr_gt_path, transform=tr_transform)
    vd_ds = Dataset(im_root=args.vd_im_path, gt_root=args.vd_gt_path, transform=vd_transform)
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=8)
    vd_dl = DataLoader(vd_ds, args.batch_size, shuffle=False, num_workers=4)
    
    return tr_dl, vd_dl

def load_model(args):
    os.makedirs(args.save_weight_path, exist_ok=True)
    u2net = U2NetPL(pretrained=args.pretrained_path, lr=args.lr, epsilon=args.epsilon)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_weight_path,
        filename="{epoch:02d}-{val_loss:.2f}-" + f"batch_size={str(args.batch_size)}",
        save_top_k=3,
        mode="min"
    )
    
    wandb_logger = WandbLogger(name='DUTS Dataset',project='U2-Net')
    trainer = pl.Trainer(logger=wandb_logger,
             callbacks=[checkpoint_callback, early_stop_callback],
#              devices=torch.cuda.device_count(), strategy='ddp',
             devices=[1,2], strategy='ddp',
             accelerator='gpu',
             min_epochs=args.min_epoch,
             max_epochs=args.max_epoch,
             profiler='simple')
    
    return trainer, u2net
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U2-Net Training')
    parser.add_argument('--min_epoch',         type=int,      default=100)
    parser.add_argument('--max_epoch',         type=int,      default=200)
    parser.add_argument('--batch_size',        type=int,      default=64)
    parser.add_argument('--lr',                type=float,    default=0.0001)
    parser.add_argument('--epsilon',           type=float,    default=1e-08)
    parser.add_argument('--tr_im_path',        type=str,      default='')
    parser.add_argument('--tr_gt_path',        type=str,      default='')
    parser.add_argument('--vd_im_path',        type=str,      default='')
    parser.add_argument('--vd_gt_path',        type=str,      default='')
    parser.add_argument('--pretrained_path',   type=str,      default='')
    parser.add_argument('--save_weight_path',  type=str,      default='')
    
    args = parser.parse_args()
    
    # dataloader
    tr_dl, vd_dl = load_dataloader(args)
    
    # model
    trainer, model = load_model(args)
    
    # run
    trainer.fit(model, tr_dl, vd_dl)

    
    