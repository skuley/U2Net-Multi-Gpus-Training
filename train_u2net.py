import os
import os.path as osp
import torch
import lightning as pl
from utils.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from model.segmentation import U2NetPL
import argparse

device = 'mps'
gpu_id = None
if torch.cuda.is_available():
    device = 'gpu'
    if torch.cuda.device_count() > 1:
        gpu_id = [0]
    else:
        gpu_id = [number for number in range(torch.cuda.device_count())]

wandb_logger = WandbLogger(name='DIS Dataset',project='U2-Net')

def load_dataloader(args):
    tr_im_path = args.tr_im_path
    tr_gt_path = args.tr_gt_path
    vd_im_path = args.vd_im_path
    vd_gt_path = args.vd_gt_path
    
    tr_transform = T.Compose([
        T.Resize((320,320)),
        T.RandomCrop((288,288))
    ])

    vd_transform = T.Compose([
        T.Resize((320, 320))
    ])
    
    tr_ds = Dataset(im_root=tr_im_path, gt_root=tr_gt_path, transform=tr_transform)
    vd_ds = Dataset(im_root=vd_im_path, gt_root=vd_gt_path, transform=vd_transform)
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=8)
    vd_dl = DataLoader(vd_ds, args.batch_size, shuffle=False, num_workers=4)
    
    return tr_dl, vd_dl

def load_model(args):
    pretrained_path = args.pretrained_path
    save_model_path = args.save_model_path
    u2net = U2NetPL(pretrained=pretrained_path)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=save_model_path,
        filename="{epoch:02d}-{val_loss:.2f}-" + f"batch_size={str(batch_size)}",
        save_top_k=3,
        mode="min"
    )

    trainer = pl.Trainer(logger=wandb_logger,
             callbacks=[checkpoint_callback, early_stop_callback],
             devices=None,
             accelerator=device,
             min_epochs=min_epoch,
             max_epochs=max_epoch,
             profiler='simple')
    
    return trainer, model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U2-Net Training')
    parser.add_argument('--epoch',             type=int,      default=100)
    parser.add_argument('--batch_size',        type=int,      default=8)
    parser.add_argument('--lr',                type=float,    default=0.0001)
    parser.add_argument('--ep',                type=float,    default=0.0001)
    parser.add_argument('--tr_im_path',        type=str,      default='')
    parser.add_argument('--tr_gt_path',        type=str,      default='')
    parser.add_argument('--vd_im_path',        type=str,      default='')
    parser.add_argument('--vd_gt_path',        type=str,      default='')
    parser.add_argument('--pretrained_path',   type=str,      default='')
    parser.add_argument('--save_weight_path',   type=str,      default='')
    args = parser.parse_args()

    # dataloader
    tr_dl, vd_dl = load_dataloader(args)
    
    # model
    trainer, model = load_model(args)
    
    # run
    trainer.fit(model, tr_dl, vd_dl)

    
    