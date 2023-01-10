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


device = 'mps'
gpu_id = None
if torch.cuda.is_available():
    device = 'gpu'
    if torch.cuda.device_count() > 1:
        gpu_id = [0]
    else:
        gpu_id = [number for number in range(torch.cuda.device_count())]

wandb_logger = WandbLogger(name='DIS Dataset',project='U2-Net')

tr_path_im = '/Users/hongsung-yong/Library/Mobile Documents/com~apple~CloudDocs/dataset/DIS5K/DIS-TR/im'
tr_path_gt = '/Users/hongsung-yong/Library/Mobile Documents/com~apple~CloudDocs/dataset/DIS5K/DIS-TR/gt'
val_path_im = '/Users/hongsung-yong/Library/Mobile Documents/com~apple~CloudDocs/dataset/DIS5K/DIS-VD/im'
val_path_gt = '/Users/hongsung-yong/Library/Mobile Documents/com~apple~CloudDocs/dataset/DIS5K/DIS-VD/gt'

save_model_path = 'saved_model/u2net'
os.makedirs(save_model_path, exist_ok=True)

batch_size = 8
min_epoch = 10
max_epoch = 20

pretrained_path = 'saved_model/pretrained/u2net.pth'

tr_transform = T.Compose([
    T.Resize((320,320)),
    T.RandomCrop((288,288)),
    T.ToTensor()
])

val_transform = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor()
])

tr_ds = Dataset(im_root=tr_path_im, gt_root=tr_path_gt, transform=tr_transform, load_on_mem=False)
val_ds = Dataset(im_root=val_path_im, gt_root=val_path_gt, transform=val_transform, load_on_mem=False)

tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=8)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# model
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

trainer.fit(u2net, tr_dl, val_dl)
