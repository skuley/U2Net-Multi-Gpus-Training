from PIL import Image, ImageOps
import os
import os.path as osp
from torch.utils.data import Dataset
from glob import glob
import cv2
from torchvision.transforms import transforms as T
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, im_root='', gt_root='', transform=None):
        super(Dataset, self).__init__()
        assert osp.isdir(im_root) == True, f"{im_root} is not a directory"
        assert osp.isdir(gt_root) == True, f"{gt_root} is not a directory"

        self.im_root = im_root
        self.gt_root = gt_root

        self.images = sorted(glob(osp.join(im_root, '*.jpg')))
        self.gts = sorted(glob(osp.join(gt_root, '*.png')))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def _transform(self, image: str, gt: str):
        image, gt = Image.open(image), Image.open(gt)
        if self.transform:
            gt = ImageOps.grayscale(gt)
            image = self.transform(image)
            gt = self.transform(gt)
        
        tn_image = T.ToTensor()(image)
        tn_gt = T.ToTensor()(gt)
        return tn_image, tn_gt

    def __getitem__(self, item):
        image = self.images[item]
        gt = self.images[item]
        image, gt = self._transform(image, gt)
        return {'image': image, 'gt': gt}