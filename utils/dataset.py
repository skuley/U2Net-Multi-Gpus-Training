from PIL import Image, ImageOps
import os
import os.path as osp
from torch.utils.data import Dataset
from glob import glob
import cv2
from torchvision.transforms import transforms as T
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, im_root='', gt_root='', transform=None, load_on_mem=False):
        super(Dataset, self).__init__()
        assert osp.isdir(im_root) == True, f"{im_root} is not a directory"
        assert osp.isdir(gt_root) == True, f"{gt_root} is not a directory"

        self.im_root = im_root
        self.gt_root = gt_root

        self.images = sorted(glob(osp.join(im_root, '*.jpg')))
        self.gts = sorted(glob(osp.join(gt_root, '*.png')))

        self.transform = transform
        self.load_on_mem = load_on_mem

        if self.load_on_mem:
            self.mem_images, self.mem_gts = [], []
            print('loading datasets on memory..')
            for image, gt in tqdm(zip(self.images, self.gts), total=len(self)):
                tn_image, tn_gt = self._transform(image, gt)
                self.mem_images.append(tn_image)
                self.mem_gts.append(tn_gt)

    def __len__(self):
        return len(self.images)

    def _transform(self, image: str, gt: str):
        if self.transform:
            image, gt = Image.open(image), Image.open(gt)
            gt = ImageOps.grayscale(gt)
            tn_image = self.transform(image)
            tn_gt = self.transform(gt)
        return tn_image, tn_gt

    def __getitem__(self, item):
        if self.load_on_mem:
            image, gt = self.mem_images[item], self.mem_gts[item]
        else:
            image = self.images[item]
            gt = self.images[item]
            image, gt = self._transform(image, gt)
        return {'image': image, 'gt': gt}