import os
from model.u2net import U2NET

import torch
from torchvision.transforms import transforms as T

from PIL import Image
import numpy as np
import cv2

import argparse

def tensor2np(tensor_img, dst_size):
    img_np = np.array(tensor_img.cpu().detach().squeeze(0)*255, np.uint8)
    img_np = img_np.transpose(1,2,0).squeeze()
    img_np = cv2.resize(img_np, dsize=(dst_size))
    return img_np

def img2tensor(img_path, img_size):
    pil_image = Image.open(img_path)
    transform = T.Compose([
        T.Resize((img_size,img_size)),
        T.ToTensor()
    ])
    tn_img = transform(pil_image).unsqueeze(0)
    return tn_img, pil_image.size

def inference(model_weight, device):
    state_dict = torch.load(model_weight, map_location='cpu')
    u2net = U2NET(3,1)
    u2net.load_state_dict(state_dict)
    device_id = device
    device = f'cuda:{str(device)}'
    u2net = u2net.to(device)
    u2net.eval()
    
    with torch.no_grad():
        output = u2net(tn_img.to(device))
    pred = output[0]    
    
    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U2-Net Inference')
    parser.add_argument('--img_path',        type=str,      default='')
    parser.add_argument('--img_size',        type=int,      default=320)
    parser.add_argument('--device',          type=int,      default=0)
    parser.add_argument('--model_weight',    type=str,      default='saved_model/duts/epoch=02-val_loss=1.84-batch_size=32.ckpt')
    parser.add_argument('--save_path',       type=str,      default='output')
    args = parser.parse_args()
    
    tn_img, init_size = img2tensor(args.img_path, args.img_size)
    pred = inference(args.model_weight, args.device)
    pred_np = tensor2np(pred, init_size)
    
    save_path = os.path.join(os.getcwd(), args.save_path)
    dst_img_path = os.path.basename(args.img_path)
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(save_path, dst_img_path), pred_np)
    
    