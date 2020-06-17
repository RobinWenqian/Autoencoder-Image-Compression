from matplotlib import pyplot as plt
import numpy as np

import torch
import argparse
from torch.autograd import Variable
from math import log10
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms
#import train_eval
#from train_eval import to_img

from Models import autoencoder
from dataloader import DataloaderCompression
from Lossfuncs import mse_loss, parsingLoss

nb_channls = 3

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=8, help='batch size')
parser.add_argument(
    '--train', required=True, type=str, help='folder of training images')
parser.add_argument(
    '--test', required=True, type=str, help='folder of testing images')
parser.add_argument(
    '--max_epochs', type=int, default=50, help='max epochs')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=100, help='unroll iterations')
parser.add_argument(
    '--image_size', type=int, default=150, help='Load image size')
parser.add_argument('--checkpoint', type=int, default=20, help='save checkpoint after ')
parser.add_argument('--workers', type=int, default=4, help='unroll iterations')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='unroll iterations')
args = parser.parse_args()

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), nb_channls, args.image_size, args.image_size)
    return x

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

model=torch.load('compressing.pth')
model.eval()

Dataloader = DataloaderCompression(args.test,args.image_size,args.batch_size,args.workers)

PSNR = []
Compressing_Ratio = []
itr = 0
for data in Dataloader:
    img, _ = data        
    img = Variable(img).to(device)

    coding, output = model(img)
    cyclicloss,r_loss,g_loss,b_loss = mse_loss(output, img)

    PSNR_value = 10*log10(255**2/cyclicloss)
    PSNR.append(PSNR_value)

    Comp_ratio = coding.size()[1]/img.size()[1]
    Compressing_Ratio.append(Comp_ratio)

    pic_ = to_img(output.to("cpu").data)
    #pic = transforms.ToPILImage(pic_)
            
    #pic_color = np.transpose(vutils.make_grid(pic.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0))
    fig = plt.figure(figsize=(128, 128))

    '''       
    ax = plt.imshow(np.transpose(vutils.make_grid(pic.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    '''

    #plt.show(fig)
    plt.savefig('output/%d.jpg'%itr)
    itr += 1

print('mean PSNR is %s'%np.mean(PSNR))
print('mean compression ratio is %s'%np.mean(Compressing_Ratio))