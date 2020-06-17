from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.utils import save_image
import torchvision.utils as vutils
from torchsummary import summary
import argparse
import sys
from math import log10

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

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), nb_channls, args.image_size, args.image_size)
    return x

Dataloader = DataloaderCompression(args.train,args.image_size,args.batch_size,args.workers)

model = autoencoder().to(device)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
summary(model, (nb_channls, args.image_size, args.image_size))

# Training Loop. Results will appear every 10th iteration.
itr = 0
training_loss = []
PSNR_list = []
for epoch in range(args.max_epochs):
    for data in Dataloader:
        img, _ = data        
        img = Variable(img).to(device)

        # Forward
        coding, output = model(img)
        cyclicloss,r_loss,g_loss,b_loss = mse_loss(output, img)
        pLoss = parsingLoss(coding, args.image_size)
        
        loss = 5*cyclicloss + 10*pLoss

        PSNR = 10*log10(255**2/cyclicloss)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''
        if itr % 10 == 0 and itr < args.iterations:
            # Log
            print('iter [{}], whole_loss:{:.4f} cyclic_loss{:.4f} pLoss{:.4f} comp_ratio{:.4f}'
              .format(itr, loss.data.item(), 5*cyclicloss.data.item(), 10*pLoss.data.item(), PSNR))
        '''
        '''
        if itr % 30 == 0 and itr < args.iterations: 
            pic = to_img(output.to("cpu").data)
            
            fig = plt.figure(figsize=(16, 16))
            
            ax = plt.imshow(np.transpose(vutils.make_grid(pic.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.show(fig)
            
            
            compress_ratio.append(comp_ratio)
        '''
        training_loss.append(loss)
        PSNR_list.append(PSNR)
        itr += 1
    
        print('epoch [{}/{}], loss:{:.4f}, cyclic_loss{:.4f} pLoss{:.4f} PSNR{:.4f}'
            .format(epoch + 1, args.max_epochs, loss.data.item(), 5*cyclicloss.data.item(), 10*pLoss.data.item(), PSNR))

    if epoch % 10 == 0:
        torch.save(model, 'Compressing_{%d}.pth'%epoch)

plt.plot(training_loss, label='Training loss')
plt.plot(PSNR, label='PSNR')
plt.legend(frameon=False)
plt.savefig("Train.png")
plt.show()