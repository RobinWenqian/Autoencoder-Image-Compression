import torch

def mse_loss(input, target):
    r = input[:,0:1,:,:] - target[:,0:1,:,:]
    g = input[:,1:2,:,:] - target[:,1:2,:,:]
    b = input[:,2:3,:,:] - target[:,2:3,:,:]
    
    r = torch.mean(r**2)
    g = torch.mean(g**2)
    b = torch.mean(b**2)
    
    mean = (r + g + b)/3
   
    return mean, r,g,b

def parsingLoss(coding, image_size):
    return torch.sum(torch.abs(coding))/(image_size**2)