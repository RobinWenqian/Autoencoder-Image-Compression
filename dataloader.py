from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torchvision import transforms

def DataloaderCompression(dataroot, image_size, batch_size, workers):
    #dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return dataloader