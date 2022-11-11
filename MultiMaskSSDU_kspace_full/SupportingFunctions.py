import torch
import os
from random import sample
from scipy.io import loadmat
from torch.utils.data import DataLoader


# kspace = fft2(image): FFT of n-slice image to kspace: [n Nx Ny Nc] --> [n Nx Ny Nc]
def fft2 (image, axis=[1,2]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image, dim=axis), dim=axis, norm='ortho'), dim=axis)

# image = ifft2(kspace): iFFT of n-slice kspace to image: [n Nx Ny Nc] --> [n Nx Ny Nc]
def ifft2 (kspace, axis=[1,2]):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=axis), dim=axis, norm='ortho'), dim=axis)

# kspace = forward(image): encoding one slice image to kspace: [2 Nx Ny] --> [1 Nx Ny Nc]
# Smaps: sensitivity map [2 Nx Ny Nc]
# mask: acceleration mask [Nx Ny]
# images are 2 many because the Smaps are 2 sets
def forward(image,Smaps,mask):
    return fft2(torch.sum(image[...,None]*Smaps,0)[None,...])*mask[None,:,:,None]

# image = backward(kspace): reconstruction from kspace to image space: [1 Nx Ny Nc] --> [2 Nx Ny]
def backward(kspace,Smaps):
    return torch.sum(ifft2(kspace)*torch.conj(Smaps),3)

# dataset generator
class OVS_DatasetTrain():
    def __init__(self,data_path,num_slice):
        self.dir_list = os.listdir(data_path)
        self.slices = sample(self.dir_list, self.num_slices)
          
    def __getitem__(self,index):
        slice_data = loadmat(self.data_path + os.path + self.slices[index])
        self.composite_kspace = slice_data['composite_kspace']
        self.sense_maps = slice_data['sense_maps']
        self.acc_mask = slice_data['acc_mask']
        self.data_consistency_masks = slice_data['data_consistency_masks']
        self.sub_slc_tf = slice_data['sub_slc_tf']
        Nx, Ny = self.acc_mask.shape
        K = (self.data_consistency_masks).shape(2)
        acc_kspace = self.composite_kspace[...,None]*self.data_consistency_masks[None,:,:,None,:]
        self.x0 = torch.zeros([K,Nx,Ny], dtype=torch.complex64)
        for k in range(K):
            self.x0[k:k+1] = backward(acc_kspace[...,k],self.sense_maps)
        
        return self.x0, self.composite_kspace, self.sense_map, self.acc_mask, self.data_consistency_masks, self.sub_slc_tf, index
    
# train dataset loader
def prepare_train_loaders(dataset,params,g):
    train_num  = int(dataset.n_slices * 0.8)
    valid_num  = dataset.n_slices - train_num

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_num,valid_num],  generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(dataset       = train_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = True,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)

    valid_loader = DataLoader(dataset       = valid_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = True,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)

    full_loader= DataLoader(dataset         = dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = True,
                            drop_last       = False,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'],
                            generator       = g)
    
    datasets = dict([('train_dataset', train_dataset),
                     ('valid_dataset', valid_dataset)])  
    
    loaders = dict([('train_loader', train_loader),
                    ('valid_loader', valid_loader),
                    ('full_loader', full_loader)])

    return loaders, datasets

