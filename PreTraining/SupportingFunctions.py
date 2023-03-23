import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from random import sample
from scipy.io import loadmat

### DEFINE FFT2 AND IFFT2 FUNCTIONS
# y = FFT(x): FFT of one slice image to kspace: [1 Nx Ny Nc] --> [1 Nx Ny Nc]
def fft2 (image, axis=[1,2]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image, dim=axis), dim=axis, norm='ortho'), dim=axis)

# x = iFFT(y): iFFT of one slice kspace to image: [1 Nx Ny Nc] --> [1 Nx Ny Nc]
def ifft2 (kspace, axis=[1,2]):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=axis), dim=axis, norm='ortho'), dim=axis)

# y = Ex: encoding one slice image to kspace: [1 Nx Ny] --> [1 Nx Ny Nc]
# S: sensitivity map
def encode(x,S,mask):
    if mask==None:
        return fft2(S*x[:,:,:,None])
    else:
        return fft2(S*x[:,:,:,None])*mask[None,:,:,None]

# y = E'x: reconstruction from kspace to image space: [1 Nx Ny Nc] --> [1 Nx Ny]
# S: sensitivity map
def decode(x,S):
    return torch.sum(ifft2(x)*torch.conj(S), axis=3)

# Normalised Mean Square Error (NMSE)
# gives the nmse between x and xref
def nmse(x,xref):
    return np.sum((x-xref)**2) / np.sum((xref)**2)

class TrainImages():
    def __init__(self,data_path,num_slice):
        self.dir_list  = os.listdir(data_path)
        if num_slice == 'all':
            self.slices = self.dir_list
        else:
            self.slices    = sample(self.dir_list, num_slice)
        self.num_slice = len(self.slices) 
        self.data_path = data_path
     
    def __getitem__(self,index):
        slice_data = loadmat(self.data_path + os.sep + self.slices[index])
        self.reference = torch.from_numpy(slice_data['reference'])
        self.noisy = torch.from_numpy(slice_data['noisy'])
        self.sub_slc_tf = torch.from_numpy(slice_data['sub_slc_tf'])
        return self.reference, self.noisy, self.sub_slc_tf, index
    
    def __len__(self):
        return self.num_slice   

# train dataset loader
def prepare_train_loaders(train_dataset,params):
    train_loader  = DataLoader(dataset       = train_dataset,
                             batch_size      = params['batch_size'],
                             shuffle         = True,
                             drop_last       = True,
                             #worker_init_fn  = seed_worker,
                             num_workers     = params['num_workers'])
    
    datasets = dict([('train_dataset', train_dataset)])  
    
    loaders = dict([('train_loader', train_loader)])

    return loaders, datasets

class ValidationImages():
    def __init__(self,data_path,num_slice):
        self.dir_list  = os.listdir(data_path)
        if num_slice == 'all':
            self.slices = self.dir_list
        else:
            self.slices    = sample(self.dir_list, num_slice)
        self.num_slice = len(self.slices) 
        self.data_path = data_path
     
    def __getitem__(self,index):
        slice_data = loadmat(self.data_path + os.sep + self.slices[index])
        self.reference = torch.from_numpy(slice_data['reference'])
        self.noisy = torch.from_numpy(slice_data['noisy'])
        self.sub_slc_tf = torch.from_numpy(slice_data['sub_slc_tf'])
        return self.reference, self.noisy, self.sub_slc_tf, index
    
    def __len__(self):
        return self.num_slice   

# valid dataset loader
def prepare_valid_loaders(valid_dataset,params):
    valid_loader  = DataLoader(dataset       = valid_dataset,
                             batch_size      = params['batch_size'],
                             shuffle         = True,
                             drop_last       = True,
                             #worker_init_fn  = seed_worker,
                             num_workers     = params['num_workers'])
    
    datasets = dict([('valid_dataset', valid_dataset)])  
    
    loaders = dict([('valid_loader', valid_loader)])

    return loaders, datasets

class TestImages():
    def __init__(self,data_path,num_slice):
        self.dir_list  = os.listdir(data_path)
        if num_slice == 'all':
            self.slices = self.dir_list
        else:
            self.slices    = sample(self.dir_list, num_slice)
        self.num_slice = len(self.slices) 
        self.data_path = data_path
     
    def __getitem__(self,index):
        slice_data = loadmat(self.data_path + os.sep + self.slices[index])
        self.reference = torch.from_numpy(slice_data['reference'])
        self.noisy = torch.from_numpy(slice_data['noisy'])
        self.sub_slc_tf = torch.from_numpy(slice_data['sub_slc_tf'])
        return self.reference, self.noisy, self.sub_slc_tf, index
    
    def __len__(self):
        return self.num_slice   

# test dataset loader
def prepare_test_loaders(test_dataset,params):
    test_loader  = DataLoader(dataset        = test_dataset,
                             batch_size      = params['batch_size'],
                             shuffle         = False,
                             drop_last       = True,
                             #worker_init_fn  = seed_worker,
                             num_workers     = params['num_workers'])
    
    datasets = dict([('test_dataset', test_dataset)])  
    
    loaders = dict([('test_loader', test_loader)])

    return loaders, datasets


# complex 1 channel to real 2 channels
def ch1to2(data1):       
    return torch.cat((data1.real,data1.imag),0)
# real 2 channels to complex 1 channel
def ch2to1(data2):       
    return data2[0:1,:,:] + 1j * data2[1:2,:,:] 


# Normalised L1-L2 loss calculation
# loss = normalised L1 loss + normalised L2 loss
def L1L2Loss(ref, recon):
    return torch.norm(recon-ref,p=1)/torch.norm(ref,p=1) + torch.norm(recon-ref,p=2)/torch.norm(ref,p=2)

