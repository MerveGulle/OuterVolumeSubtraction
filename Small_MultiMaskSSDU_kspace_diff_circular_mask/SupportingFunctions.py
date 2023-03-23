import torch
import os
from random import sample
from scipy.io import loadmat
from torch.utils.data import DataLoader
import numpy as np


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

# train dataset generator
class OVS_DatasetTrain():
    def __init__(self,data_path,sense_maps_type,num_slice):
        self.dir_list  = os.listdir(data_path)
        if num_slice == 'all':
            self.slices = self.dir_list
        else:
            self.slices    = sample(self.dir_list, num_slice)
        self.num_slice = len(self.slices) 
        self.data_path = data_path
        self.sense_maps_type = sense_maps_type
          
    def __getitem__(self,index):
        slice_data                  = loadmat(self.data_path + os.sep + self.slices[index])
        self.diff_com_kspace        = torch.from_numpy(slice_data['diff_com_kspace'])
        if (self.sense_maps_type == 'sense_maps_mask'):
            self.sense_maps         = torch.from_numpy(slice_data['sense_maps_full'])*torch.from_numpy(1.0-slice_data['ovs_mask3'])[None,:,:,None]
        else:
            self.sense_maps             = torch.from_numpy(slice_data[self.sense_maps_type])
        self.acc_mask               = torch.from_numpy(slice_data['acc_mask'])
        self.data_consistency_masks = torch.from_numpy(slice_data['data_consistency_masks'])
        self.sub_slc_tf             = torch.from_numpy(slice_data['sub_slc_tf'])
        Nx, Ny = self.acc_mask.shape
        K = (self.data_consistency_masks).shape[2]
        acc_kspace = self.diff_com_kspace[...,None]*self.data_consistency_masks[None,:,:,None,:]
        self.x0 = torch.zeros([K,Nx,Ny], dtype=torch.complex64)
        for k in range(K):
            self.x0[k:k+1] = backward(acc_kspace[...,k],self.sense_maps)
        
        return self.x0, self.diff_com_kspace, self.sense_maps, self.acc_mask, self.data_consistency_masks, self.sub_slc_tf, index
    
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

# validation dataset generator
class OVS_DatasetValidation():
    def __init__(self,data_path,sense_maps_type,num_slice):
        self.dir_list = os.listdir(data_path)
        if num_slice == 'all':
            self.slices = self.dir_list
        else:
            self.slices    = sample(self.dir_list, num_slice)
        self.num_slice = len(self.slices) 
        self.data_path = data_path
        self.sense_maps_type = sense_maps_type
          
    def __getitem__(self,index):
        slice_data                  = loadmat(self.data_path + os.sep + self.slices[index])
        self.diff_com_kspace        = torch.from_numpy(slice_data['diff_com_kspace'])
        if (self.sense_maps_type == 'sense_maps_mask'):
            self.sense_maps         = torch.from_numpy(slice_data['sense_maps_full'])*torch.from_numpy(1.0-slice_data['ovs_mask3'])[None,:,:,None]
        else:
            self.sense_maps             = torch.from_numpy(slice_data[self.sense_maps_type])
        self.acc_mask               = torch.from_numpy(slice_data['acc_mask'])
        self.data_consistency_masks = torch.from_numpy(slice_data['data_consistency_masks'])
        self.sub_slc_tf             = torch.from_numpy(slice_data['sub_slc_tf'])
        Nx, Ny = self.acc_mask.shape
        K = (self.data_consistency_masks).shape[2]
        acc_kspace = self.diff_com_kspace[...,None]*self.data_consistency_masks[None,:,:,None,:]
        self.x0 = torch.zeros([K,Nx,Ny], dtype=torch.complex64)
        for k in range(K):
            self.x0[k:k+1] = backward(acc_kspace[...,k],self.sense_maps)
        
        return self.x0, self.diff_com_kspace, self.sense_maps, self.acc_mask, self.data_consistency_masks, self.sub_slc_tf, index
    
    def __len__(self):
        return self.num_slice 
  

# validation dataset loader
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
    
# train dataset loader
''' 
# train + valid
def prepare_train_loaders(dataset,params,g):
    train_num  = int(dataset.num_slice * 0.8)
    valid_num  = dataset.num_slice - train_num

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
'''

# test dataset generator
class OVS_DatasetTest():
    def __init__(self,data_path,sense_maps_type,num_slice):
        self.dir_list = os.listdir(data_path)
        if num_slice == 'all':
            self.slices = self.dir_list
        else:
            self.slices    = sample(self.dir_list, num_slice)
        self.num_slice = len(self.slices) 
        self.data_path = data_path
        self.sense_maps_type = sense_maps_type
          
    def __getitem__(self,index):
        slice_data           = loadmat(self.data_path + os.sep + self.slices[index])
        self.kspace_data8 = torch.from_numpy(slice_data['kspace_data8'])
        if (self.sense_maps_type == 'sense_maps_mask'):
            self.sense_maps  = torch.from_numpy(slice_data['sense_maps_full'])*torch.from_numpy(1.0-slice_data['ovs_mask3'])[None,:,:,None]
        else:
            self.sense_maps  = torch.from_numpy(slice_data[self.sense_maps_type])
        self.composite_image = torch.from_numpy(slice_data['composite_image'])
        self.acc_mask        = torch.from_numpy(slice_data['acc_mask'])
        self.ovs_mask        = torch.from_numpy(slice_data['ovs_mask3'])
        self.im_tgrappa      = torch.from_numpy(slice_data['im_tgrappa'])
        self.sub_slc_tf      = torch.from_numpy(slice_data['sub_slc_tf'])
        self.x0              = backward(self.kspace_data8, self.sense_maps)
        
        return self.x0, self.kspace_data8, self.sense_maps, self.composite_image, self.acc_mask, self.ovs_mask, self.im_tgrappa, self.sub_slc_tf, index
    
    def __len__(self):
        return self.num_slice
  

# test dataset loader
def prepare_test_loaders(test_dataset,params):
    test_loader  = DataLoader(dataset       = test_dataset,
                            batch_size      = params['batch_size'],
                            shuffle         = False,
                            drop_last       = True,
                            #worker_init_fn  = seed_worker,
                            num_workers     = params['num_workers'])
    
    datasets = dict([('test_dataset', test_dataset)])  
    
    loaders = dict([('test_loader', test_loader)])

    return loaders, datasets


def cgsense(x0,kspace,Smaps,mask,max_iter=25, lambd = 1e-3):
    a = x0
    p = torch.clone(a)
    r_now = torch.clone(a)
    xn = torch.zeros_like(a)
    for i in range(max_iter):
        delta = torch.sum(r_now*torch.conj(r_now))/torch.sum(a*torch.conj(a)).abs()
        if delta.real < 1e-5:
            break
        # q = (EHE + lambda*I)p
        q = backward(forward(p,Smaps,mask),Smaps) + lambd*p
        # rr_pq = r'r/p'q
        rr_pq = torch.sum(r_now*torch.conj(r_now))/torch.sum(q*torch.conj(p))
        xn = xn + rr_pq * p
        r_next = r_now - rr_pq * q
        # p = r_next + r_next'r_next/r_now'r_now
        p = r_next + (torch.sum(r_next*torch.conj(r_next))/torch.sum(r_now*torch.conj(r_now))) * p
        r_now = torch.clone(r_next)
    return xn


# Normalised Mean Square Error (NMSE)
# gives the nmse between x and xref
def nmse(x,xref):
    return np.sum((np.abs(x)-np.abs(xref))**2) / np.sum(np.abs(xref)**2)