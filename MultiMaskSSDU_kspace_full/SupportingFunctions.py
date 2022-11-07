import torch


# kspace = fft2(image): FFT of n-slice image to kspace: [n Nx Ny] --> [n Nx Ny Nc]
def fft2 (image, axis=[1,2]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image, dim=axis), dim=axis, norm='ortho'), dim=axis)

# image = ifft2(kspace): iFFT of n-slice kspace to image: [n Nx Ny Nc] --> [n Nx Ny]
def ifft2 (kspace, axis=[1,2]):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=axis), dim=axis, norm='ortho'), dim=axis)

# kspace = forward(image): encoding one slice image to kspace: [2 Nx Ny] --> [1 Nx Ny Nc]
# Smaps: sensitivity map [2 Nx Ny Nc]
# images are 2 many because the Smaps are 2 sets
def forward(image,Smaps,mask):
    return fft2(torch.sum(image[...,None]*Smaps,0)[None,...])*mask[None,:,:,None]

# image = backward(kspace): reconstruction from kspace to image space: [1 Nx Ny Nc] --> [2 Nx Ny]
def backward(kspace,Smaps):
    return torch.sum(ifft2(kspace)*torch.conj(Smaps),3)