import torch


# kspace = fft2(image): FFT of n-slice image to kspace: [n Nx Ny] --> [n Nx Ny Nc]
def fft2 (image, axis=[1,2]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image, dim=axis), dim=axis, norm='ortho'), dim=axis)

# kspace = ifft2(image): iFFT of n-slice kspace to image: [n Nx Ny Nc] --> [n Nx Ny]
def ifft2 (kspace, axis=[1,2]):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=axis), dim=axis, norm='ortho'), dim=axis)