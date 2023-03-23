import numpy as np
import SupportingFunctions as sf
import matplotlib.pyplot as plt
from espirit import espirit
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
from SupportingFunctions import nmse

# %% Functions
def fft2 (image, axis=[0,1]):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image, axes=axis), axes=axis, norm='ortho'), axes=axis)

# image = ifft2(kspace): iFFT of n-slice kspace to image: [n Nx Ny Nc] --> [n Nx Ny Nc]
def ifft2 (kspace, axis=[0,1]):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace, axes=axis), axes=axis, norm='ortho'), axes=axis)

# kspace --> image: [Nx Ny Nc] --> [Nx Ny]
def rssq (kspace):
    return np.sqrt(np.sum(np.abs(ifft2(kspace))**2,2))

def forward(image, Smaps, mask):
    return fft2(image[...,None]*Smaps)*mask[...,None]

def backward(kspace, Smaps):
    return np.sum(ifft2(kspace)*np.conj(Smaps), 2)

def cgsense(kspace,Smaps,mask,max_iter=25, lambd = 1e-3):
    a = backward(kspace,Smaps)
    p = np.copy(a)
    r_now = np.copy(a)
    xn = np.zeros_like(a)
    for i in np.arange(max_iter):
        delta = np.sum(r_now*np.conj(r_now))/np.sum(a*np.conj(a))
        if delta < 1e-5:
            break
        # q = (EHE + lambda*I)p
        q = backward(forward(p,Smaps,mask),Smaps) + lambd*p
        # rr_pq = r'r/p'q
        rr_pq = np.sum(r_now*np.conj(r_now))/np.sum(q*np.conj(p))
        xn = xn + rr_pq * p
        r_next = r_now - rr_pq * q
        # p = r_next + r_next'r_next/r_now'r_now
        p = r_next + (np.sum(r_next*np.conj(r_next))/np.sum(r_now*np.conj(r_now))) * p
        r_now = np.copy(r_next)
    return xn

def nmse(x,xref):
    return np.sum((np.abs(x)-np.abs(xref))**2) / np.sum(np.abs(xref)**2)

# %% HyperParameters
mask_type = 'circle' # 'circle' or 'rect'

# %% Data Loading
data = loadmat('realtime_data_small_tf.mat')
y_com = data['y_com']
x_com = data['x_com']
Smaps = data['Smaps']
acc_mask = data['acc_mask']
acc_mask4 = data['acc_mask4']
ovs_mask3 = data[mask_type + str(3)]
ovs_mask9 = data[mask_type + str(9)]

Nx = y_com.shape[0]             # num of y pixels (vertical)
Ny = y_com.shape[1]             # num of x pixels (horizontal)
Nc = y_com.shape[2]             # num of coils

# %% Outer Volume Subtraction
y = y_com * acc_mask[...,None]
y_background3 = fft2(ifft2(y_com)*ovs_mask3[...,None])
y_com_diff3 = y_com - fft2(ifft2(y_com)*ovs_mask3[:,:,None])
y_diff3 = y_com_diff3 * acc_mask[...,None]
y_background9 = fft2(ifft2(y_com)*ovs_mask9[...,None])
y_com_diff9 = y_com - fft2(ifft2(y_com)*ovs_mask9[:,:,None])
y_diff9 = y_com_diff9 * acc_mask[...,None]

# %% Generate coil maps with espirit
Smaps_full = Smaps[:,:,:,0]
Smaps_mask3 = Smaps_full * (1-ovs_mask3)[:,:,None]
Smaps_diff9 = espirit(y_com_diff9[None,...], 12, 40, 0.02, 0.95)
Smaps_diff9 = Smaps_diff9[0,:,:,:,0]

'''
# %% Visualize the data
# composite kspace
figure = plt.figure(); plt.imshow(np.log(np.abs(y_com[:,:,0])+1), cmap='gray'); plt.axis('off'); plt.title('composite kspace')
# composite image
figure = plt.figure(); plt.imshow(x_com, cmap='gray', vmax=1500); plt.axis('off'); plt.title('composite image')
# Smaps
figure = plt.figure(); plt.imshow(np.abs(Smaps[:,:,0,0]), cmap='gray'); plt.axis('off'); plt.title('sensitivity maps')
# acceleration mask R=8
figure = plt.figure(); plt.imshow(acc_mask, cmap='gray'); plt.axis('off'); plt.title('R=8')
# acceleration mask R=4
figure = plt.figure(); plt.imshow(acc_mask4, cmap='gray'); plt.axis('off'); plt.title('R=4')
# ovs mask sharp
figure = plt.figure(); plt.imshow(1-ovs_mask3, cmap='gray'); plt.axis('off'); plt.title('ovs mask, sharp')
# ovs mask smooth
figure = plt.figure(); plt.imshow(1-ovs_mask9, cmap='gray'); plt.axis('off'); plt.title('ovs mask, smooth')
# kspace R=8
figure = plt.figure(); plt.imshow(np.log(np.abs(y[:,:,0])+1), cmap='gray'); plt.axis('off'); plt.title('kspace, R=8')
'''

# %% results with cgsense
# no OVS processing
kfullSfull = np.abs(sf.cgsense(y, Smaps_full, acc_mask))
# OVS from k-space
kdiffSfull = np.abs(sf.cgsense(y_diff3, Smaps_full, acc_mask))*np.sqrt(1-ovs_mask3)
# OVS from k-space and calibration in image space
kdiffSmask = np.abs(sf.cgsense(y_diff3, Smaps_mask3, acc_mask))*(1-ovs_mask3)
# OVS from k-space and calibration in k-space 
kdiffSdiff = np.abs(sf.cgsense(y_diff3, Smaps_diff9, acc_mask))*np.sqrt(1-ovs_mask3)
# reference
reference = np.abs(sf.cgsense(y_com*acc_mask4[...,None], Smaps_full, acc_mask4))

# background = cgsense(y_background3*acc_mask[:,:,None],Smaps_full,acc_mask)*ovs_mask3
background = x_com
Nx_min = np.sum((1-ovs_mask3),1).nonzero()[0][0]
Nx_max = np.sum((1-ovs_mask3),1).nonzero()[0][-1]+1
Nx_center = int(np.mean([Nx_min, Nx_max]))
Ny_min = np.sum((1-ovs_mask3),0).nonzero()[0][0]
Ny_max = np.sum((1-ovs_mask3),0).nonzero()[0][-1]+1

data_range = np.max(reference[Nx_min:Nx_max,Ny_min:Ny_max])
props = dict(boxstyle='round', facecolor='black', alpha=0.8)

vmax = 1500
figure = plt.figure(figsize=(10,4.5));
plt.subplot(1,5,1); plt.imshow(kfullSfull, cmap='gray',vmax=vmax); plt.axis('off'); plt.title('kfullSfull')
ax = plt.gca()
NMSE = nmse(kfullSfull[Nx_min:Nx_max,Ny_min:Ny_max],reference[Nx_min:Nx_max,Ny_min:Ny_max])
SSIM = ssim(reference[Nx_min:Nx_max,Ny_min:Ny_max], kfullSfull[Nx_min:Nx_max,Ny_min:Ny_max], data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(1,5,2); plt.imshow(kdiffSfull+background*np.sqrt(ovs_mask3), cmap='gray',vmax=vmax); plt.axis('off'); plt.title('kdiffSfull3')
ax = plt.gca()
NMSE = nmse((kdiffSfull+background*np.sqrt(ovs_mask3))[Nx_min:Nx_max,Ny_min:Ny_max],reference[Nx_min:Nx_max,Ny_min:Ny_max])
SSIM = ssim(reference[Nx_min:Nx_max,Ny_min:Ny_max], (kdiffSfull+background*np.sqrt(ovs_mask3))[Nx_min:Nx_max,Ny_min:Ny_max], data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(1,5,3); plt.imshow(kdiffSmask+background*ovs_mask3, cmap='gray',vmax=vmax); plt.axis('off'); plt.title('kdiffSmask')
ax = plt.gca()
NMSE = nmse((kdiffSmask+background*ovs_mask3)[Nx_min:Nx_max,Ny_min:Ny_max],reference[Nx_min:Nx_max,Ny_min:Ny_max])
SSIM = ssim(reference[Nx_min:Nx_max,Ny_min:Ny_max], (kdiffSmask+background*ovs_mask3)[Nx_min:Nx_max,Ny_min:Ny_max], data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(1,5,4); plt.imshow(kdiffSdiff+background*np.sqrt(ovs_mask3), cmap='gray',vmax=vmax); plt.axis('off'); plt.title('kdiffSdiff')
ax = plt.gca()
NMSE = nmse((kdiffSdiff+background*np.sqrt(ovs_mask3))[Nx_min:Nx_max,Ny_min:Ny_max],reference[Nx_min:Nx_max,Ny_min:Ny_max])
SSIM = ssim(reference[Nx_min:Nx_max,Ny_min:Ny_max], (kdiffSdiff+background*np.sqrt(ovs_mask3))[Nx_min:Nx_max,Ny_min:Ny_max], data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(1,5,5); plt.imshow(reference, cmap='gray',vmax=vmax); plt.axis('off'); plt.title('tgrappa')
ax = plt.gca()
plt.xticks([])
plt.yticks([])


plt.suptitle('Results for circular mask'); plt.axis('off')
plt.tight_layout()




