from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
from skimage.metrics import structural_similarity as ssim
from SupportingFunctions import nmse
import matplotlib.pyplot as plt
import os
import mat73
from espirit_shifted import espirit


# %% Hyperparameters
filename = "subject_8_slice_3.mat"
time_frame = 11
Nofy = 72

# %% Functions
def fft (image, axis=0):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image, axes=axis), axes=axis, norm='ortho'), axes=axis)

# image = ifft2(kspace): iFFT of n-slice kspace to image: [n Nx Ny Nc] --> [n Nx Ny Nc]
def ifft (kspace, axis=0):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace, axes=axis), axes=axis, norm='ortho'), axes=axis)

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
        if delta < 1e-6:
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

def kernel_cal(data_calib, R=4, Kx=5):
    Cx, Cy, Nc = data_calib.shape
    kernel = np.zeros((R-1, Nc, Kx*2*Nc), dtype=complex)
    ACS = np.zeros(((Cx-Kx+1)*(Cy-R), Kx*2*Nc), dtype=complex)
    for x in np.arange(Cx-Kx+1):
        for y in np.arange(Cy-R):
            ACS[x*(Cy-R)+y] = data_calib[x:x+Kx,[y,y+R],:].reshape(1,-1)
    iACS = np.linalg.pinv(ACS)
    for y in np.arange(R-1):
        for c in np.arange(Nc):
            kernel[y,c] = np.matmul(iACS,data_calib[(Kx-1)//2:-(Kx-1)//2,y+1:Cy-R+y+1,c].reshape(-1,1)).reshape(1,1,-1)
    return kernel
    
def grappa(kspace, kernel, time_frame, R=4, Kx=5):
    Nx, Ny, Nc = kspace.shape
    shift = np.mod(time_frame, R)
    Nl = np.mod(Ny-shift-1, R)
    ACS = np.zeros(((Nx-Kx+1)*(Ny//R-1), Kx*2*Nc), dtype=complex)
    for x in np.arange(Nx-Kx+1):
        for y in np.arange(Ny//R-1):
            ACS[x*(Ny//R-1)+y] = kspace[x:x+Kx,[y*R+shift,y*R+R+shift],:].reshape(1,-1)
    kspace_new = np.copy(kspace)
    for y in np.arange(R-1):
        for c in np.arange(Nc):
            # Ny = 72
            kspace_new[(Kx-1)//2:-(Kx-1)//2,y+1+shift:Ny-Nl-R+y+1:R,c] = np.matmul(ACS,kernel[y,c].reshape(-1,1)).reshape(Nx-Kx+1,-1)
            # Ny = 68
            # kspace_new[(Kx-1)//2:-(Kx-1)//2,y+1+shift:Ny-Nl+y+1:R,c] = np.matmul(ACS,kernel[y,c].reshape(-1,1)).reshape(Nx-Kx+1,-1)
    return kspace_new

def nmse(x,xref):
    return np.sum((np.abs(x)-np.abs(xref))**2) / np.sum(np.abs(xref)**2)

def median(array, max_iter = 10):
    eps = np.max(np.abs(array))/1e6
    med_array = np.zeros_like(array[...,0])
    w = np.mean(array,axis=-1)
    for t in range(max_iter):
        w_old = w
        w = np.sum(array/(np.abs(w_old[...,None]-array)+eps),axis=-1) / np.sum(1/(np.abs(w_old[...,None]-array)+eps),axis=-1)
        med_array = w
    return med_array

# %% Upload the whole kspace data
os.chdir('C:\\Codes\\p006_OVS\\OVS')
slice_data = mat73.loadmat(filename)
_, data = list(slice_data.items())[1]
# undersample and normalise the kspace
kspace_data = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(data,axes=0), norm='ortho', axis=0), axes=0)[80:240]
kspace_data = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(kspace_data,axes=0), norm='ortho', axis=0), axes=0)
kspace_data = kspace_data/np.max(np.abs(kspace_data))
# crop the zerofilling
kspace_data = kspace_data[34::,0:Nofy]
Nx, Ny, Nc, Ndyn = kspace_data.shape         # (126, 68, 30, 105)

# %% composite kspace
composite_kspace = np.zeros_like(kspace_data[:,:,:,3::])
for dyn in range(Ndyn-3):
    composite_kspace[:,:,:,dyn] = np.sum(kspace_data[:,:,:,dyn:dyn+4],3)

# %% big grappa_sense1 images
_, grappa_sense1 = list(slice_data.items())[0]
n = 1
figure = plt.figure()
for tf in range(time_frame-8,time_frame+8):
    plt.subplot(2,8,n);
    plt.imshow(np.abs(grappa_sense1[80:240,:,tf]),cmap='gray',vmax=0.15)
    plt.axis('off')
    plt.title('TF='+str(tf))
    n = n + 1
plt.suptitle('grappa_sense1')
plt.tight_layout()


# %% composite images
figure = plt.figure()
n = 1
for tf in range(time_frame-8,time_frame+8):
    plt.subplot(2,8,n);
    plt.imshow(rssq(composite_kspace[:,:,:,tf]),cmap='gray',vmax=0.12,vmin=0)
    plt.axis('off')
    plt.title('TF='+str(tf))
    n = n + 1
plt.suptitle('composite_images')
plt.tight_layout()

# %% Composite images with median filter
figure = plt.figure()
n = 1
for tf in range(time_frame-8,time_frame+8):
    plt.subplot(2,8,n);
    plt.imshow(rssq(fft2(median(ifft2(composite_kspace[:,:,:,tf-2:tf+3]))))
               ,cmap='gray',vmax=0.08,vmin=0)
    plt.axis('off')
    plt.title('TF='+str(tf))
    n = n + 1
plt.suptitle('composite_images (median filter over 5 time frames)')
plt.tight_layout()

# %% Composite images with mean filter
figure = plt.figure()
n = 1
for tf in range(time_frame-8,time_frame+8):
    plt.subplot(2,8,n);
    plt.imshow(rssq(fft2(np.mean(ifft2(composite_kspace[:,:,:,tf-3:tf+4]),3)))
               ,cmap='gray',vmax=0.12,vmin=0)
    plt.axis('off')
    plt.title('TF='+str(tf))
    n = n + 1
plt.suptitle('composite_images (mean filter over 7 time frames)')
plt.tight_layout()

# %% sensitivity maps
Smaps = espirit(composite_kspace[None,:,:,:,time_frame], 6 ,40, 0.02, 0.9) 
Smaps = Smaps[0,:,:,:,0]
figure = plt.figure();
plt.imshow(np.sqrt(np.sum(np.abs(Smaps)**2,2)))

# %% ovs mask
ovs_mask = np.zeros((Nx,Ny))
# N1 = 23; N2 = 50; # subject_3_slice_4
N1 = 30; N2 = 58; # subject_8_slice_3
ovs_mask[:,0:N1] = 1
ovs_mask[:,N1-2] = 2/3
ovs_mask[:,N1-1] = 1/3
ovs_mask[:,N2+1::] = 1
ovs_mask[:,N2+1] = 1/3
ovs_mask[:,N2+2] = 2/3
figure = plt.figure(); plt.imshow(ovs_mask,cmap='gray'); plt.title('ovs mask')

# %% acceleration masks
acc_mask4 = np.zeros_like(kspace_data[:,:,0,time_frame], dtype=bool)
acc_mask4[:,np.mod(time_frame,4)::4] = True
acc_mask8 = acc_mask4 * 0
acc_mask8[:,np.mod(time_frame,8)::8] = 1
centerline = (  (np.mod(time_frame,8)==0)*44 
              + (np.mod(time_frame,8)==1)*45 
              + (np.mod(time_frame,8)==2)*46 
              + (np.mod(time_frame,8)==3)*47 
              + (np.mod(time_frame,8)==4)*48 
              + (np.mod(time_frame,8)==5)*49 
              + (np.mod(time_frame,8)==6)*42 
              + (np.mod(time_frame,8)==7)*43)
acc_mask8[:,centerline] = True
figure = plt.figure(); plt.imshow(acc_mask4,cmap='gray'); plt.title('acceleration mask R=4')
figure = plt.figure(); plt.imshow(acc_mask8,cmap='gray'); plt.title('acceleration mask R=8')


# %% cg_images at R=4
cg_image4 = cgsense(kspace_data[:,:,:,time_frame], Smaps, acc_mask4) 
cg_image8 = cgsense(kspace_data[:,:,:,time_frame]*acc_mask8[...,None], Smaps, acc_mask8) 
figure = plt.figure(); plt.imshow(np.abs(cg_image4),cmap='gray',vmax=.12); plt.title('full signal CGSENSE R=4')
figure = plt.figure(); plt.imshow(np.abs(cg_image8),cmap='gray',vmax=.12); plt.title('full signal CGSENSE R=8')

# %% TGRAPPA at R=4
kernel = kernel_cal(composite_kspace[0:92,:,:,time_frame])
kspace_new = grappa(composite_kspace[:,:,:,time_frame] * acc_mask4[:,:,None], kernel, time_frame)
reference = np.sum(ifft2(kspace_new)*np.conj(Smaps),2)
figure = plt.figure(); plt.imshow(np.abs(reference),cmap='gray',vmax=.12); plt.title('reference')

# %% background signal
y_background_no_filter = fft2(ifft2(composite_kspace[...,time_frame]) * ovs_mask[...,None])
y_background_median3 = fft2(median(ifft2(composite_kspace[...,time_frame-1:time_frame+2])) * ovs_mask[...,None])
y_background_median5 = fft2(median(ifft2(composite_kspace[...,time_frame-2:time_frame+3])) * ovs_mask[...,None])
y_background_median7 = fft2(median(ifft2(composite_kspace[...,time_frame-3:time_frame+4])) * ovs_mask[...,None])
y_background_mean3 = fft2(np.mean(ifft2(composite_kspace[...,time_frame-1:time_frame+2]),3) * ovs_mask[...,None])
y_background_mean5 = fft2(np.mean(ifft2(composite_kspace[...,time_frame-2:time_frame+3]),3) * ovs_mask[...,None])
y_background_mean7 = fft2(np.mean(ifft2(composite_kspace[...,time_frame-3:time_frame+4]),3) * ovs_mask[...,None])

# %% outer volume subtraction
y_diff_no_filter = kspace_data[...,time_frame]-y_background_no_filter
y_diff_median3 = kspace_data[...,time_frame]-y_background_median3
y_diff_median5 = kspace_data[...,time_frame]-y_background_median5
y_diff_median7 = kspace_data[...,time_frame]-y_background_median7
y_diff_mean3 = kspace_data[...,time_frame]-y_background_mean3
y_diff_mean5 = kspace_data[...,time_frame]-y_background_mean5
y_diff_mean7 = kspace_data[...,time_frame]-y_background_mean7

# %% diff images R=4
cgsense_kfull_r4 = cgsense(kspace_data[:,:,:,time_frame], Smaps, acc_mask4) 
cgsense_kdiff_no_filter_r4 = cgsense(y_diff_no_filter*acc_mask4[...,None], Smaps, acc_mask4) 
cgsense_kdiff_median3_r4 = cgsense(y_diff_median3*acc_mask4[...,None], Smaps, acc_mask4) 
cgsense_kdiff_median5_r4 = cgsense(y_diff_median5*acc_mask4[...,None], Smaps, acc_mask4) 
cgsense_kdiff_median7_r4 = cgsense(y_diff_median7*acc_mask4[...,None], Smaps, acc_mask4) 
cgsense_kdiff_mean3_r4 = cgsense(y_diff_mean3*acc_mask4[...,None], Smaps, acc_mask4) 
cgsense_kdiff_mean5_r4 = cgsense(y_diff_mean5*acc_mask4[...,None], Smaps, acc_mask4) 
cgsense_kdiff_mean7_r4 = cgsense(y_diff_mean7*acc_mask4[...,None], Smaps, acc_mask4) 

# %% diff images R=8
cgsense_kfull_r8 = cgsense(kspace_data[:,:,:,time_frame]*acc_mask8[...,None], Smaps, acc_mask8) 
cgsense_kdiff_no_filter_r8 = cgsense(y_diff_no_filter*acc_mask8[...,None], Smaps, acc_mask8) 
cgsense_kdiff_median3_r8 = cgsense(y_diff_median3*acc_mask8[...,None], Smaps, acc_mask8) 
cgsense_kdiff_median5_r8 = cgsense(y_diff_median5*acc_mask8[...,None], Smaps, acc_mask8) 
cgsense_kdiff_median7_r8 = cgsense(y_diff_median7*acc_mask8[...,None], Smaps, acc_mask8) 
cgsense_kdiff_mean3_r8 = cgsense(y_diff_mean3*acc_mask8[...,None], Smaps, acc_mask8) 
cgsense_kdiff_mean5_r8 = cgsense(y_diff_mean5*acc_mask8[...,None], Smaps, acc_mask8) 
cgsense_kdiff_mean7_r8 = cgsense(y_diff_mean7*acc_mask8[...,None], Smaps, acc_mask8) 

# %%
props = dict(boxstyle='round', facecolor='black', alpha=0.8)
data_range = np.max(np.abs(reference)*(1-ovs_mask))
vmax=.08
vmin=0

figure = plt.figure();
plt.subplot(2,4,1)
plt.imshow(np.abs(reference),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('reference')
plt.axis('off')
'''
figure = plt.figure(); plt.imshow(np.abs(cgsense_kfull_r4),cmap='gray',vmax=.08); plt.title('cgsense_kfull_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kfull_r4[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kfull_r4[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])
'''
plt.subplot(2,4,5)
plt.imshow(np.abs(np.abs(cgsense_kdiff_no_filter_r4)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_no_filter_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_no_filter_r4[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_no_filter_r4[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,2)
plt.imshow(np.abs(np.abs(cgsense_kdiff_median3_r4)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median3_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median3_r4[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_median3_r4[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,3)
plt.imshow(np.abs(np.abs(cgsense_kdiff_median5_r4)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median5_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median5_r4[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_median5_r4[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,4)
plt.imshow(np.abs(np.abs(cgsense_kdiff_median7_r4)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median7_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median7_r4[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_median7_r4[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,6)
plt.imshow(np.abs(np.abs(cgsense_kdiff_mean3_r4)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean3_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean3_r4[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_mean3_r4[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,7)
plt.imshow(np.abs(np.abs(cgsense_kdiff_mean5_r4)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean5_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean5_r4[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_mean5_r4[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,8)
plt.imshow(np.abs(np.abs(cgsense_kdiff_mean7_r4)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean7_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean7_r4[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_mean7_r4[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])
plt.tight_layout()

# %%
props = dict(boxstyle='round', facecolor='black', alpha=0.8)
data_range = np.max(np.abs(reference)*(1-ovs_mask))

figure = plt.figure();
plt.subplot(2,4,1)
plt.imshow(np.abs(reference),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('reference')
plt.axis('off')
'''
figure = plt.figure(); plt.imshow(np.abs(cgsense_kfull_r8),cmap='gray',vmax=.08); plt.title('cgsense_kfull_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kfull_r8[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kfull_r8[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])
'''
plt.subplot(2,4,5)
plt.imshow(np.abs(np.abs(cgsense_kdiff_no_filter_r8)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_no_filter_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_no_filter_r8[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_no_filter_r8[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,2)
plt.imshow(np.abs(np.abs(cgsense_kdiff_median3_r8)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median3_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median3_r8[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_median3_r8[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,3)
plt.imshow(np.abs(np.abs(cgsense_kdiff_median5_r8)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median5_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median5_r8[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_median5_r8[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,4)
plt.imshow(np.abs(np.abs(cgsense_kdiff_median7_r8)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median7_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median7_r8[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_median7_r8[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,6)
plt.imshow(np.abs(np.abs(cgsense_kdiff_mean3_r8)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean3_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean3_r8[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_mean3_r8[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,7)
plt.imshow(np.abs(np.abs(cgsense_kdiff_mean5_r8)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean5_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean5_r8[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_mean5_r8[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,8)
plt.imshow(np.abs(np.abs(cgsense_kdiff_mean7_r8)-0*np.abs(reference*(1-ovs_mask))),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean7_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean7_r8[:,N1:N2], reference[:,N1:N2])
SSIM = ssim(np.abs(reference[:,N1:N2]), np.abs(cgsense_kdiff_mean7_r8[:,N1:N2]), data_range=data_range)
plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])
plt.tight_layout()


# %% R=4 cropped
props = dict(boxstyle='round', facecolor='black', alpha=0.8)
Ny1 = 45; Ny2 = 84;
data_range = np.max(np.abs(reference*(1-ovs_mask))[Ny1:Ny2])

figure = plt.figure();
plt.subplot(2,4,1)
plt.imshow(np.abs(reference[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('reference')
plt.axis('off')

plt.subplot(2,4,5)
plt.imshow(np.abs(cgsense_kdiff_no_filter_r4[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_no_filter_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_no_filter_r4[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_no_filter_r4[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,2)
plt.imshow(np.abs(cgsense_kdiff_median3_r4[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median3_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median3_r4[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_median3_r4[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,3)
plt.imshow(np.abs(cgsense_kdiff_median5_r4)[Ny1:Ny2],cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median5_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median5_r4[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_median5_r4[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,4)
plt.imshow(np.abs(cgsense_kdiff_median7_r4[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median7_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median7_r4[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_median7_r4[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,6)
plt.imshow(np.abs(cgsense_kdiff_mean3_r4[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean3_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean3_r4[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_mean3_r4[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,7)
plt.imshow(np.abs(cgsense_kdiff_mean5_r4[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean5_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean5_r4[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_mean5_r4[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,8)
plt.imshow(np.abs(cgsense_kdiff_mean7_r4[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean7_r4')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean7_r4[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_mean7_r4[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])
plt.tight_layout()

# %% R=8 cropped
props = dict(boxstyle='round', facecolor='black', alpha=0.8)
Ny1 = 45; Ny2 = 84;
data_range = np.max(np.abs(reference*(1-ovs_mask))[Ny1:Ny2])

figure = plt.figure();
plt.subplot(2,4,1)
plt.imshow(np.abs(reference[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('reference')
plt.axis('off')

plt.subplot(2,4,5)
plt.imshow(np.abs(cgsense_kdiff_no_filter_r8[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_no_filter_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_no_filter_r8[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_no_filter_r8[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,2)
plt.imshow(np.abs(cgsense_kdiff_median3_r8[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median3_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median3_r8[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_median3_r8[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,3)
plt.imshow(np.abs(cgsense_kdiff_median5_r8)[Ny1:Ny2],cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median5_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median5_r8[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_median5_r8[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,4)
plt.imshow(np.abs(cgsense_kdiff_median7_r8[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_median7_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_median7_r8[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_median7_r8[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,6)
plt.imshow(np.abs(cgsense_kdiff_mean3_r8[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean3_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean3_r8[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_mean3_r8[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,7)
plt.imshow(np.abs(cgsense_kdiff_mean5_r8[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean5_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean5_r8[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_mean5_r8[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])

plt.subplot(2,4,8)
plt.imshow(np.abs(cgsense_kdiff_mean7_r8[Ny1:Ny2]),cmap='gray',vmax=vmax,vmin=vmin); 
plt.title('cgsense_kdiff_mean7_r8')
ax = plt.gca()
NMSE = nmse(cgsense_kdiff_mean7_r8[Ny1:Ny2,N1:N2], reference[Ny1:Ny2,N1:N2])
SSIM = ssim(np.abs(reference[Ny1:Ny2,N1:N2]), np.abs(cgsense_kdiff_mean7_r8[Ny1:Ny2,N1:N2]), data_range=data_range)
plt.text(0.5, -0.2, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
plt.xticks([])
plt.yticks([])
plt.tight_layout()