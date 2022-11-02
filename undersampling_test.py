import numpy as np
import h5py
import SupportingFunctions as sf
from espirit import espirit
import SupportingFunctions as sf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# %%
data_path = 'file_brain_AXFLAIR_200_6002425.h5'
f = h5py.File(data_path, "r")
data = f['kspace']
Ns,Nc,Nx,Ny = data.shape

# %%
kspace = np.transpose(data[0], (1,2,0))
Smaps = espirit(kspace[None,...], 6,24, 0.02, 0.95)
Smaps = Smaps[0,:,:,:,0]

R = 8
accmask = np.zeros_like(kspace[:,:,0])
accmask[:,::R] = 1
imref = np.abs(sf.cgsense(kspace, Smaps, np.ones_like(kspace[:,:,0])))
im = np.zeros((Nx,Ny,R), dtype=complex)

for i in np.arange(R):
    im[:,:,i] = np.abs(sf.cgsense(kspace*np.roll(accmask,i,1)[...,None], Smaps, np.roll(accmask,i,1)))

figure = plt.figure(figsize=(14,3));
plt.subplot(1,R+1,1);
plt.imshow(np.abs(imref),cmap="gray"); 
plt.axis('off');
plt.title('reference')
for i in np.arange(R):
    plt.subplot(1,R+1,i+2);
    plt.imshow(np.abs(im[:,:,i]),cmap="gray"); 
    plt.axis('off');
    plt.title(f'{i}'+', '+f'{i+R}'+', '+f'{i+2*R}'+', ...')
    NMSE = sf.nmse(im[:,:,i],imref)
    SSIM = ssim(imref, im[:,:,i], data_range=np.max(imref)-np.min(imref))
    ax = plt.gca()
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\n'+'SSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    
# %%

R = 8
kspace = np.transpose(data[0:2*R-1],(2,3,1,0))
accmask = np.zeros_like(kspace[:,:,0,0])
accmask[:,::R] = 1

y = np.zeros_like(kspace)
for i in np.arange(2*R-1):
    y[:,:,:,i] = kspace[:,:,:,i] * np.roll(accmask,i,1)[...,None]

y_com = np.zeros((kspace.shape[0],kspace.shape[1],kspace.shape[2],R), dtype=complex)
for i in np.arange(R):
    y_com[:,:,:,i] = np.sum(y[:,:,:,i:i+R],3)
    
figure = plt.figure(figsize=(14,8));
for i in np.arange(R):
    print(i)
    Smapsref = espirit(kspace[None,:,:,:,i], 6,24, 0.02, 0.95)
    Smapsref = Smapsref[0,:,:,:,0]
    print('Smapsref')
    imref = sf.cgsense(kspace[:,:,:,i], Smapsref, np.ones_like(kspace[:,:,0,0]))
    Smaps = espirit(y_com[None,:,:,:,i], 6,24, 0.02, 0.95)
    Smaps = Smaps[0,:,:,:,0]
    print('Smaps')
    im = sf.cgsense(y[:,:,:,i], Smaps, np.roll(accmask,i,1))
    plt.subplot(2,R,i+1);
    plt.imshow(np.abs(im),cmap="gray"); 
    plt.axis('off');
    nmse = sf.nmse(im,imref)
    plt.title(f'{nmse:,.3f}')
    plt.subplot(2,R,R+i+1);
    plt.imshow(np.abs(imref),cmap="gray"); 
    plt.axis('off');

# %%
R = 8
kspace = np.transpose(data[0:2*R-1],(2,3,1,0))
accmask = np.zeros_like(kspace[:,:,0,0])
accmask[:,::R] = 1

Smapsref = espirit(kspace[None,:,:,:,0], 6,24, 0.02, 0.95)
Smapsref = Smapsref[0,:,:,:,0]
imref = sf.cgsense(kspace[:,:,:,0], Smapsref, np.ones_like(kspace[:,:,0,0]))
print('imref is done')

im = np.zeros((Nx,Ny,R), dtype=complex)
Smaps = np.zeros((Nx,Ny,Nc,R), dtype=complex)
y_com = np.zeros((Nx,Ny,Nc,R), dtype=complex)
for i in np.arange(R):
    print(i)
    for j in np.arange(R):
        y_com[:,:,:,i] = y_com[:,:,:,i] + kspace[:,:,:,j]*np.roll(accmask,i+j,1)[...,None]
    Smap = espirit(y_com[None,:,:,:,i], 6,24, 0.02, 0.95)
    Smap = Smap[0,:,:,:,0]
    Smaps[:,:,:,i] = Smap
    im[:,:,i] = sf.cgsense(kspace[:,:,:,0]*np.roll(accmask,i,1)[...,None], Smap, np.roll(accmask,i,1))


figure = plt.figure(figsize=(14,3));
plt.subplot(1,R+1,1);
plt.imshow(np.abs(imref),cmap="gray"); 
plt.axis('off');
plt.title('reference')
im = np.abs(im)
imref = np.abs(imref)
for i in np.arange(R):
    plt.subplot(1,R+1,i+2);
    plt.imshow(np.abs(im[:,:,i]),cmap="gray"); 
    plt.axis('off');
    plt.title(f'{i}'+', '+f'{i+R}'+', '+f'{i+2*R}'+', ...')
    NMSE = sf.nmse(im[:,:,i],imref)
    SSIM = ssim(imref, im[:,:,i], data_range=np.max(imref)-np.min(imref))
    ax = plt.gca()
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\n'+'SSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    
    
