import numpy as np
import SupportingFunctions as sf
import mat73
import matplotlib.pyplot as plt
from espirit import espirit


# %% Data Loading
# Load full data (12 slices) --> [320, 120, 30, 153, 12]
# realtime_data = mat73.loadmat('realtime_data.mat')
# Load small data (1 slice) --> [320, 120, 30, 153]
realtime_data = mat73.loadmat('realtime_data_small.mat')
_, datas = list(realtime_data.items())[0]
datas = datas[68:320,0:72,:,:]
datas = datas.astype('complex64')

Nx = datas.shape[0]             # num of y pixels (vertical)
Ny = datas.shape[1]             # num of x pixels (horizontal)
Nc = datas.shape[2]             # num of coils
Ndyn = datas.shape[3]           # num of time frames

# %% Zerofilled image
acc_mask = np.zeros((Nx,Ny), dtype=bool)
acc_mask[:,::8] = True
# first zerofilled image:x0
x0 = sf.rssq(sf.kspace_to_im(datas[:,:,:,0]*acc_mask[...,None]))
figure = plt.figure(); plt.imshow(np.abs(x0), cmap="gray") 

# %% composite data
y_com = np.zeros([Nx,Ny,Nc,Ndyn-3], dtype=np.complex64)
for ind in range(Ndyn-3):
    y_com[:,:,:,ind] = np.sum(datas[:,:,:,ind:ind+4],3)
x_com = sf.rssq(sf.kspace_to_im(y_com))
im_composite = x_com[:,:,0]
figure = plt.figure(); plt.imshow(np.abs(im_composite), cmap="gray"); plt.axis('off')


# %% mask selection
ovs_mask = np.ones((Nx,Ny), dtype=bool)
ovs_mask[:,25:55] = False
# subtact these out from the data
y_com1 = y_com[:,:,:,0]
y_background = sf.im_to_kspace(sf.kspace_to_im(y_com1)*ovs_mask[...,None])
# subtract out background
y1 = datas[:,:,:,0]*acc_mask[...,None]
y_diff = y1 - y_background*acc_mask[...,None]

figure = plt.figure(); plt.imshow(sf.rssq(sf.kspace_to_im(y_diff)), cmap="gray"); plt.axis('off')
figure = plt.figure(); plt.imshow(sf.rssq(sf.kspace_to_im(y1)), cmap="gray"); plt.axis('off') 


# %% Generate coil maps
y_low = np.zeros([Nx,Ny,Nc], dtype=np.complex64)
ACS_size = 32
LPF = np.hanning(48)[:,None] * np.hanning(ACS_size)[None,:]
y_low[160-69-24:160-69+24,48-ACS_size//2:48+ACS_size//2,:] = y_com1[160-69-24:160-69+24,48-ACS_size//2:48+ACS_size//2,:] * LPF[...,None]
x_low = sf.kspace_to_im(y_low)
Smaps = x_low / sf.rssq(x_low + 1e-8)[...,None]
Smaps_mask = Smaps * (1 - ovs_mask[...,None])

y_low_diff = np.zeros([Nx,Ny,Nc], dtype=np.complex64)
y_low_diff[160-69-24:160-69+24,48-ACS_size//2:48+ACS_size//2,:] = (y_com1-y_background)[160-69-24:160-69+24,48-ACS_size//2:48+ACS_size//2,:] * LPF[...,None]
x_low_diff = sf.kspace_to_im(y_low_diff)
Smaps_diff = x_low_diff / sf.rssq(x_low_diff + 1e-8)[...,None]

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps[:,:,4],Smaps[:,:,7],Smaps[:,:,8],Smaps[:,:,10],Smaps[:,:,13],Smaps[:,:,20],Smaps[:,:,25],Smaps[:,:,27]),axis=1)), cmap="gray"); plt.axis('off')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps_mask[:,:,4],Smaps_mask[:,:,7],Smaps_mask[:,:,8],Smaps_mask[:,:,10],Smaps_mask[:,:,13],Smaps_mask[:,:,20],Smaps_mask[:,:,25],Smaps_mask[:,:,27]),axis=1)), cmap="gray"); plt.axis('off') 

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps_diff[:,:,4],Smaps_diff[:,:,7],Smaps_diff[:,:,8],Smaps_diff[:,:,10],Smaps_diff[:,:,13],Smaps_diff[:,:,20],Smaps_diff[:,:,25],Smaps_diff[:,:,27]),axis=1)), cmap="gray") 


# %% results
# no OVS processing
cg_sense = sf.cgsense(y1, Smaps, acc_mask)
# OVS from k-space
cg_sense_OVS = sf.cgsense(y_diff, Smaps, acc_mask)
# OVS from k-space and calibration in image space
cg_sense_mask = sf.cgsense(y_diff, Smaps_mask, acc_mask)
# OVS from k-space and calibration in k-space 
cg_sense_diff = sf.cgsense(y_diff, Smaps_diff, acc_mask)

background = im_composite * ovs_mask
figure = plt.figure(); plt.imshow(np.abs(np.concatenate((cg_sense,cg_sense_OVS+background,cg_sense_mask+background,cg_sense_diff+background), axis=1)), cmap="gray", vmax=0.003); plt.axis('off')














