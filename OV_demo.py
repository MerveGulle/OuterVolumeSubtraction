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
figure = plt.figure(); plt.imshow(np.abs(x0), cmap="gray"); 
plt.title('zerofilled image'); plt.axis('off')


# %% composite data
y_com = np.zeros([Nx,Ny,Nc,Ndyn-3], dtype=np.complex64)
for ind in range(Ndyn-3):
    y_com[:,:,:,ind] = np.sum(datas[:,:,:,ind:ind+4],3)
x_com = sf.rssq(sf.kspace_to_im(y_com))
im_composite = x_com[:,:,0]
figure = plt.figure(); plt.imshow(np.abs(im_composite), cmap="gray"); plt.axis('off')
plt.title('composite image'); plt.axis('off')


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
plt.title('OVS zerofilled image'); plt.axis('off')
figure = plt.figure(); plt.imshow(sf.rssq(sf.kspace_to_im(y1)), cmap="gray"); plt.axis('off') 
plt.title('zerofilled image'); plt.axis('off')

# %% Generate coil maps with low resolution images
y_low = np.zeros([Nx,Ny,Nc], dtype=np.complex64)
ACS_size = 32
LPF = np.hanning(48)[:,None] * np.hanning(ACS_size)[None,:]
y_low[160-69-24:160-69+24,48-ACS_size//2:48+ACS_size//2,:] = y_com1[160-69-24:160-69+24,48-ACS_size//2:48+ACS_size//2,:] * LPF[...,None]
x_low = sf.kspace_to_im(y_low)
Smaps1 = x_low / sf.rssq(x_low + 1e-8)[...,None]
Smaps1_mask = Smaps1 * (1 - ovs_mask[...,None])

y_low_diff = np.zeros([Nx,Ny,Nc], dtype=np.complex64)
y_low_diff[160-69-24:160-69+24,48-ACS_size//2:48+ACS_size//2,:] = (y_com1-y_background)[160-69-24:160-69+24,48-ACS_size//2:48+ACS_size//2,:] * LPF[...,None]
x_low_diff = sf.kspace_to_im(y_low_diff)
Smaps1_diff = x_low_diff / sf.rssq(x_low_diff + 1e-8)[...,None]

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps1[:,:,4],Smaps1[:,:,7],Smaps1[:,:,8],Smaps1[:,:,10],Smaps1[:,:,13],Smaps1[:,:,20],Smaps1[:,:,25],Smaps1[:,:,27]),axis=1)), cmap="gray"); plt.axis('off')
plt.title('Sensitivity Maps - Low Res Img')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps1_mask[:,:,4],Smaps1_mask[:,:,7],Smaps1_mask[:,:,8],Smaps1_mask[:,:,10],Smaps1_mask[:,:,13],Smaps1_mask[:,:,20],Smaps1_mask[:,:,25],Smaps1_mask[:,:,27]),axis=1)), cmap="gray"); plt.axis('off') 
plt.title('Masked Sensitivity Maps - Low Res Img')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps1_diff[:,:,4],Smaps1_diff[:,:,7],Smaps1_diff[:,:,8],Smaps1_diff[:,:,10],Smaps1_diff[:,:,13],Smaps1_diff[:,:,20],Smaps1_diff[:,:,25],Smaps1_diff[:,:,27]),axis=1)), cmap="gray") 
plt.title('OVS Sensitivity Maps - Low Res Img')


# %% Generate coil maps with espirit
Smaps2 = espirit(y_com1[None,...], 6, 24, 0.02, 0.95)
Smaps2 = Smaps2[0,:,:,:,0]
Smaps2 = x_low / sf.rssq(x_low + 1e-8)[...,None]
Smaps2_mask = Smaps2 * (1 - ovs_mask[...,None])

Smaps2_diff = espirit((y_com1-y_background)[None,...], 6, 24, 0.02, 0.95)
Smaps2_diff = Smaps2_diff[0,:,:,:,0]

figure = plt.figure();
plt.imshow(np.abs(np.concatenate((Smaps2[:,:,4],Smaps2[:,:,7],Smaps2[:,:,8],Smaps2[:,:,10],Smaps2[:,:,13],Smaps2[:,:,20],Smaps2[:,:,25],Smaps2[:,:,27]),axis=1)), cmap="gray"); plt.axis('off')
plt.title('Sensitivity Maps - Espirit')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps2_mask[:,:,4],Smaps2_mask[:,:,7],Smaps2_mask[:,:,8],Smaps2_mask[:,:,10],Smaps2_mask[:,:,13],Smaps2_mask[:,:,20],Smaps2_mask[:,:,25],Smaps2_mask[:,:,27]),axis=1)), cmap="gray"); plt.axis('off') 
plt.title('Masked Sensitivity Maps - Espirit')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps2_diff[:,:,4],Smaps2_diff[:,:,7],Smaps2_diff[:,:,8],Smaps2_diff[:,:,10],Smaps2_diff[:,:,13],Smaps2_diff[:,:,20],Smaps2_diff[:,:,25],Smaps2_diff[:,:,27]),axis=1)), cmap="gray") 
plt.title('OVS Sensitivity Maps - Espirit')


# %% results with low resolution img Smaps
# no OVS processing
cg_sense = sf.cgsense(y1, Smaps1, acc_mask)
# OVS from k-space
cg_sense_OVS = sf.cgsense(y_diff, Smaps1, acc_mask)
# OVS from k-space and calibration in image space
cg_sense_mask = sf.cgsense(y_diff, Smaps1_mask, acc_mask)
# OVS from k-space and calibration in k-space 
cg_sense_diff = sf.cgsense(y_diff, Smaps1_diff, acc_mask)

background = im_composite * ovs_mask
figure = plt.figure(); plt.imshow(np.abs(np.concatenate((cg_sense,cg_sense_OVS+background,cg_sense_mask+background,cg_sense_diff+background), axis=1)), cmap="gray", vmax=0.003); plt.axis('off')
plt.title('Results for low res img Smaps'); plt.axis('off')

# %% results with espirit Smaps
# no OVS processing
cg_sense = sf.cgsense(y1, Smaps2, acc_mask)
# OVS from k-space
cg_sense_OVS = sf.cgsense(y_diff, Smaps2, acc_mask)
# OVS from k-space and calibration in image space
cg_sense_mask = sf.cgsense(y_diff, Smaps2_mask, acc_mask)
# OVS from k-space and calibration in k-space 
cg_sense_diff = sf.cgsense(y_diff, Smaps2_diff, acc_mask)

background = im_composite * ovs_mask
figure = plt.figure(); plt.imshow(np.abs(np.concatenate((cg_sense,cg_sense_OVS+background,cg_sense_mask+background,cg_sense_diff+background), axis=1)), cmap="gray", vmax=0.003); plt.axis('off')
plt.title('Results for espirit Smaps'); plt.axis('off')











