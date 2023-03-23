import numpy as np
import SupportingFunctions as sf
import mat73
import matplotlib.pyplot as plt
from espirit import espirit
from scipy.io import loadmat


# %% Data Loading
# Load full data (12 slices) --> [320, 120, 30, 153, 12]
# realtime_data = mat73.loadmat('realtime_data.mat')
# Load small data (1 slice) --> [320, 120, 30, 153]
# realtime_data = mat73.loadmat('realtime_data_small.mat')
realtime_data = mat73.loadmat('realtime_data_small.mat')
_, datas = list(realtime_data.items())[0]
datas = datas[68:320,0:72,:,:]*1e6
datas = datas.astype('complex64')

Nx = datas.shape[0]             # num of y pixels (vertical)
Ny = datas.shape[1]             # num of x pixels (horizontal)
Nc = datas.shape[2]             # num of coils
Ndyn = datas.shape[3]           # num of time frames


# %% composite data
y_com = np.zeros([Nx,Ny,Nc,Ndyn-3], dtype=np.complex64)
for ind in range(Ndyn-3):
    y_com[:,:,:,ind] = np.sum(datas[:,:,:,ind:ind+4],3)


# %% image recontruction for the first 20 time frames
num_img = 20
imagesN = np.zeros_like(y_com[:,:,0,0:num_img])
mask = np.zeros((y_com.shape[0], y_com.shape[1]))
mask[:,::4] = 1
for tf in np.arange(num_img):
    Smaps = espirit(y_com[None,:,:,:,tf], 6, 24, 0.02, 0.95)
    Smaps = Smaps[0,:,:,:,0]
    imagesN[:,:,tf] = sf.cgsense(datas[:,:,:,tf], Smaps, np.roll(mask,tf,axis=1))

for tf in np.arange(num_img):
    kernel = sf.kernel_cal(y_com[80:104,34:58,:,tf])
    imagesN[:,:,tf] = sf.grappa(datas[:,:,:,tf], kernel, tf)
    
figure = plt.figure(figsize=(12,12))
for tf in np.arange(num_img):
    plt.subplot(2,num_img//2,tf+1)
    plt.imshow(np.abs(imagesN[:,:,tf]),cmap="gray",vmax=2000)
    plt.title('TF #'+f'{tf+1}',fontsize=12)
    plt.axis("off")
   
figure = plt.figure(figsize=(12,4))
for tf in np.arange(num_img):
    plt.subplot(2,num_img//2,tf+1)
    plt.imshow(np.abs(imagesN[93:160,11:65,tf]),cmap="gray",vmax=2000)
    plt.title('TF #'+f'{tf+1}',fontsize=12)
    plt.axis("off")

figure = plt.figure(figsize=(12,12))
for tf in np.arange(num_img):
    plt.subplot(2,10,tf+1)
    plt.imshow(sf.rssq(sf.kspace_to_im(datas[:,:,:,tf])),cmap="gray",vmax=1000)
    plt.axis("off")

# %% time frame = 8 (Ndyn=7) : Cardiac contraction
TF = 5
shift = np.mod(TF, 8)


# %% Zerofilled image
acc_mask = np.zeros((Nx,Ny), dtype=bool)
acc_mask[:,shift::8] = True
# zerofilled image:x0
x0 = sf.rssq(sf.kspace_to_im(datas[:,:,:,TF]*acc_mask[...,None]))
"""
figure = plt.figure(); plt.imshow(np.abs(x0), cmap="gray"); 
plt.title('zerofilled image'); plt.axis('off')
"""


# %% composite images
x_com = sf.rssq(sf.kspace_to_im(y_com))
im_composite = x_com[:,:,TF]
"""
figure = plt.figure(); plt.imshow(np.abs(im_composite), cmap="gray", vmax=2000); plt.axis('off')
plt.title('composite image'); plt.axis('off')
"""


# %% mask selection
"""
# rectangular mask
# ovs_mask = np.ones((Nx,Ny), dtype=bool)
# ovs_mask[:,25:55] = False
"""
# only crop the heart
ovs_mask = loadmat('only_heart_252x72.mat')['target_mask']==1
# crop the outer volume
# ovs_mask = loadmat('outer_volume_252x72.mat')['target_mask']==1
# subtact these out from the data
y_com1 = y_com[:,:,:,TF]
y_background = sf.im_to_kspace(sf.kspace_to_im(y_com1)*ovs_mask[...,None])
# subtract out background
y1 = datas[:,:,:,TF]*acc_mask[...,None]
y1_diff = y1 - y_background*acc_mask[...,None]
"""
figure = plt.figure(); plt.imshow(sf.rssq(sf.kspace_to_im(y1_diff)), cmap="gray"); plt.axis('off')
plt.title('OVS zerofilled image'); plt.axis('off')
figure = plt.figure(); plt.imshow(sf.rssq(sf.kspace_to_im(y1)), cmap="gray"); plt.axis('off') 
plt.title('zerofilled image'); plt.axis('off')
"""
"""
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
Smaps1_diff = x_low_diff / sf.rssq(x_low_diff + 1e-20)[...,None]

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps1[:,:,4],Smaps1[:,:,7],Smaps1[:,:,8],Smaps1[:,:,10],Smaps1[:,:,13],Smaps1[:,:,20],Smaps1[:,:,25],Smaps1[:,:,27]),axis=1)), cmap="gray"); plt.axis('off')
plt.title('Sensitivity Maps - Low Res Img')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps1_mask[:,:,4],Smaps1_mask[:,:,7],Smaps1_mask[:,:,8],Smaps1_mask[:,:,10],Smaps1_mask[:,:,13],Smaps1_mask[:,:,20],Smaps1_mask[:,:,25],Smaps1_mask[:,:,27]),axis=1)), cmap="gray"); plt.axis('off') 
plt.title('Masked Sensitivity Maps - Low Res Img')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps1_diff[:,:,4],Smaps1_diff[:,:,7],Smaps1_diff[:,:,8],Smaps1_diff[:,:,10],Smaps1_diff[:,:,13],Smaps1_diff[:,:,20],Smaps1_diff[:,:,25],Smaps1_diff[:,:,27]),axis=1)), cmap="gray") 
plt.title('OVS Sensitivity Maps - Low Res Img')
"""


# %% Generate coil maps with espirit
Smaps2 = espirit(y_com1[None,...], 6, 44, 0.02, 0.95)
Smaps2 = Smaps2[0,:,:,:,0]
Smaps2_mask = Smaps2 * (1 - ovs_mask[...,None])
# ovs mask
# Smaps2_diff = espirit((y_com1-y_background)[None,...], 6, 44, 0.02, 0.95)
# heart mask
Smaps2_diff = espirit((y_com1-y_background)[None,...], 14, 44, 0.02, 0.95)
Smaps2_diff = Smaps2_diff[0,:,:,:,0]

"""
figure = plt.figure();
plt.imshow(np.abs(np.concatenate((Smaps2[:,:,4],Smaps2[:,:,7],Smaps2[:,:,8],Smaps2[:,:,10],Smaps2[:,:,13],Smaps2[:,:,20],Smaps2[:,:,25],Smaps2[:,:,27]),axis=1)), cmap="gray", vmax = 0.5); 
plt.axis('off')
plt.title('Full Sensitivity Maps - Amplitude')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps2_mask[:,:,4],Smaps2_mask[:,:,7],Smaps2_mask[:,:,8],Smaps2_mask[:,:,10],Smaps2_mask[:,:,13],Smaps2_mask[:,:,20],Smaps2_mask[:,:,25],Smaps2_mask[:,:,27]),axis=1)), cmap="gray", vmax = 0.5); 
plt.axis('off') 
plt.title('Masked Sensitivity Maps - Amplitude')

figure = plt.figure(); 
plt.imshow(np.abs(np.concatenate((Smaps2_diff[:,:,4],Smaps2_diff[:,:,7],Smaps2_diff[:,:,8],Smaps2_diff[:,:,10],Smaps2_diff[:,:,13],Smaps2_diff[:,:,20],Smaps2_diff[:,:,25],Smaps2_diff[:,:,27]),axis=1)), cmap="gray", vmax = 0.5); 
plt.axis('off') 
plt.title('OVS Sensitivity Maps - Amplitude')


figure = plt.figure();
plt.imshow(np.angle(np.concatenate((Smaps2[:,:,4],Smaps2[:,:,7],Smaps2[:,:,8],Smaps2[:,:,10],Smaps2[:,:,13],Smaps2[:,:,20],Smaps2[:,:,25],Smaps2[:,:,27]),axis=1)), cmap="gray"); 
plt.axis('off')
plt.title('Full Sensitivity Maps - Phase')

figure = plt.figure(); 
plt.imshow(np.angle(np.concatenate((Smaps2_mask[:,:,4],Smaps2_mask[:,:,7],Smaps2_mask[:,:,8],Smaps2_mask[:,:,10],Smaps2_mask[:,:,13],Smaps2_mask[:,:,20],Smaps2_mask[:,:,25],Smaps2_mask[:,:,27]),axis=1)), cmap="gray"); 
plt.axis('off') 
plt.title('Masked Sensitivity Maps - Phase')

figure = plt.figure(); 
plt.imshow(np.angle(np.concatenate((Smaps2_diff[:,:,4],Smaps2_diff[:,:,7],Smaps2_diff[:,:,8],Smaps2_diff[:,:,10],Smaps2_diff[:,:,13],Smaps2_diff[:,:,20],Smaps2_diff[:,:,25],Smaps2_diff[:,:,27]),axis=1)), cmap="gray"); 
plt.axis('off')  
plt.title('OVS Sensitivity Maps - Phase')
"""

"""
# %% results with low resolution img Smaps
# no OVS processing
cg_sense = sf.cgsense(y1, Smaps1, acc_mask)
# OVS from k-space
cg_sense_OVS = sf.cgsense(y1_diff, Smaps1, acc_mask)
# OVS from k-space and calibration in image space
cg_sense_mask = sf.cgsense(y1_diff, Smaps1_mask, acc_mask)
# OVS from k-space and calibration in k-space 
cg_sense_diff = sf.cgsense(y1_diff, Smaps1_diff, acc_mask)

background = im_composite * ovs_mask
figure = plt.figure(); plt.imshow(np.abs(np.concatenate((cg_sense,cg_sense_OVS+background,cg_sense_mask+background,cg_sense_diff+background), axis=1)), cmap="gray", vmax=0.003); plt.axis('off')
plt.title('Results for low res img Smaps'); plt.axis('off')
"""


# %% results with cgsense
# no OVS processing
cg_sense = sf.cgsense(y1, Smaps2, acc_mask)
# OVS from k-space
cg_sense_OVS = sf.cgsense(y1_diff, Smaps2, acc_mask)
# OVS from k-space and calibration in image space
cg_sense_mask = sf.cgsense(y1_diff, Smaps2_mask, acc_mask)
# OVS from k-space and calibration in k-space 
cg_sense_diff = sf.cgsense(y1_diff, Smaps2_diff, acc_mask)

background = im_composite * ovs_mask * 0
figure = plt.figure(); plt.imshow(np.abs(np.concatenate((cg_sense,cg_sense_OVS+background,cg_sense_mask+background,cg_sense_diff+background), axis=1)), cmap="gray", vmax=1800); plt.axis('off')
plt.title('Results for espirit Smaps'); plt.axis('off')

# figure = plt.figure(); plt.imshow(np.log(np.abs(np.concatenate((sf.im_to_kspace(cg_sense),sf.im_to_kspace(cg_sense_OVS),sf.im_to_kspace(cg_sense_mask),sf.im_to_kspace(cg_sense_diff)), axis=1))), cmap="gray", vmax=10); plt.axis('off')


# %% results with sense
# no OVS processing
sense = sf.sense(y1, Smaps2)
# OVS from k-space
sense_OVS = sf.sense(y1_diff, Smaps2)
# OVS from k-space and calibration in image space
sense_mask = sf.sense(y1_diff, Smaps2_mask)
# OVS from k-space and calibration in k-space 
sense_diff = sf.sense(y1_diff, Smaps2_diff)

background = im_composite * ovs_mask * 0
figure = plt.figure(); plt.imshow(np.abs(np.concatenate((sense,sense_OVS+background,sense_mask+background,sense_diff+background), axis=1)), cmap="gray", vmax=700); plt.axis('off')
plt.title('Results for espirit Smaps'); plt.axis('off')

# figure = plt.figure(); plt.imshow(np.log(np.abs(np.concatenate((sf.im_to_kspace(sense),sf.im_to_kspace(sense_OVS),sf.im_to_kspace(sense_mask),sf.im_to_kspace(sense_diff)), axis=1))), cmap="gray", vmax=10); plt.axis('off')


# %% Condition number check
# Sx = y,    inv(SHS) is needed
# condition number of SHS matrix
y = np.array([110, 130, 150])
con_num = np.zeros((3, y.shape[0]))
for i in np.arange(y.shape[0]):
    A = Smaps2[y[i], 4:-1:Ny//8, :]
    con_num[0,i] = np.linalg.cond(np.matmul(A, np.conj(A).T))
print(f'Condition numbers for Smaps: {con_num[0,0]:.2e}, {con_num[0,1]:.2e} and {con_num[0,2]:.2e}')

for i in np.arange(y.shape[0]):
    B = Smaps2_mask[y[i], 4:-1:Ny//8, :]
    A = B[np.sum(B,1)!=0]
    con_num[1,i] = np.linalg.cond(np.matmul(A, np.conj(A).T))
print(f'Condition numbers for Smaps: {con_num[1,0]:.2e}, {con_num[1,1]:.2e} and {con_num[1,2]:.2e}')


for i in np.arange(y.shape[0]):
    A = Smaps2_diff[y[i], 4:-1:Ny//8, :]
    con_num[2,i] = np.linalg.cond(np.matmul(A, np.conj(A).T))
print(f'Condition numbers for Smaps: {con_num[2,0]:.2e}, {con_num[2,1]:.2e} and {con_num[2,2]:.2e}')


# %% g-factor maps
gmap = sf.gfactor_MC(Smaps2)
gmap_mask = sf.gfactor_MC(Smaps2_mask)
gmap_diff = sf.gfactor_MC(Smaps2_diff)

gmax = np.max(gmap)

figure = plt.figure()
plt.subplot(1,3,1)
plt.imshow(np.abs(gmap),cmap="turbo",vmax=gmax)
plt.axis("off")
plt.title("g-factor map of full sens map")
plt.subplot(1,3,2)
plt.imshow(np.abs(gmap_mask),cmap="turbo",vmax=gmax)
plt.axis("off")
plt.title("g-factor map of masked sens map")
plt.subplot(1,3,3)
plt.imshow(np.abs(gmap_diff),cmap="turbo",vmax=gmax)
plt.axis("off")
plt.title("g-factor map of ovs sens map")
plt.colorbar()


    

