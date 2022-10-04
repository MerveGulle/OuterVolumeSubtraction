import numpy as np
import SupportingFunctions as sf
import mat73
import matplotlib.pyplot as plt
from espirit import espirit
from scipy.io import loadmat
# from scipy.io import savemat


# %% Hyperparameters
mask_type = "only_heart"
# mask_type = only_heart or outer_volume or no_diaphragm
subject_number = 3
slice_number = 4


# %% Data Loading
# Load one slice data
filename = "subject_" + str(subject_number) + "_slice_" + str(slice_number) + ".mat"
realtime_data = mat73.loadmat(filename)
# Images for each time frame, created with TGRAPPA --> [Nx(full), Ny(full), Ndyn]
_, im_tgrappa_full = list(realtime_data.items())[0]
im_tgrappa_full = im_tgrappa_full[:,:,0:40]
# Kspace data (partial) --> [Nx(full), Ny(full), Nc, Ndyn]
_, datas = list(realtime_data.items())[1]
datas = datas[:,:,:,0:40]
datas = datas[68:320,0:72,:,0:40]*1e6
datas = datas.astype('complex64')

Nx = datas.shape[0]             # num of RO pixels (vertical)
Ny = datas.shape[1]             # num of PE pixels (horizontal)
Nc = datas.shape[2]             # num of coils
Ndyn = datas.shape[3]           # num of time frames
"""
figure = plt.figure(figsize=(8,4))
for tf in np.arange(4):
    plt.subplot(1,4,tf+1)
    plt.imshow(np.log(np.abs(datas[:,:,0,tf])+1),cmap="gray")
    plt.title('TF #'+f'{tf+1}',fontsize=10)
    plt.axis("off")
plt.suptitle("acquired data with R=4",fontsize=12)
"""

# %% images for the first 24 time frames
"""
num_img = 24
    
figure = plt.figure(figsize=(15,12))
for tf in np.arange(num_img):
    plt.subplot(3,num_img//3,tf+1)
    plt.imshow(np.abs(im_tgrappa_full[145:185,40:90,tf]),cmap="gray",vmax=0.1)
    plt.title('TF #'+f'{tf+1}',fontsize=12)
    plt.axis("off")
"""

# %% Cardiac contraction : time frame = 7 (Ndyn=6)
TF = 0
shift = np.mod(TF, 8)


# %% Zerofilled image
# acceleration mask
acc_mask = np.zeros((Nx,Ny), dtype=bool)
acc_mask[:,shift::8] = True
acc_mask[:,Ny-4:Ny] = True 
# zerofilled image:x0
x0 = sf.rssq(sf.kspace_to_im(datas[:,:,:,TF]*acc_mask[...,None]))
"""
figure = plt.figure(); plt.imshow(np.abs(x0), cmap="gray"); 
plt.title('zerofilled image'); plt.axis('off')
"""


# %% composite data
y_com = np.zeros([Nx,Ny,Nc,Ndyn-3], dtype=np.complex64)
for ind in range(Ndyn-3):
    y_com[:,:,:,ind] = np.sum(datas[:,:,:,ind:ind+4],3)
"""
figure = plt.figure()
for tf in np.arange(8):
    plt.subplot(1,8,tf+1)
    plt.imshow(np.log(np.abs(y_com[:,:,0,tf])),cmap="gray")
    plt.title('TF #'+f'{tf+1}',fontsize=10)
    plt.axis("off")
plt.suptitle("composite kspace",fontsize=12)
"""


# %% composite image
x_com = sf.rssq(sf.kspace_to_im(y_com))
im_composite = x_com[:,:,TF]
"""
figure = plt.figure(figsize=(2,6)); plt.imshow(np.abs(im_composite), cmap="gray", vmax=2000); plt.axis('off')
plt.title('composite image'); plt.axis('off')
"""


# %% Subtraction the outer volume signal
filename = str(mask_type) + "_subject_" + str(subject_number) + "_slice_" + str(slice_number) + ".mat"
ovs_mask = loadmat(filename)['target_mask']
# subtact these out from the data
y_com1 = y_com[:,:,:,TF]
y_background = sf.im_to_kspace(sf.kspace_to_im(y_com1)*ovs_mask[...,None])
# subtract out background
y1 = datas[:,:,:,TF]*acc_mask[...,None]
y1_diff = y1 - y_background*acc_mask[...,None]
y1_diff[:,Ny-4:Ny] = 0
"""
figure = plt.figure(); plt.imshow(sf.rssq(sf.kspace_to_im(y1_diff)), cmap="gray"); plt.axis('off')
plt.title('OVS zerofilled image'); plt.axis('off')
figure = plt.figure(); plt.imshow(sf.rssq(sf.kspace_to_im(y1)), cmap="gray"); plt.axis('off') 
plt.title('zerofilled image'); plt.axis('off')
"""


# %% Generate coil maps with espirit
# k: kspace kernel size
# r: calibration region size
if (mask_type == "only_heart"):
    k = 14
elif (mask_type == "no_diaphragm"):
    k = 16
elif (mask_type == "outer_volume"):
    k = 8
# full smaps
Smaps = espirit(y_com1[None,...], 8, 44, 0.02, 0.95)
Smaps = Smaps[0,:,:,:,0]
# masked smaps
Smaps_mask = Smaps * (1 - ovs_mask[...,None])
# outer volume signal subtracted smaps
y_com_diff = y_com1-y_background
y_com_diff[:,Ny-4:Ny] = 0
Smaps_diff = espirit((y_com_diff)[None,...], k, 44, 0.02, 0.95)
Smaps_diff = Smaps_diff[0,:,:,:,0]

coils_to_visualize = np.array([0,4,8,12,16,20,24,28])

"""
figure = plt.figure();
plt.imshow(np.abs(Smaps[:,:,coils_to_visualize].swapaxes(0,2).reshape(8*Ny,Nx).swapaxes(0,1)), cmap="gray", vmax = 0.5); 
plt.axis('off')
plt.title('Full Sensitivity Maps - Amplitude')

figure = plt.figure(); 
plt.imshow(np.abs(Smaps_mask[:,:,coils_to_visualize].swapaxes(0,2).reshape(8*Ny,Nx).swapaxes(0,1)), cmap="gray", vmax = 0.5); 
plt.axis('off') 
plt.title('Masked Sensitivity Maps - Amplitude')

figure = plt.figure(); 
plt.imshow(np.abs(Smaps_diff[:,:,coils_to_visualize].swapaxes(0,2).reshape(8*Ny,Nx).swapaxes(0,1)), cmap="gray", vmax = 0.5); 
plt.axis('off') 
plt.title('OVS Sensitivity Maps - Amplitude')


figure = plt.figure();
plt.imshow(np.angle(Smaps[:,:,coils_to_visualize].swapaxes(0,2).reshape(8*Ny,Nx).swapaxes(0,1)), cmap="gray"); 
plt.axis('off')
plt.title('Full Sensitivity Maps - Phase')

figure = plt.figure(); 
plt.imshow(np.angle(Smaps_mask[:,:,coils_to_visualize].swapaxes(0,2).reshape(8*Ny,Nx).swapaxes(0,1)), cmap="gray"); 
plt.axis('off') 
plt.title('Masked Sensitivity Maps - Phase')

figure = plt.figure(); 
plt.imshow(np.angle(Smaps_diff[:,:,coils_to_visualize].swapaxes(0,2).reshape(8*Ny,Nx).swapaxes(0,1)), cmap="gray"); 
plt.axis('off')  
plt.title('OVS Sensitivity Maps - Phase')
"""  

# %% Reference Images
cg_sense_ref = sf.cgsense(datas[:,:,:,TF], Smaps, np.abs(datas[:,:,0,TF])!=0)
# figure = plt.figure(); plt.imshow(np.abs(cg_sense_ref),cmap="gray",vmax=1800)


# %% results with cgsense
# no OVS processing
cg_sense = sf.cgsense(y1, Smaps, acc_mask)
# OVS from k-space
cg_sense_OVS = sf.cgsense(y1_diff, Smaps, acc_mask)
# OVS from k-space and calibration in image space
cg_sense_mask = sf.cgsense(y1_diff, Smaps_mask, acc_mask)
# OVS from k-space and calibration in k-space 
cg_sense_diff = sf.cgsense(y1_diff, Smaps_diff, acc_mask)

background = im_composite * ovs_mask
figure = plt.figure(); plt.imshow(np.abs(np.concatenate((np.abs(cg_sense),np.abs(cg_sense_OVS)+background,np.abs(cg_sense_mask)+background,np.abs(cg_sense_diff)+background,np.abs(cg_sense_ref)), axis=1)), cmap="gray", vmax=1800); plt.axis('off')
plt.title('Results for CG-SENSE'); plt.axis('off')


# %% results with admm
# no OVS processing
admm = sf.ADMM(y1, Smaps, acc_mask, np.max(np.abs(sf.haar2(cg_sense))))
# OVS from k-space
admm_OVS = sf.ADMM(y1_diff, Smaps, acc_mask, np.max(np.abs(sf.haar2(cg_sense_OVS))))
# OVS from k-space and calibration in image space
admm_mask = sf.ADMM(y1_diff, Smaps_mask, acc_mask, np.max(np.abs(sf.haar2(cg_sense_mask))))
# OVS from k-space and calibration in k-space 
admm_diff = sf.ADMM(y1_diff, Smaps_diff, acc_mask, np.max(np.abs(sf.haar2(cg_sense_diff))))

background = im_composite * ovs_mask
figure = plt.figure(); plt.imshow(np.abs(np.concatenate((np.abs(admm),np.abs(admm_OVS)+background,np.abs(admm_mask)+background,np.abs(admm_diff)+background,np.abs(cg_sense_ref)), axis=1)), cmap="gray", vmax=1800); plt.axis('off')
plt.title('Results for ADMM'); plt.axis('off')



  