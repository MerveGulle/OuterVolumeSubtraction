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
















