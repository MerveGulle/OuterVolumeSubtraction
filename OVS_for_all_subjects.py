import numpy as np
import SupportingFunctions as sf
import mat73
import matplotlib.pyplot as plt
from espirit import espirit
from scipy.io import loadmat


# %% Data Loading
# Load one slice data
subject_number = 3
filename = "subject" + str(subject_number) + ".mat"
realtime_data = mat73.loadmat(filename)
# Images for each time frame, created with TGRAPPA --> [Nx(full), Ny(full), Ndyn]
_, im_tgrappa_full = list(realtime_data.items())[0]
# Kspace data (partial) --> [Nx(full), Ny(full), Nc, Ndyn]
_, datas = list(realtime_data.items())[1]
datas = datas[68:320,0:72,:,:]*1e6
datas = datas.astype('complex64')

Nx = datas.shape[0]             # num of RO pixels (vertical)
Ny = datas.shape[1]             # num of PE pixels (horizontal)
Nc = datas.shape[2]             # num of coils
Ndyn = datas.shape[3]           # num of time frames


# %% images for the first 24 time frames
num_img = 24
    
figure = plt.figure(figsize=(15,12))
for tf in np.arange(num_img):
    plt.subplot(3,num_img//3,tf+1)
    plt.imshow(np.abs(im_tgrappa_full[123:213,23:103,tf]),cmap="gray",vmax=0.2)
    plt.title('TF #'+f'{tf+1}',fontsize=12)
    plt.axis("off")
    

# %% time frame = 22 (Ndyn=21) : Cardiac contraction
TF = 21
shift = np.mod(TF, 8)


    
    
    