import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat

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

full_data_path = "C:\\Codes\\p006_OVS\\OVS\\TestDatasetSmall"
ovs_data_path = "C:\\Codes\\p006_OVS\\OVS\\TestDatasetSmallRectangularMask"

data_indices = loadmat("C:\\Codes\\p006_OVS\\OVS\\test_data_indices.mat")
_, indices = list(data_indices.items())[3]

number_of_subjects = indices.shape[1]

for sub in range(number_of_subjects):
    subject_number = indices[0,sub][0,0]
    print('subject number = '+ f'{indices[0,sub][0,0]}')
    slc_counter = 0
    for slc in indices[1,sub][0]:
        print('slice number = '+ f'{slc}')
        for TF in range(5):
            time_frame_no = indices[2,sub][slc_counter,TF]-1
            print('time frame = '+ f'{time_frame_no}')
            filename = "C:\\Codes\\p006_OVS\\OVS\\TestDatasetSmall\\subject_" + str(indices[0,sub][0,0]) + "_slice_" + str(slc) + "_" + str(TF+1) + ".mat"
            composite_kspace = loadmat(filename)['composite_kspace'][0]
            # acc_mask = loadmat(filename)['acc_mask']
            filename = "C:\\Codes\\p006_OVS\\OVS\\TestDatasetSmallCircularMask\\subject_" + str(indices[0,sub][0,0]) + "_slice_" + str(slc) + "_" + str(TF+1) + ".mat"
            # diff_com_kspace = loadmat(filename)['diff_com_kspace'][0]
            # sense_maps_full = loadmat(filename)['sense_maps_full'][0]
            ovs_mask3 = loadmat(filename)['ovs_mask3']
            # y_background3 = (composite_kspace - diff_com_kspace) * acc_mask[...,None]
            # background = cgsense(y_background3,sense_maps_full,acc_mask)
            background = rssq(composite_kspace)
            filename = "C:\\Codes\\p006_OVS\\OVS\\Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\\background004\\subject_"+str(subject_number)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat"

            datadir = {"background": background,
                       "ovs_mask": ovs_mask3}
            savemat(filename, datadir) 

        slc_counter = slc_counter + 1



