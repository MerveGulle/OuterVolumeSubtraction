from scipy.io import loadmat
import numpy as np
import os
import mat73
from espirit_shifted import espirit
from scipy.io import savemat

# parameters
K = 5        # number of masks
acs_size = 8 # DC calibration region size
rho = 0.6    # data consistency / whole mask

# get data indices
data_indices = loadmat('data_indices.mat')
_, indices = list(data_indices.items())[3]
# indices.shape = (3,14)
# indices[0]: subject numbers
# indices[1]: slice numbers
# indices[2]: time frame numbers (number of slices x 3)

os.chdir('/home/daedalus1-raid1/akcakaya-group-data/ScannerData/Data/Volunteer/RealTime_fromPK/database_short_axis')
number_of_subjects = indices.shape[1]
sub_counter = 0
for sub in np.arange(number_of_subjects):
    print('subject number = '+ f'{sub}')
    
    slc_counter = 0
    for slc in indices[1,sub][0]:
        print('slice number = '+ f'{slc}')
        
        # read the data
        filename = "subject_" + str(sub) + "_slice_" + str(slc) + ".mat"
        filename = "subject_" + str(sub) + "_slice_" + str(slc) + ".mat"
        realtime_data = mat73.loadmat(filename)
        
        _, kspace_data = list(realtime_data.items())[1]
        
        for TF in range(3):
            time_frame_no = indices[2,sub_counter][slc_counter,TF]-1
            print('time frame = '+ f'{time_frame_no+1}')
            
            # data indices
            sub_slc_tf = np.array([sub,slc,time_frame_no])
            
            # composite kspace
            composite_kspace = np.sum(kspace_data[:,:,:,time_frame_no:time_frame_no+4],3)
            composite_kspace = composite_kspace[np.abs(np.sum(composite_kspace[:,:,0],1))!=0]
            composite_kspace = composite_kspace[:,np.abs(np.sum(composite_kspace[:,:,0],0))!=0]
            composite_kspace = (composite_kspace/np.max(np.abs(composite_kspace)))[None,...]
            
            # sensitivity maps
            sense_maps = espirit(composite_kspace, 8 ,40, 0.02, 0.95)
            sense_maps = sense_maps[0,:,:,:,0:2].transpose(3,0,1,2)
            
            # acceleration mask
            [_,Nx,Ny,Nc] = composite_kspace.shape
            start_line = np.mod(time_frame_no, 8)
            Cx = np.argmax(np.abs(composite_kspace).sum((0,2,3))) # center line in x direction
            Cy = np.argmax(np.abs(composite_kspace).sum((0,1,3))) # center line in y direction
            center_line = Cy + 3 - np.mod(Cy-start_line-1, 8)
            acc_mask = np.zeros((Nx,Ny), dtype=bool)
            acc_mask[:,start_line::8] = True
            acc_mask[:,center_line] = True
            
            # data consistency masks
            size = np.floor(Nx * Ny * K * rho).astype(int)
            ind = np.random.choice(np.arange(Nx * Ny * K), size=size, replace=False)
            DC_masks = np.zeros(Nx*Ny*K)
            DC_masks[ind] = 1
            DC_masks = DC_masks.reshape((Nx,Ny,K))
            DC_masks[Cx-acs_size//2:Cx+acs_size//2,Cy-acs_size//2:Cy+acs_size//2] = 1
            DC_masks = DC_masks * acc_mask[...,None]
            DC_masks = (DC_masks==1)
            
            # save the slice data
            datadir = {"composite_kspace": composite_kspace, 
                       "sense_maps": sense_maps,
                       "acc_mask": acc_mask,
                       "data_consistency_masks": DC_masks,
                       "sub_slc_tf": sub_slc_tf}
            data_filename = "/home/naxos2-raid12/glle0001/TrainData/subject_" + str(sub) + "_slice_" + str(slc) + "_" + str(TF+1) + ".mat"
            savemat(data_filename, datadir)
        
        slc_counter += 1
    sub_counter += 1