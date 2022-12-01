from scipy.io import loadmat
import numpy as np


data_indices = loadmat('test_data_indices.mat')
_, indices = list(data_indices.items())[3]
# indices.shape = (3,2)
# indices[0]: subject numbers
# indices[1]: slice numbers
# indices[2]: time frame numbers (number of slices x 5)

number_of_subjects = indices.shape[1]

for sub_counter in range(number_of_subjects):
    sub = indices[0,sub_counter][0,0]
    print('subject number = '+ f'{sub}')
    
    slc_counter = 0
    for slc in indices[1,sub_counter][0]:
        print('slice number = '+ f'{slc}')
        
        for TF in range(5):
            time_frame_no = indices[2,sub_counter][slc_counter,TF]-1
            print('time frame = '+ f'{time_frame_no}')
            
            slice_data = loadmat("C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_full\Results\Smaps_full_001\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            im_tgrappa = slice_data['im_tgrappa']
            kfull_Sfull = slice_data['SSDU_kfull_Sfull']
            slice_data = loadmat("C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_diff\Results\Smaps_full_001\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            
            breakpoint()
            
        slc_counter +=1