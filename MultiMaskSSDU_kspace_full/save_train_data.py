from scipy.io import loadmat
import numpy as np
import os
import mat73


data_indices = loadmat('data_indices.mat')
_, indices = list(data_indices.items())[3]
# indices.shape = (3,14)
# indices[0]: subject numbers
# indices[1]: slice numbers
# indices[2]: time frame numbers (number of slices x 3)

number_of_subjects = indices.shape[1]
sub_counter = 0
for sub in np.arange(number_of_subjects):
    print('subject number = '+ f'{sub}')
    
    slc_counter = 0
    for slc in indices[1,sub][0]:
        print('slice number = '+ f'{slc}')
        
        os.chdir('/home/daedalus1-raid1/akcakaya-group-data/ScannerData/Data/Volunteer/RealTime_fromPK/database_short_axis')
        filename = "subject_" + str(sub) + "_slice_" + str(slc) + ".mat"
        filename = "subject_" + str(sub) + "_slice_" + str(slc) + ".mat"
        realtime_data = mat73.loadmat(filename)
        
        slc_counter += 1
    sub_counter += 1