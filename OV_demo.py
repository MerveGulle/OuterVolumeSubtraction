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
Nc = datas.shape[2]          # num of coils
Ndyn = datas.shape[3]        # num of time frames