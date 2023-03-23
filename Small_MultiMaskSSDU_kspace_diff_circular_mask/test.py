import torch
import SupportingFunctions as sf
import model
from scipy.io import savemat
from matplotlib import pyplot as plt
import numpy as np


### HYPERPARAMETERS
params = dict([('sense_maps_type', 'sense_maps_full'),  #1 # 'sense_maps_full', 'sense_maps_diff', 'sense_maps_mask'
               ('num_epoch', 100),
               ('batch_size', 1),
               ('learning_rate', 1e-3),
               ('num_training_slice', 'all'),
               ('num_validation_slice', 'all'),
               ('num_test_slice', 'all'),
               ('num_workers', 0),          # It should be 0 for Windows machines
               ('use_cpu', False),
               ('num_mask', 3),             # number of masks
               ('T', 10)])                  # number of iterations

### PATHS 
test_data_path = "C:\Codes\p006_OVS\OVS\TestDatasetSmallCircularMaskNew"
# test_data_path = "C:\Codes\p006_OVS\OVS\subject3slice4TF17"
# test_data_path  = "/home/naxos2-raid12/glle0001/TestDataCircularMask/"

# 1) Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and (not(params['use_cpu']))) else 'cpu')

# 2) Load the Train & Validation Data
test_dataset = sf.OVS_DatasetTest(test_data_path, params['sense_maps_type'], params['num_test_slice'])
test_loader, test_datasets= sf.prepare_test_loaders(test_dataset,params)

###############################################################################
############## TEST CODE ######################################################
###############################################################################

denoiser = model.ResNet().to(device)
denoiser.load_state_dict(torch.load('OVS_multimaskSSDU_full.pt'))  #2
denoiser.eval()

for i, (x0, kspace_data8, sense_maps, composite_image, acc_mask, ovs_mask, im_tgrappa, sub_slc_tf, index) in enumerate(test_loader['test_loader']):
    with torch.no_grad():
        # Forward pass
        x0                     = x0[0].to(device)                       # [1,Nx,Ny]
        kspace_data8           = kspace_data8[0].to(device)             # [1,Nx,Ny,Nc]
        sense_maps             = sense_maps[0].to(device)               # [1,Nx,Ny,Nc]
        composite_image        = composite_image[0].to(device).detach().cpu().numpy()          # [1,Nx,Ny]
        acc_mask               = acc_mask[0].to(device)                 # [Nx,Ny]
        ovs_mask               = ovs_mask[0].to(device).detach().cpu().numpy()              # [Nx,Ny]
        im_tgrappa             = im_tgrappa[0].to(device)               # [Nx,Ny]
        # Forward pass
        xt = torch.clone(x0)
        for t in range(params['T']):
            L, zt = denoiser(xt[None,...])
            xt = model.DC_layer(x0,zt[0],L,sense_maps,acc_mask)
        
        #zerofilled = x0[0].detach().cpu().numpy()*(1-ovs_mask)
        cg_sense = sf.cgsense(x0,kspace_data8,sense_maps,acc_mask)[0].detach().cpu().numpy()
        SSDU = xt[0].detach().cpu().numpy()
        
        
        #im_tgrappa = im_tgrappa.detach().cpu().numpy()
        sub = sub_slc_tf[0,0,0].detach().cpu().numpy()
        slc = sub_slc_tf[0,0,1].detach().cpu().numpy()
        tf  = sub_slc_tf[0,0,2].detach().cpu().numpy()
        
        filename = "C:\Codes\p006_OVS\OVS\Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_full_008\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(tf)+".mat" #3

        datadir = {"SSDU_kdiff_Sfull": SSDU, #4
                   "cg_sense": cg_sense}
        savemat(filename, datadir) 





