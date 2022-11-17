import torch
import SupportingFunctions as sf
import model


### HYPERPARAMETERS
params = dict([('num_epoch', 100),
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
test_data_path = "C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_full\TestData"
# test_data_path  = "/home/naxos2-raid12/glle0001/TestData/"

# 1) Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and (not(params['use_cpu']))) else 'cpu')

# 2) Load the Train & Validation Data
test_dataset = sf.OVS_DatasetTest(test_data_path, params['num_test_slice'])
test_loader, test_datasets= sf.prepare_test_loaders(test_dataset,params)

###############################################################################
############## TEST CODE ######################################################
###############################################################################

denoiser = model.ResNet().to(device)
denoiser.load_state_dict(torch.load('OVS_multimaskSSDU_045.pt'))
denoiser.eval()

for i, (x0, composite_kspace, sense_map, acc_mask, sub_slc_tf, index) in enumerate(test_loader['test_loader']):
    with torch.no_grad():
        # Forward pass
        x0                     = x0[0].to(device)                       # [2,Nx,Ny]
        composite_kspace       = composite_kspace[0].to(device)         # [1,Nx,Ny,Nc]
        sense_map              = sense_map[0].to(device)                # [2,Nx,Ny,Nc]
        acc_mask               = acc_mask[0].to(device)                 # [Nx,Ny]
        # Forward pass
        xt = torch.clone(x0)
        for t in range(params['T']):
            L, zt = denoiser(xt[None,...])
            xt = model.DC_layer(x0,zt[0],L,sense_map,acc_mask)







