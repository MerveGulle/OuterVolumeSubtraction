import model
import torch
import numpy as np
import random
import SupportingFunctions as sf
import sys
from matplotlib import pyplot as plt


### HYPERPARAMETERS
params = dict([('num_epoch', 20),
               ('batch_size', 1),
               ('learning_rate', 1e-3),
               ('num_training_slice', 10),
               ('num_workers', 0),          # It should be 0 for Windows machines
               ('use_cpu', False),
               ('num_mask', 2),             # number of masks
               ('T', 3)])                  # number of iterations

### PATHS 
train_data_path = "C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_full\TrainDataset"
# train_data_path = "/home/naxos2-raid12/glle0001/TrainData/"


# 0) Fix randomness for reproducible experiment
torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# 1) Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and (not(params['use_cpu']))) else 'cpu')

# 2) Load the Train Data
dataset = sf.OVS_DatasetTrain(train_data_path,params['num_training_slice'])
loaders, datasets= sf.prepare_train_loaders(dataset,params,g)

# 3) Create Model structure
denoiser = model.ResNet().to(device)
denoiser.apply(model.weights_init_normal)
optimizer = torch.optim.Adam(denoiser.parameters(),lr=params['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# 4) Loss Function Definition
def L1L2Loss(recon, ref):
    return torch.norm(recon-ref,p=1)/torch.norm(ref,p=1) + torch.norm(recon-ref,p=2)/torch.norm(ref,p=2)

# 5) Start Training
# Set the initial loss values
loss_arr       = np.zeros(params['num_epoch'])
loss_arr_valid = np.zeros(params['num_epoch'])

# training
for epoch in range(params['num_epoch']):
    for i, (x0, composite_kspace, sense_map, acc_mask, data_consistency_masks, sub_slc_tf, index) in enumerate(loaders['train_loader']):
        x0                     = x0[0].to(device)                       # [K,Nx,Ny]
        composite_kspace       = composite_kspace[0].to(device)         # [1,Nx,Ny,Nc]
        sense_map              = sense_map[0].to(device)                # [2,Nx,Ny,Nc]
        acc_mask               = acc_mask[0].to(device)                 # [Nx,Ny]
        data_consistency_masks = data_consistency_masks[0].to(device)   # [Nx,Ny,K]
        # Forward pass
        loss = 0
        for k in range(params['num_mask']):
            loss_mask = acc_mask - data_consistency_masks[...,k]
            xk0 = x0[2*k:2*k+2]   # x0 for kth DC mask
            xt = xk0              # iteration starts with xt
            for t in range(params['T']):
                L, zt = denoiser(xt[None,...])
                xt = model.DC_layer(xk0,zt[0],L,sense_map,data_consistency_masks[...,k])
            # loss calculation for kth mask
            kspace_loss = sf.forward(xt, sense_map, loss_mask)
            # loss calculation
            loss += L1L2Loss(kspace_loss, composite_kspace*loss_mask[None,...,None])/params['num_mask']
            
        optimizer.zero_grad()
        
        if (torch.isnan(loss)):
            torch.save(denoiser.state_dict(), 'OVS_multimaskSSDU_' + f'{epoch:03d}'+ '.pt')
            print ('-----------------------------')
            print (f'Epoch [{epoch+1}/{params["num_epoch"]}], \
                   loss: {loss:.08f}, \
                   index: {index}')
            print ('-----------------------------')
            torch.save(loss_arr, 'train_loss.pt')
            torch.save(loss_arr_valid, 'valid_loss.pt')
            sys.exit()
            
        loss_arr[epoch] += loss.item()/len(datasets['train_dataset'])
        loss.backward()
        
        # Optimize
        optimizer.step()

    for i, (x0, composite_kspace, sense_map, acc_mask, data_consistency_masks, sub_slc_tf, index) in enumerate(loaders['train_loader']):
        x0                     = x0[0].to(device)                       # [K,Nx,Ny]
        composite_kspace       = composite_kspace[0].to(device)         # [1,Nx,Ny,Nc]
        sense_map              = sense_map[0].to(device)                # [2,Nx,Ny,Nc]
        acc_mask               = acc_mask[0].to(device)                 # [Nx,Ny]
        data_consistency_masks = data_consistency_masks[0].to(device)   # [Nx,Ny,K]
        # Forward pass
        loss = 0
        for k in range(params['num_mask']):
            loss_mask = acc_mask - data_consistency_masks[...,k]
            xk0 = x0[2*k:2*k+2]   # x0 for kth DC mask
            xt = xk0              # iteration starts with xt
            for t in range(params['T']):
                L, zt = denoiser(xt[None,...])
                xt = model.DC_layer(xk0,zt[0],L,sense_map,data_consistency_masks[...,k])
            # loss calculation for kth mask
            kspace_loss = sf.forward(xt, sense_map, loss_mask)
            # loss calculation
            loss += L1L2Loss(kspace_loss, composite_kspace*loss_mask[None,...,None])/params['num_mask']
        loss_arr_valid[epoch] += loss.item()/len(datasets['valid_dataset'])
        
    if ((epoch+1)%5==0):
        torch.save(denoiser.state_dict(), 'OVS_multimaskSSDU_' + f'{epoch+1:03d}'+ '.pt')
        torch.save(loss_arr, 'train_loss.pt')
        torch.save(loss_arr_valid, 'valid_loss.pt')
        torch.save(L, 'L.pt')
          
    scheduler.step()
    
    print ('-----------------------------')
    print (f'Epoch [{epoch+1}/{params["num_epoch"]}], \
           Loss training: {loss_arr[epoch]:.6f}, \
           Loss validation: {loss_arr_valid[epoch]:.6f}')
    print ('-----------------------------')

breakpoint()
# 6) Plot the Loss Graph
figure = plt.figure()
n_epoch = np.arange(1,params['num_epoch']+1)
plt.plot(n_epoch,loss_arr,n_epoch,loss_arr_valid)
plt.xlabel('epoch')
plt.title('Loss Graph')
plt.legend(['train loss', 'validation loss'])
figure.savefig('loss_graph.png')

torch.save(loss_arr, 'train_loss.pt')
torch.save(loss_arr_valid, 'valid_loss.pt')