import model
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import SupportingFunctions as sf
import sys

print('Training code has been started.')

### HYPERPARAMETERS
params = dict([('num_epoch', 500),
               ('batch_size', 1),
               ('learning_rate', 3e-4),
               ('num_training_slice', 'all'),
               ('num_validation_slice', 'all'),
               ('num_test_slice', 'all'),
               ('num_workers', 0),          # It should be 0 for Windows machines
               ('use_cpu', False)])          

### PATHS          
train_data_path = "PreTrainDataset"
valid_data_path = "PreValidDataset"
                 
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

# 2) Load the Train & Validation Data
train_dataset = sf.TrainImages(train_data_path, params['num_training_slice'])
train_loader, train_datasets= sf.prepare_train_loaders(train_dataset,params)

validation_dataset = sf.ValidationImages(valid_data_path,params['num_validation_slice'])
validation_loader, validation_datasets= sf.prepare_valid_loaders(validation_dataset,params)


# 3) Create Model structure
denoiser = model.ResNet().to(device)
denoiser.apply(model.weights_init_normal)
optimizer = torch.optim.Adam(denoiser.parameters(),lr=params['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

loss_arr       = np.zeros(params['num_epoch'])
loss_arr_valid = np.zeros(params['num_epoch'])

for epoch in range(params['num_epoch']):
    for i, (reference, noisy, sub_slc_tf, index) in enumerate(train_loader['train_loader']):
        reference = reference.to(device)
        noisy     = noisy.to(device)
        # Forward pass
        L, recon = denoiser(noisy)
        
        optimizer.zero_grad()
        # Loss calculation
        #loss = sf.L1L2Loss(xref, xk)
        loss = sf.L1L2Loss(reference, recon)
        
        if (torch.isnan(loss)):
            torch.save(denoiser.state_dict(), 'model_t_' + f'_ResNet_{epoch:03d}'+ '.pt')
            print ('-----------------------------')
            print (f'Epoch [{epoch+1}/{params["num_epoch"]}], \
                   loss: {loss:.08f}, \
                   index: {index}')
            print ('-----------------------------')
            torch.save(loss_arr, 'train_loss.pt')
            torch.save(loss_arr_valid, 'valid_loss.pt')
            sys.exit()
            
        loss_arr[epoch] += loss.item()/len(train_datasets['train_dataset'])
        loss.backward()
        
        # Optimize
        optimizer.step()
        
    for i, (reference, noisy, sub_slc_tf, index) in enumerate(train_loader['train_loader']):
        with torch.no_grad():
            reference = reference.to(device)
            noisy     = noisy.to(device)
            # Forward pass
            L, recon = denoiser(noisy)
            
            # Loss calculation
            #loss = sf.L1L2Loss(xref, xk)
            loss = sf.L1L2Loss(reference, recon)
            loss_arr_valid[epoch] += loss.item()/len(validation_datasets['valid_dataset'])
        
        if ((epoch+1)%10==0):
            torch.save(denoiser.state_dict(), 'model_t_' + f'_ResNet_{epoch+1:03d}'+ '.pt')
            torch.save(loss_arr, 'train_loss.pt')
            torch.save(loss_arr_valid, 'valid_loss.pt')
    scheduler.step()
   
    print ('-----------------------------')
    print (f'Epoch [{epoch+1}/{params["num_epoch"]}], \
           Loss training: {loss_arr[epoch]:.4f}, \
           Loss validation: {loss_arr_valid[epoch]:.4f}')
    print ('-----------------------------')

figure = plt.figure()
n = np.arange(1,params['num_epoch']+1)
plt.plot(n,loss_arr,n,loss_arr_valid)
plt.xlabel('epoch')
plt.title('Loss Graph')
plt.legend(['train loss', 'validation loss'])
figure.savefig('loss_graph.png')
