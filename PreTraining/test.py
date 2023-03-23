import model
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import SupportingFunctions as sf
import os
from skimage.metrics import structural_similarity as ssim
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print('Test code has been started.')

### HYPERPARAMETERS
### HYPERPARAMETERS
params = dict([('num_epoch', 100),
               ('batch_size', 1),
               ('learning_rate', 3e-4),
               ('num_training_slice', 'all'),
               ('num_validation_slice', 'all'),
               ('num_test_slice', 'all'),
               ('num_workers', 0),          # It should be 0 for Windows machines
               ('use_cpu', False)])                  # number of iterations  

### PATHS          
test_data_path = "PreTestDataset" 
                  
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

# 2) Load Data
test_dataset = sf.TestImages(test_data_path, params['num_test_slice'])
test_loader, test_datasets= sf.prepare_test_loaders(test_dataset,params)

####################################################
############## TEST CODE ###########################
####################################################
denoiser = model.ResNet().to(device)
denoiser.load_state_dict(torch.load('model_t__ResNet_200.pt'))
denoiser.eval()

for i, (reference, noisy, sub_slc_tf, index) in enumerate(test_loader['test_loader']):
    with torch.no_grad():
        reference = reference.to(device)
        noisy     = noisy.to(device)
        # Forward pass
        L, recon = denoiser(noisy)
            
        reference = np.abs(reference.cpu().detach().numpy()[0,0])
        noisy = np.abs(noisy.cpu().detach().numpy()[0,0])
        recon = np.abs(recon.cpu().detach().numpy()[0,0])
        
        data_range=reference.max()
        ssim_0 = ssim(reference, noisy, data_range=data_range)
        ssim_f = ssim(reference, recon, data_range=data_range)
        nmse_0 = sf.nmse(noisy,reference)
        nmse_f = sf.nmse(recon,reference)
        
        props = dict(boxstyle='round', facecolor='black', alpha=0.8)
        figure = plt.figure(figsize=(6,4))
        plt.subplot(1,3,1)
        plt.imshow(noisy,cmap='gray',vmax=data_range/5)
        ax = plt.gca()
        plt.text(0.5, 0.1, 'NMSE:'+f'{nmse_0:,.3f}'+'\nSSIM:'+f'{ssim_0:,.3f}', color = 'white', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
        plt.title('noisy')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(1,3,2)
        plt.imshow(recon,cmap='gray',vmax=data_range/5)
        ax = plt.gca()
        plt.text(0.5, 0.1, 'NMSE:'+f'{nmse_f:,.3f}'+'\nSSIM:'+f'{ssim_f:,.3f}', color = 'white', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
        plt.title('recon')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(1,3,3)
        plt.imshow(reference,cmap='gray',vmax=data_range/5)
        ax = plt.gca()
        plt.title('reference')
        plt.xticks([])
        plt.yticks([])
        
        sub = sub_slc_tf[0,0,0].detach().cpu().numpy()
        slc = sub_slc_tf[0,0,1].detach().cpu().numpy()
        time_frame_no  = sub_slc_tf[0,0,2].detach().cpu().numpy()
        
        plt.suptitle('Subject:'+f'{sub}'+', Slice:'+f'{slc}'+', Time Frame:'+ f'{time_frame_no}', fontsize=14)
        plt.tight_layout()
        #breakpoint()
        plt.savefig("subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".png")
        plt.close()
        
