import torch
import SupportingFunctions as sf
import model
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.io import savemat


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
test_data_path = "C:\Codes\p006_OVS\OVS\TestDataset"
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
denoiser.load_state_dict(torch.load('OVS_multimaskSSDU_100.pt'))
denoiser.eval()

for i, (x0, composite_kspace, sense_maps, acc_mask, im_tgrappa, sub_slc_tf, index) in enumerate(test_loader['test_loader']):
    with torch.no_grad():
        # Forward pass
        x0                     = x0[0].to(device)                       # [2,Nx,Ny]
        composite_kspace       = composite_kspace[0].to(device)         # [1,Nx,Ny,Nc]
        sense_maps             = sense_maps[0].to(device)                # [2,Nx,Ny,Nc]
        acc_mask               = acc_mask[0].to(device)                 # [Nx,Ny]
        im_tgrappa             = im_tgrappa[0].to(device)
        # Forward pass
        xt = torch.clone(x0)
        for t in range(params['T']):
            L, zt = denoiser(xt[None,...])
            xt = model.DC_layer(x0,zt[0],L,sense_maps,acc_mask)
        
        #zerofilled = x0[0].detach().cpu().numpy()
        #cg_sense = sf.cgsense(x0,composite_kspace,sense_maps,acc_mask)[0].detach().cpu().numpy()
        SSDU = xt[0].detach().cpu().numpy()
        im_tgrappa = im_tgrappa.detach().cpu().numpy()
        '''
        data_range = np.abs(im_tgrappa).max() - np.abs(im_tgrappa).min()
        vmax = np.abs(SSDU).max() * 0.3
        figure = plt.figure(figsize=(9,7))
        plt.subplot(1,4,1)
        plt.imshow(np.abs(zerofilled), cmap='gray', vmax=vmax/4)
        ax = plt.gca()
        NMSE = sf.nmse(zerofilled,im_tgrappa)
        SSIM = ssim(np.abs(im_tgrappa), np.abs(zerofilled), data_range=data_range)
        plt.text(0.5, 0.05, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.axis('off')
        plt.title('Zerofilled')
        
        plt.subplot(1,4,2)
        plt.imshow(np.abs(cg_sense), cmap='gray', vmax=vmax)
        ax = plt.gca()
        NMSE = sf.nmse(cg_sense,im_tgrappa)
        SSIM = ssim(np.abs(im_tgrappa), np.abs(cg_sense), data_range=data_range)
        plt.text(0.5, 0.05, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.axis('off')
        plt.title('CG-SENSE')
        
        plt.subplot(1,4,3)
        plt.imshow(np.abs(SSDU), cmap='gray', vmax=vmax)
        ax = plt.gca()
        NMSE = sf.nmse(SSDU,im_tgrappa)
        SSIM = ssim(np.abs(im_tgrappa), np.abs(SSDU), data_range=data_range)
        plt.text(0.5, 0.05, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.axis('off')
        plt.title('SSDU')
        
        plt.subplot(1,4,4)
        plt.imshow(np.abs(im_tgrappa), cmap='gray', vmax=vmax)
        ax = plt.gca()
        plt.text(0.5, 0.05, 'im_tgrappa', color = 'white', 
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.axis('off')
        plt.title('Reference')

        sub = sub_slc_tf[0,0,0].detach().cpu().numpy()
        slc = sub_slc_tf[0,0,1].detach().cpu().numpy()
        tf  = sub_slc_tf[0,0,2].detach().cpu().numpy()
        plt.suptitle('Subject:'+f'{sub}'+', Slice:'+f'{slc}'+', Time Frame:'+ f'{tf}', fontsize=14)
        
        # plt.savefig("C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_full\Results\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(tf)+".png")
        # plt.close()
        '''
        sub = sub_slc_tf[0,0,0].detach().cpu().numpy()
        slc = sub_slc_tf[0,0,1].detach().cpu().numpy()
        tf  = sub_slc_tf[0,0,2].detach().cpu().numpy()
        
        filename = "C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_full\Results\Smaps_full_001\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(tf)+".mat"

        datadir = {"im_tgrappa": im_tgrappa, 
                   "SSDU_kfull_Sfull": SSDU}
        savemat(filename, datadir) 


