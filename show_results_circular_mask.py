from scipy.io import loadmat
import numpy as np
from skimage.metrics import structural_similarity as ssim
from SupportingFunctions import nmse
import matplotlib.pyplot as plt


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
            kfull_Sfull = np.abs(slice_data['SSDU_kfull_Sfull'])
            cg_kfull_Sfull = np.abs(slice_data['cg_sense'])
            slice_data = loadmat("C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_full_002\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            ovs_mask = slice_data['ovs_mask']
            background = slice_data['background']
            kfull_Sfull = kfull_Sfull*(1-ovs_mask)
            cg_kfull_Sfull = cg_kfull_Sfull*(1-ovs_mask)
            kdiff_Sfull = np.abs(slice_data['SSDU_kdiff_Sfull'])
            cg_kdiff_Sfull = np.abs(slice_data['cg_sense'])
            slice_data = loadmat("C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_mask_002\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kdiff_Smask = np.abs(slice_data['SSDU_kdiff_Smask'])
            cg_kdiff_Smask = np.abs(slice_data['cg_sense'])
            slice_data = loadmat("C:\Codes\p006_OVS\OVS\MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_diff_002\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kdiff_Sdiff = np.abs(slice_data['SSDU_kdiff_Sdiff'])
            cg_kdiff_Sdiff = np.abs(slice_data['cg_sense'])
            im_tgrappa_inner = np.abs(im_tgrappa*(1-ovs_mask))
            breakpoint()
            figure = plt.figure(figsize=(10,7))
            vmax = np.max(im_tgrappa_inner)
            data_range = np.max(im_tgrappa_inner)
            Nx_min = np.sum((1-ovs_mask),1).nonzero()[0][0]
            Nx_max = np.sum((1-ovs_mask),1).nonzero()[0][-1]
            Nx_center = int(np.mean([Nx_min, Nx_max]))
            Ny_min = np.sum((1-ovs_mask),0).nonzero()[0][0]
            Ny_max = np.sum((1-ovs_mask),0).nonzero()[0][-1]
            props = dict(boxstyle='round', facecolor='black', alpha=0.8)
            
            plt.subplot(2,5,1)
            plt.imshow(cg_kfull_Sfull[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(cg_kfull_Sfull[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], cg_kfull_Sfull[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.title('\n\nkspace full\nSmaps full')
            plt.ylabel('CG-SENSE', fontsize=15)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,6)
            plt.imshow(kfull_Sfull[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(kfull_Sfull[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], kfull_Sfull[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.ylabel('SSDU', fontsize=15)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,2)
            plt.imshow(cg_kdiff_Sfull[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(cg_kdiff_Sfull[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], cg_kdiff_Sfull[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps full')
            
            plt.subplot(2,5,7)
            plt.imshow(kdiff_Sfull[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(kdiff_Sfull[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], kdiff_Sfull[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,3)
            plt.imshow(cg_kdiff_Smask[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(cg_kdiff_Smask[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], cg_kdiff_Smask[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps mask')
            
            plt.subplot(2,5,8)
            plt.imshow(kdiff_Smask[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(kdiff_Smask[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], kdiff_Smask[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.axis('off')
            
            plt.subplot(2,5,4)
            plt.imshow(cg_kdiff_Sdiff[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(cg_kdiff_Sdiff[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], cg_kdiff_Sdiff[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps diff')
            
            plt.subplot(2,5,9)
            plt.imshow(kdiff_Sdiff[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(kdiff_Sdiff[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], kdiff_Sdiff[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,5)
            plt.imshow(np.abs(im_tgrappa_inner)[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            plt.title('reference')
            
            plt.subplot(2,5,10)
            plt.imshow(np.abs(im_tgrappa_inner)[Nx_center-40:Nx_center+60], cmap='gray',vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            
            plt.suptitle('Subject:'+f'{sub}'+', Slice:'+f'{slc}'+', Time Frame:'+ f'{time_frame_no}', fontsize=14)
            plt.tight_layout()
            plt.savefig("C:\Codes\p006_OVS\OVS\ResultsCircularNarrowerMask\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".png")
            plt.close()
            
        slc_counter +=1