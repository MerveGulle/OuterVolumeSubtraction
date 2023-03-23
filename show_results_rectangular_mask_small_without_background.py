from scipy.io import loadmat
import numpy as np
from skimage.metrics import structural_similarity as ssim
from SupportingFunctions import nmse
import matplotlib.pyplot as plt
import os

os.chdir('C:\\Codes\\p006_OVS\\OVS')
data_indices = loadmat("test_data_indices.mat")
_, indices = list(data_indices.items())[3]
# indices.shape = (3,2)
# indices[0]: subject numbers
# indices[1]: slice numbers
# indices[2]: time frame numbers (number of slices x 5)

number_of_subjects = indices.shape[1]
NMSE_array = np.empty((4,0))
SSIM_array = np.empty((4,0))

for sub_counter in range(number_of_subjects):
    sub = indices[0,sub_counter][0,0]
    print('subject number = '+ f'{sub}')
    
    slc_counter = 0
    for slc in indices[1,sub_counter][0]:
        print('slice number = '+ f'{slc}')

        for TF in range(5):
            time_frame_no = indices[2,sub_counter][slc_counter,TF]-1
            print('time frame = '+ f'{time_frame_no}')
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_rectangular_mask\Results\\background003\\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            ovs_mask = slice_data['ovs_mask']
            background = slice_data['background']
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\\background004\\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            ovs_mask_circle = slice_data['ovs_mask']
            slice_data = loadmat("TestDatasetSmallRectangularMaskNew\\subject_"+str(sub)+"_slice_"+str(slc)+"_"+str(TF+1)+".mat")
            im_tgrappa = np.abs(slice_data['im_tgrappa'])
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_full\\Results\\Smaps_full_005\\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kfull_Sfull = np.abs(slice_data['SSDU_kfull_Sfull'])*(1-ovs_mask)
            cg_kfull_Sfull = np.abs(slice_data['cg_sense'])*(1-ovs_mask)
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_rectangular_mask\Results\Smaps_full_006\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kdiff_Sfull = np.abs(slice_data['SSDU_kdiff_Sfull'])*(1-ovs_mask)
            cg_kdiff_Sfull = np.abs(slice_data['cg_sense'])*(1-ovs_mask)
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_rectangular_mask\Results\Smaps_mask_006\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kdiff_Smask = np.abs(slice_data['SSDU_kdiff_Smask'])*(1-ovs_mask)
            cg_kdiff_Smask = np.abs(slice_data['cg_sense'])*(1-ovs_mask)
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_rectangular_mask\Results\Smaps_diff_006\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kdiff_Sdiff = np.abs(slice_data['SSDU_kdiff_Sdiff'])*(1-ovs_mask)
            cg_kdiff_Sdiff = np.abs(slice_data['cg_sense'])*(1-ovs_mask)
            im_tgrappa_inner = im_tgrappa*(1-ovs_mask)
            
            figure = plt.figure(figsize=(10,5.8))
            vmax = np.max(im_tgrappa_inner*(1-ovs_mask_circle))
            data_range = np.max(np.abs(im_tgrappa_inner)*np.sqrt(1-ovs_mask_circle))
            Nx_min = np.sum((1-ovs_mask_circle),1).nonzero()[0][0]
            Nx_max = np.sum((1-ovs_mask_circle),1).nonzero()[0][-1]+1
            Nx_center = int(np.mean([Nx_min, Nx_max]))
            Ny_min = np.sum((1-ovs_mask),0).nonzero()[0][0]
            Ny_max = np.sum((1-ovs_mask),0).nonzero()[0][-1]+1
            props = dict(boxstyle='round', facecolor='black', alpha=0.8)
            
            plt.subplot(2,5,1)
            plt.imshow(cg_kfull_Sfull[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse((cg_kfull_Sfull*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim((im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (cg_kfull_Sfull*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.title('\n\nkspace full\nSmaps full')
            plt.ylabel('CG-SENSE', fontsize=15)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,6)
            plt.imshow(kfull_Sfull[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE1 = nmse((kfull_Sfull*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM1 = ssim((im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (kfull_Sfull*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE1:,.3f}'+'\nSSIM:'+f'{SSIM1:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.ylabel('SSDU', fontsize=15)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,2)
            plt.imshow(cg_kdiff_Sfull[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse((cg_kdiff_Sfull*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim((im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (cg_kdiff_Sfull*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps full')
            
            plt.subplot(2,5,7)
            plt.imshow(kdiff_Sfull[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE2 = nmse((kdiff_Sfull*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM2 = ssim((im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (kdiff_Sfull*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE2:,.3f}'+'\nSSIM:'+f'{SSIM2:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,3)
            plt.imshow(cg_kdiff_Smask[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse((cg_kdiff_Smask*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim((im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (cg_kdiff_Smask*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps mask')
            
            plt.subplot(2,5,8)
            plt.imshow(kdiff_Smask[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE3 = nmse((kdiff_Smask*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM3 = ssim((im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (kdiff_Smask*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE3:,.3f}'+'\nSSIM:'+f'{SSIM3:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.axis('off')
            
            plt.subplot(2,5,4)
            plt.imshow(cg_kdiff_Sdiff[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse((cg_kdiff_Sdiff*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM = ssim((im_tgrappa_inner*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], (cg_kdiff_Sdiff*np.sqrt(1-ovs_mask_circle))[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps diff')
            
            plt.subplot(2,5,9)
            plt.imshow(kdiff_Sdiff[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE4 = nmse(kdiff_Sdiff[Nx_min:Nx_max, Ny_min:Ny_max],im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max])
            SSIM4 = ssim(im_tgrappa_inner[Nx_min:Nx_max, Ny_min:Ny_max], kdiff_Sdiff[Nx_min:Nx_max, Ny_min:Ny_max], data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE4:,.3f}'+'\nSSIM:'+f'{SSIM4:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,5)
            plt.imshow(np.abs(im_tgrappa_inner)[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            plt.title('reference')
            
            plt.subplot(2,5,10)
            plt.imshow(np.abs(im_tgrappa_inner)[Nx_center-30:Nx_center+50], cmap='gray',vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            
            plt.suptitle('Subject:'+f'{sub}'+', Slice:'+f'{slc}'+', Time Frame:'+ f'{time_frame_no}', fontsize=14)
            plt.tight_layout()
            plt.savefig("ResultsSmallRectangularMaskWithoutBackground5\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".png")
            plt.close()
            
            NMSE_array = np.append(NMSE_array,[[NMSE1],[NMSE2],[NMSE3],[NMSE4]],axis=1)
            SSIM_array = np.append(SSIM_array,[[SSIM1],[SSIM2],[SSIM3],[SSIM4]],axis=1)
        slc_counter +=1
'''       
figure = plt.figure()
N_slc = NMSE_array.shape[1]
n_array = np.arange(1,N_slc+1)
plt.bar(n_array,NMSE_array[0],color = 'red',width=0.2)
plt.bar(n_array+0.25,NMSE_array[1],color = 'blue',width=0.2)
plt.bar(n_array+0.5,NMSE_array[2],color = 'green',width=0.2)
plt.bar(n_array+0.75,NMSE_array[3],color = 'yellow',width=0.2)
# plt.legend(['kfullSfull','kdiffSfull','kdiffSmask','kdiffSdiff'])
plt.title('NMSE')
plt.tight_layout()
plt.grid()
plt.xlim(0.75,N_slc+1)

figure = plt.figure()
N_slc = NMSE_array.shape[1]
n_array = np.arange(1,N_slc+1)
plt.bar(n_array,np.min(NMSE_array,axis=0),color = 'red',width=0.2)
# plt.legend(['kfullSfull','kdiffSfull','kdiffSmask','kdiffSdiff'])
plt.title('NMSE')
plt.tight_layout()
plt.grid()
plt.xlim(0.75,N_slc+0.25)

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
figure = plt.figure()
N_slc = SSIM_array.shape[1]
n_array = np.arange(1,N_slc+1)
plt.plot(n_array,np.max(SSIM_array,axis=0),'r*')
plt.plot(n_array,np.max(SSIM_array_circle,axis=0),'b*')
plt.legend(['rectangle','circle'])
# plt.legend(['kfullSfull','kdiffSfull','kdiffSmask','kdiffSdiff'])
plt.title('SSIM')
plt.tight_layout()
ax = plt.gca()
ax.grid(which='minor', linestyle=':')
ax.grid(which='major')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(10))
plt.xlim(0,N_slc+1)
plt.ylim(0.65,1.0)
'''