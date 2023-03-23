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

for sub_counter in range(number_of_subjects):
    sub = indices[0,sub_counter][0,0]
    print('subject number = '+ f'{sub}')
    
    slc_counter = 0
    for slc in indices[1,sub_counter][0]:
        print('slice number = '+ f'{slc}')

        for TF in range(5):
            time_frame_no = indices[2,sub_counter][slc_counter,TF]-1
            print('time frame = '+ f'{time_frame_no}')
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\\background004\\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            ovs_mask = slice_data['ovs_mask']
            background = slice_data['background']
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_full\\Results\\Smaps_full_005\\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            im_tgrappa = np.abs(slice_data['im_tgrappa'])
            kfull_Sfull = np.abs(slice_data['SSDU_kfull_Sfull'])+background*(ovs_mask)
            cg_kfull_Sfull = np.abs(slice_data['cg_sense'])+background*(ovs_mask)
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_full_008\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kdiff_Sfull = np.abs(slice_data['SSDU_kdiff_Sfull'])*(1-ovs_mask)+background*(ovs_mask)
            cg_kdiff_Sfull = np.abs(slice_data['cg_sense'])*(1-ovs_mask)+background*(ovs_mask)
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_mask_008\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kdiff_Smask = np.abs(slice_data['SSDU_kdiff_Smask'])*(1-ovs_mask)+background*(ovs_mask)
            cg_kdiff_Smask = np.abs(slice_data['cg_sense'])*(1-ovs_mask)+background*(ovs_mask)
            slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_diff_008\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".mat")
            kdiff_Sdiff = np.abs(slice_data['SSDU_kdiff_Sdiff'])*(1-ovs_mask)+background*(ovs_mask)
            cg_kdiff_Sdiff = np.abs(slice_data['cg_sense'])*(1-ovs_mask)+background*(ovs_mask)
            im_tgrappa_inner = im_tgrappa*(1-ovs_mask)
            
            figure = plt.figure(figsize=(10,6.5))
            vmax = np.max(im_tgrappa_inner)
            data_range = np.max(np.abs(im_tgrappa))
            Nx_min = np.sum((1-ovs_mask),1).nonzero()[0][0]
            Nx_max = np.sum((1-ovs_mask),1).nonzero()[0][-1]+1
            Nx_center = int(np.mean([Nx_min, Nx_max]))
            Ny_min = np.sum((1-ovs_mask),0).nonzero()[0][0]
            Ny_max = np.sum((1-ovs_mask),0).nonzero()[0][-1]+1
            props = dict(boxstyle='round', facecolor='black', alpha=0.8)
            
            plt.subplot(2,5,1)
            plt.imshow(cg_kfull_Sfull, cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(cg_kfull_Sfull,im_tgrappa)
            SSIM = ssim(im_tgrappa, cg_kfull_Sfull, data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.title('\n\nkspace full\nSmaps full')
            plt.ylabel('CG-SENSE', fontsize=15)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,6)
            plt.imshow(kfull_Sfull, cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(kfull_Sfull,im_tgrappa)
            SSIM = ssim(im_tgrappa, kfull_Sfull, data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.ylabel('SSDU', fontsize=15)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,2)
            plt.imshow(cg_kdiff_Sfull, cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(cg_kdiff_Sfull,im_tgrappa)
            SSIM = ssim(im_tgrappa, cg_kdiff_Sfull, data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps full')
            
            plt.subplot(2,5,7)
            plt.imshow(kdiff_Sfull, cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(kdiff_Sfull,im_tgrappa)
            SSIM = ssim(im_tgrappa, kdiff_Sfull, data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,3)
            plt.imshow(cg_kdiff_Smask, cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(cg_kdiff_Smask,im_tgrappa)
            SSIM = ssim(im_tgrappa, cg_kdiff_Smask, data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps mask')
            
            plt.subplot(2,5,8)
            plt.imshow(kdiff_Smask, cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(kdiff_Smask,im_tgrappa)
            SSIM = ssim(im_tgrappa, kdiff_Smask, data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.axis('off')
            
            plt.subplot(2,5,4)
            plt.imshow(cg_kdiff_Sdiff, cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(cg_kdiff_Sdiff,im_tgrappa)
            SSIM = ssim(im_tgrappa, cg_kdiff_Sdiff, data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            plt.title('kspace diff\nSmaps diff')
            
            plt.subplot(2,5,9)
            plt.imshow(kdiff_Sdiff, cmap='gray',vmax=vmax)
            ax = plt.gca()
            NMSE = nmse(kdiff_Sdiff,im_tgrappa)
            SSIM = ssim(im_tgrappa, kdiff_Sdiff, data_range=data_range)
            plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,5)
            plt.imshow(np.abs(im_tgrappa), cmap='gray',vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            plt.title('reference')
            
            plt.subplot(2,5,10)
            plt.imshow(np.abs(im_tgrappa), cmap='gray',vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            
            plt.suptitle('Subject:'+f'{sub}'+', Slice:'+f'{slc}'+', Time Frame:'+ f'{time_frame_no}', fontsize=14)
            plt.tight_layout()
            plt.savefig("ResultsSmallRectangularCircularMask1\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(time_frame_no)+".png")
            plt.close()
        slc_counter +=1


'''
sub = 3
slc = 4
tf = 17

slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\\background003\\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(tf)+".mat")
ovs_mask = slice_data['ovs_mask']
slice_data = loadmat("Small_MultiMaskSSDU_kspace_full\\Results\\Smaps_full_001\\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(tf)+".mat")
im_tgrappa = np.abs(slice_data['im_tgrappa'])
kfull_Sfull = np.abs(slice_data['SSDU_kfull_Sfull'])
cg_kfull_Sfull = np.abs(slice_data['cg_sense'])
im_tgrappa_inner = im_tgrappa*(1-ovs_mask)

vmax = np.max(im_tgrappa_inner)
data_range = np.max(np.abs(im_tgrappa))
Nx_min = np.sum((1-ovs_mask),1).nonzero()[0][0]
Nx_max = np.sum((1-ovs_mask),1).nonzero()[0][-1]+1
Nx_center = int(np.mean([Nx_min, Nx_max]))
Ny_min = np.sum((1-ovs_mask),0).nonzero()[0][0]
Ny_max = np.sum((1-ovs_mask),0).nonzero()[0][-1]+1
props = dict(boxstyle='round', facecolor='black', alpha=0.8)

for TF in range(tf-8,tf+8):
    print('time frame = '+ f'{TF}')
    
    slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\\background003\\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(TF)+".mat")
    background = slice_data['background']
    slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_full_003\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(TF)+".mat")
    kdiff_Sfull = np.abs(slice_data['SSDU_kdiff_Sfull'])*np.sqrt(1-ovs_mask)+background*np.sqrt(ovs_mask)
    cg_kdiff_Sfull = np.abs(slice_data['cg_sense'])*np.sqrt(1-ovs_mask)+background*np.sqrt(ovs_mask)
    slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_mask_003\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(TF)+".mat")
    kdiff_Smask = np.abs(slice_data['SSDU_kdiff_Smask'])*np.sqrt(1-ovs_mask)+background*np.sqrt(ovs_mask)
    cg_kdiff_Smask = np.abs(slice_data['cg_sense'])*np.sqrt(1-ovs_mask)+background*np.sqrt(ovs_mask)
    slice_data = loadmat("Small_MultiMaskSSDU_kspace_diff_circular_mask\Results\Smaps_diff_003\images\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(TF)+".mat")
    kdiff_Sdiff = np.abs(slice_data['SSDU_kdiff_Sdiff'])*np.sqrt(1-ovs_mask)+background*np.sqrt(ovs_mask)
    cg_kdiff_Sdiff = np.abs(slice_data['cg_sense'])*np.sqrt(1-ovs_mask)+background*np.sqrt(ovs_mask)
    
            
    figure = plt.figure(figsize=(10,6.5))
    
    plt.subplot(2,5,1)
    plt.imshow(cg_kfull_Sfull, cmap='gray',vmax=vmax)
    ax = plt.gca()
    NMSE = nmse(cg_kfull_Sfull,im_tgrappa)
    SSIM = ssim(im_tgrappa, cg_kfull_Sfull, data_range=data_range)
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
    plt.title('\n\nkspace full\nSmaps full')
    plt.ylabel('CG-SENSE', fontsize=15)
    plt.xticks([])
    plt.yticks([])
            
    plt.subplot(2,5,6)
    plt.imshow(kfull_Sfull, cmap='gray',vmax=vmax)
    ax = plt.gca()
    NMSE = nmse(kfull_Sfull,im_tgrappa)
    SSIM = ssim(im_tgrappa, kfull_Sfull, data_range=data_range)
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
    plt.ylabel('SSDU', fontsize=15)
    plt.xticks([])
    plt.yticks([])
            
    plt.subplot(2,5,2)
    plt.imshow(cg_kdiff_Sfull, cmap='gray',vmax=vmax)
    ax = plt.gca()
    NMSE = nmse(cg_kdiff_Sfull,im_tgrappa)
    SSIM = ssim(im_tgrappa, cg_kdiff_Sfull, data_range=data_range)
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
    plt.xticks([])
    plt.yticks([])
    plt.title('kspace diff\nSmaps full')
            
    plt.subplot(2,5,7)
    plt.imshow(kdiff_Sfull, cmap='gray',vmax=vmax)
    ax = plt.gca()
    NMSE = nmse(kdiff_Sfull,im_tgrappa)
    SSIM = ssim(im_tgrappa, kdiff_Sfull, data_range=data_range)
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
    plt.xticks([])
    plt.yticks([])
            
    plt.subplot(2,5,3)
    plt.imshow(cg_kdiff_Smask, cmap='gray',vmax=vmax)
    ax = plt.gca()
    NMSE = nmse(cg_kdiff_Smask,im_tgrappa)
    SSIM = ssim(im_tgrappa, cg_kdiff_Smask, data_range=data_range)
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
    plt.xticks([])
    plt.yticks([])
    plt.title('kspace diff\nSmaps mask')
            
    plt.subplot(2,5,8)
    plt.imshow(kdiff_Smask, cmap='gray',vmax=vmax)
    ax = plt.gca()
    NMSE = nmse(kdiff_Smask,im_tgrappa)
    SSIM = ssim(im_tgrappa, kdiff_Smask, data_range=data_range)
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
    plt.axis('off')
            
    plt.subplot(2,5,4)
    plt.imshow(cg_kdiff_Sdiff, cmap='gray',vmax=vmax)
    ax = plt.gca()
    NMSE = nmse(cg_kdiff_Sdiff,im_tgrappa)
    SSIM = ssim(im_tgrappa, cg_kdiff_Sdiff, data_range=data_range)
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
    plt.xticks([])
    plt.yticks([])
    plt.title('kspace diff\nSmaps diff')
            
    plt.subplot(2,5,9)
    plt.imshow(kdiff_Sdiff, cmap='gray',vmax=vmax)
    ax = plt.gca()
    NMSE = nmse(kdiff_Sdiff,im_tgrappa)
    SSIM = ssim(im_tgrappa, kdiff_Sdiff, data_range=data_range)
    plt.text(0.5, 0.1, 'NMSE:'+f'{NMSE:,.3f}'+'\nSSIM:'+f'{SSIM:,.3f}', color = 'white', 
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props)
    plt.xticks([])
    plt.yticks([])
            
    plt.subplot(2,5,5)
    plt.imshow(np.abs(im_tgrappa), cmap='gray',vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    plt.title('reference')
            
    plt.subplot(2,5,10)
    plt.imshow(np.abs(im_tgrappa), cmap='gray',vmax=vmax)
    plt.xticks([])
    plt.yticks([])
            
    plt.suptitle('Subject:'+f'{sub}'+', Slice:'+f'{slc}'+', Time Frame:'+ f'{TF}', fontsize=14)
    plt.tight_layout()
    plt.savefig("ResultsSmallCircularMask4(sub3slc4tf17)\subject_"+str(sub)+"_slice_"+str(slc)+"_TF_"+str(TF)+".png")
    plt.close()

'''