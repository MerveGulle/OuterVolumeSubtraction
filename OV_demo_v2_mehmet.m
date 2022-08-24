clear all
close all
%path(path, '/home/naxos2-raid1/omer/new_cine');
path(path, '/home/daedalus1-raid1/akcakaya-group-data/Tools/IMAGESCNTOOLBOX');


load realtime_data_small
datas = single(datas * 1e6);
datas = datas(69:end, 1:72,:,:);
%kspace_to_im = @(x) fftshift(fftshift(ifft2(x),1),2);
ifftshift2 = @(x) ifftshift(ifftshift(x,1),2); 
kspace_to_im = @(x) ifftshift2(ifft2(ifftshift2(x)));
%im_to_kspace = @(x) fft2(ifftshift(ifftshift(x,1),2));
fftshift2 = @(x) fftshift(fftshift(x,1),2); 
im_to_kspace = @(x) fftshift2(fft2(fftshift2(x)));
rssq = @(x) squeeze(sum(abs(x).^2,3)).^(1/2);
[m,n,no_c,no_dyn] =size(datas);
% datas = cat(2, datas, zeros(m,8,no_c,no_dyn)); [m,n,no_c,no_dyn] =size(datas);  % my sense code has a bug, so I added this to avoid it (lines 22-23)

% acquired data at  R = 8

im_mask = zeros(m,n, 'single'); im_mask(:,1:8:end) = 1;
im1 = rssq(kspace_to_im(datas(:,:,:,1) .* repmat(im_mask, [1 1 no_c])));
figure, imshow(im1, [0 0.8*max(im1(:))])


%composite data
kspace = zeros(m,n,no_c, no_dyn -3 ,'single');
for ind = 1:no_dyn-3, kspace(:,:,:,ind) = sum(datas(:,:,:,ind:ind+3),4); end
img = rssq(kspace_to_im(kspace));
im_composite = img(:,:,1);
figure, imshow(im_composite, [0 0.5*max(im_composite(:))])

% %mask selection
% set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
% message = sprintf('Left click and hold to begin drawing.\nSimply lift the mouse button to finish');
% uiwait(msgbox(message));
% hFH = imfreehand();
% % Create a binary image ("mask") from the ROI object.
% outerROI = hFH.createMask();
% figure, imshow(outerROI, [])
% 
% figure, imshow(im_composite, [0 0.5*max(im_composite(:))])
% set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
% message = sprintf('Left click and hold to begin drawing.\nSimply lift the mouse button to finish');
% uiwait(msgbox(message));
% hFH = imfreehand();
% innerROI = hFH.createMask();
% figure, imshow(innerROI, [])
% 
% figure, imshow(im_composite, [0 0.5*max(im_composite(:))])
% set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
% message = sprintf('Left click and hold to begin drawing.\nSimply lift the mouse button to finish');
% uiwait(msgbox(message));
% hFH = imfreehand();
% sideROI = hFH.createMask();
% figure, imshow(sideROI, [])
% 
% total_mask = (outerROI & not(innerROI)) | sideROI;

% load total_mask_dyn1

% subtract out the outer volume

% figure, imshow(im_composite - im_composite .* total_mask2, [0 0.5*max(im_composite(:))])


total_mask2 = ones(m,n);
total_mask2(:, 26:55) = 0;
% aa = tukeywin(128-56, .1);
% bb = zeros(1, 128); bb(1:40) = aa(end-39:end); bb(97:end) = aa(1:end-40);
% total_mask2 = total_mask2 .* repmat(bb, [m 1]);

% subtract these out from the data
k1 = kspace(:,:,:,1);
k1_background = im_to_kspace(repmat(total_mask2,[1 1 no_c]) .* kspace_to_im(k1));

%subtract out background
datas_R8 = datas(:,:,:,1) .* repmat(im_mask, [1 1 no_c]);
k_diff = datas_R8(:,:,:,1)  - k1_background .* (abs(datas_R8(:,:,:,1))> 0);
figure, imshow(rssq(kspace_to_im(k_diff)), [])
figure, imshow(rssq(kspace_to_im(datas_R8(:,:,:,1))), [])


%generate coils
path(path, '/home/range1-raid1/akcakaya/sense_recon'); 
k_low = zeros(size(k1), 'single');
ACS_size = 32;
k_low(160-69-23:160-69+24,49-ACS_size/2:49+ACS_size/2-1,:) = k1(160-69-23:160-69+24,49-ACS_size/2:49+ACS_size/2-1,:) .* repmat(tukeywin(48,.9) * tukeywin(ACS_size,.9)', [1 1 no_c]);
% k_low = (k1) .* repmat(tukeywin(m,.9) * tukeywin(n,.9)', [1 1 no_c]);
img_low = kspace_to_im(k_low);
Coils = img_low ./ repmat(rssq(img_low + eps), [1 1 no_c]);
Coils_mask = Coils .* repmat(1-total_mask2, [1 1 no_c]); 


k_low_diff = zeros(size(k1), 'single');
k_low_diff(160-69-23:160-69+24,49-ACS_size/2:49+ACS_size/2-1,:) = (k1(160-69-23:160-69+24,49-ACS_size/2:49+ACS_size/2-1,:) - k1_background(160-69-23:160-69+24,49-ACS_size/2:49+ACS_size/2-1,:)) .* repmat(tukeywin(48,.9) * tukeywin(ACS_size,.9)', [1 1 no_c]);
% k_low_diff = (k1 - k1_background) .* repmat(tukeywin(m,.9) * tukeywin(n,.9)', [1 1 no_c]);
img_low_diff = kspace_to_im(k_low_diff);
Coils_diff = img_low_diff ./ repmat(rssq(img_low_diff + eps), [1 1 no_c]);


coils_to_show = [ 5,8,9,11,14,21,26,28];

imagescn(cat(3, abs(Coils(:,:,coils_to_show)), abs(Coils_mask(:,:,coils_to_show))), [], [2 length(coils_to_show)])
imagescn(cat(3, abs(Coils(:,:,coils_to_show)), abs(Coils_diff(:,:,coils_to_show))), [], [2 length(coils_to_show)])


%no OVS processing
acc_rate = 8;
k_space_undersampled = datas_R8(:,1:acc_rate:end,:);
A = @(x) system_matrix_forward(x, Coils, acc_rate);        
AT = @(x) system_matrix_backward(x, Coils);
F = @(x) AT(A(x)); b = AT(k_space_undersampled(:));
cg_sense = reshape(cg_solve(b, F, 5*acc_rate, 1e-4),size(Coils,1),size(Coils,2)); % no OVS processing
sense = sense_recon(k_space_undersampled, Coils);

kspace_aliased =permute(k_space_undersampled ,[2 1 3]);
ACS_grap = permute(k_low , [2 1 3]);
kspace_grappa_recon_nocenter = k_diff;
kspace_recon = grappa_new(kspace_aliased,ACS_grap,acc_rate, [],5,4);
kspace_grappa_recon_nocenter(:, 1:size(kspace_recon,1), :) = permute(kspace_recon, [2 1 3]);
im_grap = rssq(kspace_to_im(kspace_grappa_recon_nocenter));


%OVS from k-space
k_space_undersampled = k_diff(:,1:acc_rate:end,:);
A = @(x) system_matrix_forward(x, Coils, acc_rate);        
AT = @(x) system_matrix_backward(x, Coils);
F = @(x) AT(A(x)); b = AT(k_space_undersampled(:));
cg_sense_OVS = reshape(cg_solve(b, F, 5*acc_rate, 1e-4),size(Coils,1),size(Coils,2)); % outer volume subtracted from k-space
sense_OVS = sense_recon(k_space_undersampled, Coils);

kspace_aliased = permute(k_space_undersampled ,[2 1 3]);
ACS_grap = permute(k_low , [2 1 3]);
kspace_grappa_recon_nocenter = k_diff;
kspace_recon = grappa_new(kspace_aliased,ACS_grap,acc_rate, [],5,4);
kspace_grappa_recon_nocenter(:, 1:size(kspace_recon,1), :) = permute(kspace_recon, [2 1 3]);
im_grap_OVS = rssq(kspace_to_im(kspace_grappa_recon_nocenter));

%OVS from k-space and calibration in image space
k_space_undersampled = k_diff(:,1:acc_rate:end,:);
A = @(x) system_matrix_forward(x, Coils_mask, acc_rate);        
AT = @(x) system_matrix_backward(x, Coils_mask);
F = @(x) AT(A(x)); b = AT(k_space_undersampled(:));
% F = @(x) AT(A(x)) + 1e-4 *x; b = AT(k_space_undersampled(:));
cg_sense_mask = reshape(cg_solve(b, F, 5*acc_rate, 1e-4),size(Coils,1),size(Coils,2)); % outer volume subtracted from k-space and calibration
sense_mask = sense_recon(k_space_undersampled, Coils_mask);

kspace_aliased = permute(k_space_undersampled ,[2 1 3]);
ACS_grap = permute(k_low_diff, [2 1 3]);
kspace_grappa_recon_nocenter = k_diff;
kspace_recon = grappa_new(kspace_aliased,ACS_grap,acc_rate, [],5,4);
kspace_grappa_recon_nocenter(:, 1:size(kspace_recon,1), :) = permute(kspace_recon, [2 1 3]);
im_grap_mask = rssq(kspace_to_im(kspace_grappa_recon_nocenter));


%OVS from k-space and calibration in k-space  (same for grappa)
A = @(x) system_matrix_forward(x, Coils_diff, acc_rate);        
AT = @(x) system_matrix_backward(x, Coils_diff);
F = @(x) AT(A(x)); b = AT(k_space_undersampled(:));
% F = @(x) AT(A(x)) + 1e-4 *x; b = AT(k_space_undersampled(:));
cg_sense_diff = reshape(cg_solve(b, F, 5*acc_rate, 1e-4),size(Coils,1),size(Coils,2));
sense_diff = sense_recon(k_space_undersampled, Coils_diff);

kspace_aliased = permute(k_space_undersampled ,[2 1 3]);
ACS_grap = permute(k_low_diff, [2 1 3]);
kspace_grappa_recon_nocenter = k_diff;
kspace_recon = grappa_new(kspace_aliased,ACS_grap,acc_rate, [],5,4);
kspace_grappa_recon_nocenter(:, 1:size(kspace_recon,1), :) = permute(kspace_recon, [2 1 3]);
im_grap_diff = rssq(kspace_to_im(kspace_grappa_recon_nocenter));

imagescn(abs(cat(3, cg_sense, cg_sense_OVS, cg_sense_mask, cg_sense_diff)), [], [1 4])
imagescn(abs(cat(3, sense, sense_OVS, sense_mask, sense_diff)), [], [1 4])
imagescn(abs(cat(3, im_grap, im_grap_OVS, im_grap_mask, im_grap_diff)), [], [1 4])

imagescn(abs(cat(3, cg_sense, cg_sense_OVS +im_composite .* total_mask2, cg_sense_mask + im_composite .* total_mask2, cg_sense_diff + im_composite .* total_mask2)), [], [1 4])
imagescn(abs(cat(3, cg_sense, cg_sense_OVS +im_composite .* total_mask2, cg_sense_mask + im_composite .* total_mask2)), [0 0.5*max(im_composite(:))], [1 3])