import numpy as np

def im_to_kspace (image, axis=[0,1]):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image, axes=axis), axes=axis, norm='ortho'), axes=axis)

def kspace_to_im (kspace, axis=[0,1]):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace, axes=axis), axes=axis, norm='ortho'), axes=axis)

def rssq (x):
    return np.sqrt(np.sum(np.abs(x)**2,2))

def A(x, Smaps, mask):
    return im_to_kspace(x[...,None]*Smaps)*mask[...,None]

def AT(x, Smaps):
    return np.sum(kspace_to_im(x)*np.conj(Smaps), 2)

def cgsense(kspace,Smaps,mask,max_iter=30):
    a = AT(kspace,Smaps)
    p = np.copy(a)
    r_now = np.copy(a)
    xn = np.zeros_like(a)
    for i in np.arange(max_iter):
        delta = np.sum(r_now*np.conj(r_now))/np.sum(a*np.conj(a))
        if delta < 1e-5:
            break
        # q = (EHE)p
        q = AT(A(p,Smaps,mask),Smaps)
        # rr_pq = r'r/p'q
        rr_pq = np.sum(r_now*np.conj(r_now))/np.sum(q*np.conj(p))
        xn = xn + rr_pq * p
        r_next = r_now - rr_pq * q
        # p = r_next + r_next'r_next/r_now'r_now
        p = r_next + (np.sum(r_next*np.conj(r_next))/np.sum(r_now*np.conj(r_now))) * p
        r_now = np.copy(r_next)
    return xn


def ADMM(kspace, Smaps, mask, lmbda , mu=1e-2, max_iter=20,cg_max_iter=10):
    # initialization
    xj = rssq(kspace_to_im(kspace))
    SHFHd = AT(kspace, Smaps)
    nu1j = np.zeros_like(xj)
    for j in np.arange(max_iter):
        # update u1(j) - sparsity term
        u1j = np.divide((xj+nu1j), np.abs(xj+nu1j),where=(np.abs(xj+nu1j)!=0)) * np.maximum(np.abs(xj+nu1j)-lmbda/mu,0)
        # update x(j) = data consistency term
        a = SHFHd + mu*(u1j-nu1j)
        p = np.copy(a)
        r_now = np.copy(a)
        xn = np.zeros_like(a)
        for i in np.arange(cg_max_iter):
            # q = (SHFHFS+muI)p
            q = AT(A(p,Smaps,mask),Smaps) + mu*p
            # rr_pq = r'r/p'q
            rr_pq = np.sum(r_now*np.conj(r_now))/np.sum(q*np.conj(p))
            xn = xn + rr_pq * p
            r_next = r_now - rr_pq * q
            # p = r_next + r_next'r_next/r_now'r_now
            p = r_next + (np.sum(r_next*np.conj(r_next))/np.sum(r_now*np.conj(r_now))) * p
            r_now = np.copy(r_next)
        xj = xn
        # update nu1(j)
        nu1j = nu1j - (u1j - xj)
    return xj


def gfactor(Smaps, R):
    Nx, Ny, Nc = Smaps.shape
    gmap = np.zeros((Nx,Ny), dtype=complex)
    for x in np.arange(Nx):
        for y in np.arange(Ny):
            C = np.zeros((1,R,Nc), dtype=complex)
            ind_mat = np.zeros((1,8), dtype=complex)
            for Cy in np.arange(R):
                cur_x = x
                cur_y = np.mod(y+(Cy*Ny)//R, Ny)
                C[0,Cy,:] = Smaps[cur_x, cur_y, :]
                ind_mat[0,Cy] = x * Ny + y
            C = C.reshape(R,Nc)
            zero_col = np.argwhere(np.all(C[...,:]!=0, axis=0))
            np.delete(C, zero_col, axis=1)
            if C.shape[1]==0:
                gmap[x,y] = 0
            else:
                CtC = np.matmul(np.conj(C).T,C)
                CtCi = np.linalg.pinv(CtC)
                gmap[x,y] = np.sqrt(CtC[0,0]*CtCi[0,0])
    return gmap


def gfactor_MC(Smaps, R=8):
    Nx, Ny, Nc = Smaps.shape
    N = 100
    mask = np.zeros((Nx,Ny))
    mask[:,::R] = 1
    image0 = np.zeros((Nx,Ny,N), dtype=np.complex)
    image = np.zeros((Nx,Ny,N), dtype=np.complex)
    for n in np.arange(N):
        image[:,:,n] = np.random.randn(Nx,Ny) + 1j*np.random.randn(Nx,Ny)
        kspace = A(image[:,:,n], Smaps, mask)
        image0[:,:,n] = AT(kspace*mask[:,:,None], Smaps)
    gmap = np.std(image0.real,axis=2)/np.std(image.real,axis=2)/np.sqrt(R)
    return gmap

"""
def gfactor_MC(kspace, Smaps, R):
    N = 100
    Nx,Ny,Nc = Smaps.shape
    mask = np.zeros((Nx,Ny), dtype=complex)
    mask[:,::R] = 1

    recons = np.zeros((Nx,Ny,N), dtype=complex)
    for i in range(N):
        ksp_noise = np.random.randn(*kspace.shape) + 1j*np.random.randn(*kspace.shape)
        ksp_noise = mask[...,None] * (ksp_noise + kspace)
        recons[:,:,i] = AT(ksp_noise, Smaps)
    
    recons_noise = np.zeros((Nx,Ny,N), dtype=np.complex)  
    for i in range(N):
        ksp_noise = np.random.randn(*kspace.shape) + 1j*np.random.randn(*kspace.shape)
        recons_noise[:,:,i] = AT(ksp_noise, Smaps)
    
    recons_std = np.std(recons.real, axis=2)
    recon_noise_std = np.std(recons_noise, axis=2)

    gfactor = np.divide(recons_std, recon_noise_std, where=abs(recons[:,:,0].squeeze()) != 0) / np.sqrt(R)

    return gfactor
"""


def sense(kspace, Smaps, R=8):
    Nx, Ny, Nc = Smaps.shape
    kspace = kspace[:,np.abs(kspace[92,:,0])!=0,:]
    x0 = kspace_to_im(kspace)
    image = np.zeros((Nx, Ny), dtype=complex)
    Cy = Ny//R
    for xx in np.arange(Nx):
        for yy in np.arange(Cy):
            S = Smaps[xx,yy::Cy,:].T
            invS = np.matmul(np.linalg.inv(np.matmul(np.conj(S).T,S) + 1e-2*np.identity(S.shape[1])), np.conj(S).T)
            image[xx,yy::Cy] = np.matmul(invS, x0[xx,np.mod(yy+Cy//2,Cy),:])
    return image


def kernel_cal(data_calib, R=4, Kx=5):
    Cx, Cy, Nc = data_calib.shape
    kernel = np.zeros((R-1, Nc, Kx*2*Nc), dtype=complex)
    ACS = np.zeros(((Cx-Kx+1)*(Cy-R), Kx*2*Nc), dtype=complex)
    for x in np.arange(Cx-Kx+1):
        for y in np.arange(Cy-R):
            ACS[x*(Cy-R)+y] = data_calib[x:x+Kx,[y,y+R],:].reshape(1,-1)
    iACS = np.linalg.pinv(ACS)
    for y in np.arange(R-1):
        for c in np.arange(Nc):
            kernel[y,c] = np.matmul(iACS,data_calib[(Kx-1)//2:-(Kx-1)//2,y+1:Cy-R+y+1,c].reshape(-1,1)).reshape(1,1,-1)
    return kernel


def grappa(kspace, kernel, time_frame, R=4, Kx=5):
    Nx, Ny, Nc = kspace.shape
    shift = np.mod(time_frame, R)
    Nl = np.mod(Ny-shift-1, R)
    ACS = np.zeros(((Nx-Kx+1)*(Ny//R-1), Kx*2*Nc), dtype=complex)
    for x in np.arange(Nx-Kx+1):
        for y in np.arange(Ny//R-1):
            ACS[x*(Ny//R-1)+y] = kspace[x:x+Kx,[y*R+shift,y*R+R+shift],:].reshape(1,-1)
    kspace_new = np.copy(kspace)
    for y in np.arange(R-1):
        for c in np.arange(Nc):
            kspace_new[(Kx-1)//2:-(Kx-1)//2,y+1+shift:Ny-Nl-R+y+1:R,c] = np.matmul(ACS,kernel[y,c].reshape(-1,1)).reshape(Nx-Kx+1,-1)
    image = rssq(kspace_to_im(kspace_new))
    return image


