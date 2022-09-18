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

def cgsense(kspace,Smaps,mask,max_iter=50):
    a = AT(kspace,Smaps)
    p = np.copy(a)
    r_now = np.copy(a)
    xn = np.zeros_like(a)
    for i in np.arange(max_iter):
        delta = np.sum(r_now*np.conj(r_now))/np.sum(a*np.conj(a))
        if delta > 1e-4:
            # q = (EHE)p
            q = AT(A(p,Smaps,mask),Smaps)
            # rr_pq = r'r/p'q
            rr_pq = np.sum(r_now*np.conj(r_now))/np.sum(q*np.conj(p))
            xn = xn + rr_pq * p
            r_next = r_now - rr_pq * q
            # p = r_next + r_next'r_next/r_now'r_now
            p = r_next + (np.sum(r_next*np.conj(r_next))/np.sum(r_now*np.conj(r_now))) * p
            r_now = np.copy(r_next)
        else:
            break
    return xn


def ADMM(kspace,Smaps,mask,mu=1e-2,lmbda=1,max_iter=20,cg_max_iter=8):
    # initialization
    xj = rssq(kspace_to_im(kspace))
    SHFHd = AT(kspace, Smaps)
    nu1j = np.zeros_like(xj)
    for j in np.arange(max_iter):
        # update u1(j)
        u1j = ((xj+nu1j)/(np.abs(xj+nu1j)+1e-30)) * np.maximum(np.abs(xj+nu1j)-lmbda/mu,0)
        # update x(j)
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


def gfactor_MC(Smaps, R):
    Nx, Ny, Nc = Smaps.shape
    N = 200
    mask = np.zeros((Nx,Ny))
    mask[:,::R] = Ny/((Ny//R)+1)
    image = np.random.randn(Nx,Ny,2*N).view(complex)
    image0 = np.zeros_like(image)
    for n in np.arange(N):
        kspace = A(image[:,:,n], Smaps, mask)
        image0[:,:,n] = AT(kspace*mask[:,:,None], Smaps)
    gmap = np.std(image0,axis=2)/np.std(image,axis=2)/np.sqrt(R)
    return gmap


def sense(Smaps, kspace, R):
    Nx, Ny, Nc = Smaps.shape
    kspace = kspace[:,::8]
    x0 = kspace_to_im(kspace)
    image = np.zeros((Nx, Ny), dtype=complex)
    Cy = Ny//R + 1 * (np.mod(Ny,R)!=0)
    for xx in np.arange(Nx):
        for yy in np.arange(Cy):
            S = Smaps[xx,yy::Cy,:].T
            image[xx,yy::Cy] = np.matmul(np.linalg.pinv(S), x0[xx,yy,:])
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
    ACS = np.zeros(((Nx-Kx+1)*((Ny-R)//R), Kx*2*Nc), dtype=complex)
    for x in np.arange(Nx-Kx+1):
        for y in np.arange((Ny-R)//R):
            ACS[x*((Ny-R)//R)+y] = kspace[x:x+Kx,[y*R+shift,y*R+R+shift],:].reshape(1,-1)
    kspace_new = kspace
    for y in np.arange(R-1):
        for c in np.arange(Nc):
            kspace_new[(Kx-1)//2:-(Kx-1)//2,y+1+shift:Ny-Nl-R+y+1:R,c] = np.matmul(ACS,kernel[y,c].reshape(-1,1)).reshape(Nx-Kx+1,-1)
    image = rssq(kspace_to_im(kspace_new))
    return image



