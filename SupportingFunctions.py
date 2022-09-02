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

def cgsense(kspace,Smaps,mask,max_iter=20):
    a = AT(kspace,Smaps)
    p = np.copy(a)
    r_now = np.copy(a)
    xn = np.zeros_like(a)
    for i in np.arange(max_iter):
        delta = np.sum(r_now*np.conj(r_now))/np.sum(a*np.conj(a))
        if delta<1e-6:
            break
        else:
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