import torch
import SupportingFunctions as sf

# x0      : initial solution [2 Nx Ny]
# zn      : Output of nth denoiser block [2 Nx Ny]
# L       : regularization coefficient=quadratic relaxation parameter
# S       : sensitivity maps [2 Nx Ny Nc]
# mask    : acceleration mask for forward operator [Nx Ny]
# cg_iter : number of iterations
# xn      : denoised image [2 Nx Ny] 
# (EHE + LI)xn = x0 + L*zn, DC_layer solves xn
def DC_layer(x0,zn,L,S,mask,cg_iter=10):
    p = x0 + L * zn
    r_now = torch.clone(p)
    xn = torch.zeros_like(p)
    for i in range(cg_iter):
        # q = (EHE + LI)p
        q = sf.backward(sf.forward(p,S,mask),S) + L*p  
        # rr_pq = r'r/p'q
        rr_pq = torch.sum(r_now*torch.conj(r_now))/torch.sum(q*torch.conj(p)) 
        xn = xn + rr_pq * p
        r_next = r_now - rr_pq * q
        # p = r_next + r_next'r_next/r_now'r_now
        p = (r_next + 
             (torch.sum(r_next*torch.conj(r_next))/torch.sum(r_now*torch.conj(r_now))) * p)
        r_now = torch.clone(r_next)
    return xn

