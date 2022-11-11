import torch
import torch.nn as nn
import SupportingFunctions as sf

#                     ______________________________________nth_iteration___
#                    |  ______________                          __________  |
#        x(n)        | |              |          z(n)          |          | |    x(n+1)
#   ---------------->| | CNN denoiser | ---------------------> | DC layer | |-------------->
#     [2 2Nx Ny]  |  | |______________| [2 2Nx Ny]-->[2 Nx Ny] |__________| |  | [2 Nx Ny]
#       (real)    |  |                    (real)  to (complex)              |  | (complex)
#                 |  |______________________________________________________|  |
#                 |----------<--------<--------<--------<---------<------------|
#                          (complex) [2 Nx Ny] to (real) [2 2Nx 2Ny]


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

# complex 1 channel to real 2 channels
def ch1to2(data1):       
    return torch.cat((data1.real,data1.imag),0)
# real 2 channels to complex 1 channel
def ch2to1(data2):       
    return data2[0:1,:,:] + 1j * data2[1:2,:,:] 


# x: complex [2 Nx Ny] ---> y: real [2 2Nx Ny]
# prepare the DC layer output for the CNN denoiser
def complex2real(x):
    y = torch.cat((x[0],x[1]),1)
    y = torch.cat((y.real,y.imag),0)
    return y


# x: real [2 2Nx Ny] ---> y: complex [2 Nx Ny]
# prepare the CNN denoiser output for the DC layer
def real2complex(x):
    Nx = x.shape(1)//2
    y = x[0:1] + 1j*x[1:2]
    y = torch.cat((y[:,0:Nx],y[:,Nx:2*Nx]),0)
    return y


# define RB:residual block (conv + ReLU + conv + xScale)
# input(xn) : output of DC layer, noisy image [2 2*Nx Ny]
# output(zn): denoised image [2 2*Nx Ny]
# convolutional blocks share the same coefficients
class RB(nn.Module):
    def __init__(self, C=0.1):
        super().__init__()
        self.conv  = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.relu  = nn.ReLU()
        self.C     = C
    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.conv(y)
        y = y*self.C
        y = y + x
        return y
    

# define ResNet Block
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False)
        self.RB1   = RB()
        self.RB2   = RB()
        self.RB3   = RB()
        self.RB4   = RB()
        self.RB5   = RB()
        self.RB6   = RB()
        self.RB7   = RB()
        self.RB8   = RB()
        self.RB9   = RB()
        self.RB10  = RB()
        self.RB11  = RB()
        self.RB12  = RB()
        self.RB13  = RB()
        self.RB14  = RB()
        self.RB15  = RB()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=3, padding=1, bias=False)
        self.L = nn.Parameter(torch.tensor(0.05, requires_grad=True))
    def forward(self, x):
        z = sf.complex2real(x).float()
        z = self.conv1(z)
        r = self.RB1(z)
        r = self.RB2(r)
        r = self.RB3(r)
        r = self.RB4(r)
        r = self.RB5(r)
        r = self.RB6(r)
        r = self.RB7(r)
        r = self.RB8(r)
        r = self.RB9(r)
        r = self.RB10(r)
        r = self.RB11(r)
        r = self.RB12(r)
        r = self.RB13(r)
        r = self.RB14(r)
        r = self.RB15(r)
        r = self.conv2(r)
        z = r + z
        z = self.conv3(z)
        z = sf.real2complex(z)
        return self.L, z
    
    
def weights_init_normal(m):
  if isinstance(m, nn.Conv2d):
      nn.init.normal_(m.weight.data,mean=0.0,std=0.05)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)