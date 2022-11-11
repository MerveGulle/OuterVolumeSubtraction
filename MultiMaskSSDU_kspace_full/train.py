import model
import torch
import numpy as np
import random


### HYPERPARAMETERS
params = dict([('num_epoch', 100),
               ('batch_size', 1),
               ('learning_rate', 1e-3),
               ('use_cpu', False),
               ('T', 10)])                  # number of iterations

### PATHS 
train_data_path = "placeholder"

# 0) Fix randomness for reproducible experiment
torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# 1) Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and (not(params['use_cpu']))) else 'cpu')

# 2) Load Data

# 3) Create Model structure
denoiser = model.ResNet().to(device)
denoiser.apply(model.weights_init_normal)
optimizer = torch.optim.Adam(denoiser.parameters(),lr=params['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# 4) Start Training

# 5) Plot the Loss Graph
