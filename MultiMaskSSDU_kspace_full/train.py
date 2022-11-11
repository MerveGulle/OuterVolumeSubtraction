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

# 2) Load Data

# 3) Create Model structure

# 4) Start Training

# 5) Plot the Loss Graph
