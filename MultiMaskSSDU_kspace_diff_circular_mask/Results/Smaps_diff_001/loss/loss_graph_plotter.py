import torch
import numpy as np
import matplotlib.pyplot as plt


loss_arr = torch.load("train_loss.pt")
loss_arr_valid = torch.load("valid_loss.pt")
loss_arr_valid = loss_arr_valid[~(loss_arr_valid==0)]

figure = plt.figure()
n = np.arange(loss_arr_valid.shape[0])
plt.plot(n+1,loss_arr[n],n+1,loss_arr_valid)
plt.xlabel('epoch')
plt.title('Loss Graph')
plt.legend(['train loss', 'validation loss'])
figure.savefig('loss_graph.png')