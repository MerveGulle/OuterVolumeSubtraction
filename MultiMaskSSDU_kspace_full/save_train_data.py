from scipy.io import loadmat

data_indices = loadmat('data_indices.mat')
_, indices = list(data_indices.items())[3]
# indices.shape = (3,14)
# indices[0]: subject numbers
# indices[1]: slice numbers
# indices[2]: time frame numbers (number of slices x 3)


