import numpy as np

slopes_array = np.load('slopes.npy')
intercept_array = np.load('intercept.npy')
data = np.vstack((slopes_array, intercept_array)).T
print(data[0])
