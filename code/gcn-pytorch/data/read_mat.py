import scipy.io as sio
import numpy as np

a_cora = sio.loadmat('A_Cora.mat')
print(a_cora['A'])


