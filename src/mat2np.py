import scipy.io as sio
import numpy as np

def mat2np(path):
    mat_data = sio.loadmat(path)

    mydata = mat_data['data']

    data = []

    for i in range(mydata.shape[0]):
        struct = mydata[i, 0]
        x = struct['x'][0][0]
        data.append(np.array(x).T)

    return data

data = mat2np('ro/BuckConverter.mat')
print(data)