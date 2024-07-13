import time

import scipy.sparse as sp
import scipy.sparse.linalg as spla 
import scipy.linalg 
import dolfin as dl
import numpy as np

def compute_data_normalization(data_arrays):
    normalizations = []
    for data_array in data_arrays:
        data_mean = np.mean(data_array, axis=0)
        data_sd = np.std(data_array, axis=0)
        normalizations.append([data_mean, data_sd])
    return normalizations

def project_data(data_arrays, basis_arrays, shifts=None):
    """
    Applies projection operator for each data basis pair
    """
    assert len(data_arrays) == len(basis_arrays)
    reduced_data_arrays = []

    if shifts is None:
        for data_array, basis_array in zip(data_arrays, basis_arrays):
            reduced_data_arrays.append(data_array @ basis_array)

    else:
        for data_array, basis_array, shift in zip(data_arrays, basis_arrays, shifts):
            if shift is None:
                reduced_data_arrays.append(data_array @ basis_array)
            else:
                reduced_data_arrays.append((data_array - shift) @ basis_array)


    return reduced_data_arrays


def project_jacobian_data(Jz_data, u_basis):
    """
    Applies projection operator for jacobian stored in `Jz_data`
    """

    Uz_data = Jz_data['Uz_data']
    sigmaz_data = Jz_data['sigmaz_data']
    Vz_data = Jz_data['Vz_data']

    pod_Jz_data = []
    for i in range(Uz_data.shape[0]):
        temp = u_basis.T @ Uz_data[i]
        temp = temp @ np.diag(sigmaz_data[i])
        pod_Jz = temp @ Vz_data[i].T
        pod_Jz_data.append(pod_Jz)

        if i % 100 == 0:
            print("Processing data point %d" %(i))
    return pod_Jz_data