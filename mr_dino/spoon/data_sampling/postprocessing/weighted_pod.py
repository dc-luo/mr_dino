import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
import scipy.linalg 
import dolfin as dl
import numpy as np

def build_mass_matrix_csr(Vh):
    trial_fun = dl.TrialFunction(Vh)
    test_fun = dl.TestFunction(Vh)
    M = dl.PETScMatrix()
    dl.assemble(dl.inner(trial_fun, test_fun) * dl.dx, tensor=M)

    M_mat = dl.as_backend_type(M).mat()
    row,col,val = M_mat.getValuesCSR()
    M_csr = sp.csr_matrix((val,col,row))
    return M_csr

def _weighted_l2_norm_vector(x, W):
    Wx = W @ x 
    norm2 = np.einsum('ij,ij->j', Wx, x)
    return np.sqrt(norm2)

def generate_weighted_pod(u_data, M_csr, num_modes, method='hep', verify=True):
    """
    Compute the matrix weighted POD 
    - :code: `u_data`: numpy array with each column being a data vector
    - :code: `M_csr`: scipy.sparse.csr_matrix for the weighting
    - :code: `num_modes`: number of POD modes to compute 
    - :code: `verify`: Boolean, verify the accuracy of the POD approximation 
    """
    tpre0 = time.time()
    if method == 'ghep':
        # Mass matrix and inverse as linear operators
        M_lu_factors = spla.splu(M_csr.tocsc())
        M_inv_op = spla.LinearOperator(shape=M_csr.shape, matvec=M_lu_factors.solve)
        M_op = spla.aslinearoperator(M_csr)

        # Compute AA^T for the data matrix 
        H = u_data @ u_data.T 
        H_op = spla.aslinearoperator(H)
        
        tpre1 = time.time()
        print(f"Preprocessing took {tpre1 - tpre0:.3g} seconds")
        # solve generalized eigenvalue problem 
        print("Solving eigenvalue problem")

        t0 = time.time()

        d, Mphi = spla.eigsh(H, k=num_modes, M=M_inv_op, Minv=M_op)
        d = np.flipud(d)
        Mphi = np.fliplr(Mphi)

        t1 = time.time()
        print("Post process eigenvectors to get POD modes")
        phi = M_inv_op @ Mphi 
        t2 = time.time()
        print("Done")
        print(f"Eigenvalue solve took {t1 - t0:.3g} seconds")
        print(f"Postprocessing by matrix solve took {t2 - t1:.3g} seconds")


    elif method == 'hep':
        t0 = time.time()
        UtMU = u_data.T @ M_csr @ u_data 
        t1 = time.time()

        s, U = scipy.linalg.eigh(UtMU) 
        d = np.flipud(s)[0:num_modes]
        U = np.fliplr(U)[:,0:num_modes]

        t2 = time.time()
        phi = u_data @ U 
        t3 = time.time()

        phi = phi/_weighted_l2_norm_vector(phi, M_csr)
        Mphi = M_csr @ phi 
        print(f"Preprocessing took {t1 - t0:.3g} seconds")
        print(f"Eigenvalue solve took {t2 - t1:.3g} seconds")
        print(f"Postprocessing took {t3 - t2:.3g} seconds")

    else:
        raise ValueError("Unavailable method")

    if verify:
        phi_orth_error = np.linalg.norm(phi[:,:-1].T @ M_csr @ phi[:,:-1] - np.eye(num_modes-1))
        print(f"PHI Orthogonality error: {phi_orth_error}")

        Mphi_orth_error = np.linalg.norm(phi[:,:-1].T @ Mphi[:,:-1] - np.eye(num_modes-1))
        print(f"PHI MPHI Orthogonality error: {Mphi_orth_error}")

        reconstruction_diff = u_data - (phi[:,:-1] @ (Mphi[:,:-1].T @ u_data))

        error_with_data = _weighted_l2_norm_vector(reconstruction_diff, M_csr)
        norm_of_data = _weighted_l2_norm_vector(u_data, M_csr)
        rel_error_in_each_data = error_with_data/norm_of_data
        print(f"Mean reconstruction error: {np.mean(rel_error_in_each_data):.3e}")
        print(f"Max reconstruction error: {np.max(rel_error_in_each_data):.3e}")

    return d, phi, Mphi
