import numpy as np
import dolfin as dl
import tensorflow as tf
import scipy.sparse as sp
from .nnCompleteQoI import batch_inner_product

# def batch_inner_product_numpy(x, y):
#     x = np.expand_dims(x, -1)
#     y = np.expand_dims(y, -1)
#     xTy = np.einsum('ijk,ijk->ik', x, y)
#     return xTy

class ReducedBasisL2QoI:
    """
    Class for a quadratic QoI rb net
    """
    def __init__(self, W, u_basis, u_projector=None, u_shift=None, d=None):
        """
        - :code:`W` Weighting matrix for the L2 norm
        - :code:`u_basis` output basis 
        - :code:`u_projector` dual of the output basis, 
            i.e. `u_projector.T` projects into reduced coefficients. 
            Assumed to be the same as `u_basis` if not supplied. 
            i.e. `u_basis.T @ u_basis = identity`
        - :code: `u_shift` shift vector for the output. Assumed to be zero
            if not supplied
        - :code:`d` data vector. Assumed to be zero if not supplied.
        """
        self.W = W
        self.d = d
        self.u_basis = u_basis
        if u_projector is None:
            self.u_projector = u_basis
        else:
            self.u_projector = u_projector
        self.u_shift = u_shift 

        W_mat = dl.as_backend_type(W).mat()
        row, col, val = W_mat.getValuesCSR()
        self.W_csr = sp.csr_matrix((val, col, row))

        # Reducing the components
        W_times_u_basis = self.W_csr @ u_basis
        self.reduced_W = tf.constant(u_basis.T @ W_times_u_basis, dtype=tf.float32)

        if self.d is None:
            self.d_np = np.zeros(u_basis.shape[0])
        else:
            self.d_np = d.get_local()

        if u_shift is not None:
            self.d_np -= u_shift

        # Project data into reduced cooredinates
        self.reduced_d = self.u_projector.T @ self.d_np

        # Residual of projection
        self.perp_d = self.d_np - self.u_basis @ self.reduced_d

        # Contribution due to mixed term
        self.reduced_perp_d = self.W_csr @ self.perp_d
        self.reduced_perp_d = self.u_basis.T @ self.reduced_perp_d

        # Contribution due to pure residual
        self.reduced_residual = np.inner(self.perp_d, self.W_csr @ self.perp_d)

        self.reduced_d = tf.constant(self.reduced_d, dtype=tf.float32)
        self.perp_d = tf.constant(self.perp_d, dtype=tf.float32)
        self.reduced_perp_d = tf.constant(self.reduced_perp_d, dtype=tf.float32)
        

    def eval_u(self, reduced_u_np):
        # if self.d is None:
            # diff = reduced_u_np
        # else:
        diff = reduced_u_np - self.reduced_d
        Wdiff = diff @ self.reduced_W
        reduced_qoi = batch_inner_product(diff, Wdiff)

        # if self.d is not None:
        reduced_qoi -= 2 * diff @ tf.expand_dims(self.reduced_perp_d, axis=-1)
        reduced_qoi += self.reduced_residual
        return reduced_qoi

    def grad_u(self, reduced_u_np):
        # if self.d is None:
            # grad = 2 * reduced_u_np @ self.reduced_W
        # else:
        diff = reduced_u_np - self.reduced_d
        grad = 2 * diff @ self.reduced_W - 2 * self.reduced_perp_d
        return grad
