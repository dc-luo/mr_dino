import tensorflow as tf 
import numpy as np 
import dolfin as dl
import scipy.sparse as sp 

import hippylib as hp
import soupy

def batch_inner_product(x, y):
    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)
    xTy = tf.einsum('ijk,ijk->ik', x, y)
    return xTy


def build_sparse_l2_qoi_network(neural_operator, W_sp, d_np):
    """
    Build neural networks based on input neural operator given the L2 misfit
    QoI defined by sparse weighting matrix `W_sp` and data `d_np`
    """

    input_m = neural_operator.nn.inputs[0]
    input_z = neural_operator.nn.inputs[1]

    output_u = neural_operator.nn([input_m, input_z])

    diff = output_u - d_np
    Mdiff = tf.sparse.sparse_dense_matmul(diff, W_sp)

    output_qoi = batch_inner_product(diff, Mdiff)
    qoi_model = tf.keras.Model(inputs=[input_m, input_z], outputs=output_qoi)

    output_grad = neural_operator.JztProd([input_m, input_z, 2 * Mdiff])
    qoi_grad_model = tf.keras.Model(inputs=[input_m, input_z], outputs=output_grad)

    return qoi_model, qoi_grad_model


def build_complete_qoi_network(neural_operator, neural_qoi):
    """
    Build neural networks based on input neural operator given the L2 misfit
    """

    input_m = neural_operator.nn.inputs[0]
    input_z = neural_operator.nn.inputs[1]
    output_u = neural_operator.nn([input_m, input_z])

    output_qoi = neural_qoi.eval_u(output_u)
    qoi_model = tf.keras.Model(inputs=[input_m, input_z], outputs=output_qoi)

    output_grad = neural_operator.JztProd([input_m, input_z, neural_qoi.grad_u(output_u)])
    qoi_grad_model = tf.keras.Model(inputs=[input_m, input_z], outputs=output_grad)

    return qoi_model, qoi_grad_model




class NNSparseL2QoI:
    def __init__(self, W, d=None):
        self.W = W 
        self.d = d 

        W_mat = dl.as_backend_type(W).mat()
        row, col, val = W_mat.getValuesCSR()
        W_csr = sp.csr_matrix((val, col, row))
        nonzeroinds = np.split(W_csr.indices, W_csr.indptr)[1:-1]
        inds = []
        for i in range(len(nonzeroinds)):
            for j in nonzeroinds[i]:
                inds.append((i,j))

        vals = W_csr.data 
        self.W_sp = tf.sparse.SparseTensor(inds, vals.astype(np.float32), W_csr.shape)
        if self.d is not None:
            self.d_np = d.get_local()

    def eval_u(self, u_np):
        if self.d is None:
            diff = u_np 
        else:
            diff = u_np - self.d_np
        Mdiff = tf.sparse.sparse_dense_matmul(diff, self.W_sp)
        return batch_inner_product(diff, Mdiff)

    def grad_u(self, u_np):
        if self.d is None:
            diff = u_np 
        else:
            diff = u_np - self.d_np
        Mdiff = tf.sparse.sparse_dense_matmul(diff, self.W_sp)
        return 2 * Mdiff
