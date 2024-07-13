import unittest

import dolfin as dl 
import tensorflow as tf 
import numpy as np 

if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()


import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))

import hippylib as hp
import soupy

from buildMultiInputNN import build_multi_input_network
sys.path.append("../../../")
from spoon import ControlNeuralOperator, NNControlModel, build_complete_qoi_network, NNSparseL2QoI


def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)


class TestNNControlModel(unittest.TestCase):
    def setUp(self):
        self.nx = 20
        self.ny = 20
        self.n_control = 25
        
        # Make spaces
        self.mesh = dl.UnitSquareMesh(self.nx, self.ny)
        Vh_STATE = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_CONTROL = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.n_control)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

        u_np = dl.Function(Vh_STATE).vector().get_local()
        m_np = dl.Function(Vh_PARAMETER).vector().get_local()
        z_np = dl.Function(Vh_CONTROL).vector().get_local()

        dQ = u_np.shape[0]
        dM = m_np.shape[0]
        dZ = z_np.shape[0]

        def l2norm(u,m,z):
            return u**2*dl.dx 

        self.qoi = soupy.VariationalControlQoI(self.Vh, l2norm)
        nn = build_multi_input_network(dM, dZ, dQ, 1, 1, 50)
        self.neural_operator = ControlNeuralOperator(self.Vh, nn)
        self.controlModel = NNControlModel(self.Vh, self.neural_operator, self.qoi)
        
        # Set up the monolithic networks 
        u_trial = dl.TrialFunction(Vh_STATE)
        u_test = dl.TestFunction(Vh_STATE)
        W = dl.assemble(u_trial * u_test * dl.dx) 
        d = dl.Function(self.Vh[hp.STATE]).vector()
        self.qoi_nn, self.grad_nn = build_complete_qoi_network(self.neural_operator, NNSparseL2QoI(W, d))



    def testEvaluation(self):
        dQ = self.controlModel.dQ
        dM = self.controlModel.dM
        dZ = self.controlModel.dZ

        N_vec = 10
        m_np = np.random.randn(N_vec, dM)
        z_np = np.random.randn(N_vec, dZ)
        q_np = self.controlModel.cost(m_np, z_np)
        u_np = self.neural_operator.eval(m_np, z_np)
        print("Qoi evaluated from neural operator: ", q_np)

        x = self.controlModel.generate_vector("ALL")
        q_by_qoi = np.zeros(N_vec)
        for i in range(N_vec):
            x[soupy.STATE].set_local(u_np[i,:])
            x[soupy.PARAMETER].set_local(m_np[i,:])
            x[soupy.CONTROL].set_local(z_np[i,:])
            q_by_qoi[i] = self.qoi.cost(x)

        print("Qoi evaluated from fenics: ", q_by_qoi)
        ATOL = 1e-3

        print(m_np.shape)
        q_by_nn = self.qoi_nn.predict([m_np, z_np])[:,0]
        print("QoI by end to end nn: ", q_by_nn)

        self.assertTrue(np.linalg.norm(q_by_nn - q_np, 2) < ATOL)


    def testGradient(self):
        dQ = self.controlModel.dQ
        dM = self.controlModel.dM
        dZ = self.controlModel.dZ

        m_np = np.random.randn(1, dM)
        z_np = np.random.randn(1, dZ)
        g_np = self.controlModel.evalGradientControlBatched(m_np, z_np)

        g_nn = self.grad_nn.predict([m_np, z_np])

        delta = 1e-3

        q0 = self.controlModel.cost(m_np, z_np)[0]
        g_fd = np.zeros(dZ)

        for i in range(dZ):
            dz_np = np.zeros((1, dZ))
            dz_np[0, i] = 1.0
            q1 = self.controlModel.cost(m_np, z_np + delta * dz_np)[0]
            g_fd[i] = (q1-q0)/delta

        print(g_np)
        print(g_fd)
        print(g_nn)
        RTOL = 1e-2

        ref_norm = np.linalg.norm(g_np)
        diff_norm = np.linalg.norm(g_np - g_nn)

        print("Derivative error: %.3e" %(diff_norm/ref_norm))
        self.assertTrue(diff_norm/ref_norm < RTOL)


if __name__ == "__main__":
    unittest.main()
