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
from spoon import NNCompleteQoIControlModel

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

class TestNNCompleteQoIControlModel(unittest.TestCase):
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

        dQ = 1
        self.dU = u_np.shape[0]
        self.dM = m_np.shape[0]
        self.dZ = z_np.shape[0]

        nn_qoi = build_multi_input_network(self.dM, self.dZ, dQ, 1, 1, 50)
        self.completeQoIControlModel = NNCompleteQoIControlModel(self.Vh, nn_qoi) 

    def testGeneration(self): 
        u = self.completeQoIControlModel.generate_vector(soupy.STATE)
        m = self.completeQoIControlModel.generate_vector(soupy.PARAMETER)
        p = self.completeQoIControlModel.generate_vector(soupy.ADJOINT)
        z = self.completeQoIControlModel.generate_vector(soupy.CONTROL)

        x_list = [u, m, p, z]
        x = self.completeQoIControlModel.generate_vector()

        self.assertEqual(u.get_local().shape[0], self.dU)
        self.assertEqual(m.get_local().shape[0], self.dM)
        self.assertEqual(p.get_local().shape[0], self.dU)
        self.assertEqual(z.get_local().shape[0], self.dZ)

        for ind in [soupy.STATE, soupy.PARAMETER, soupy.ADJOINT, soupy.CONTROL]:
            self.assertEqual(x[ind].get_local().shape[0], x_list[ind].get_local().shape[0])

        print("Vectors generated have the correct shape")

    def testGradient(self):
        dM = self.completeQoIControlModel.dM
        dZ = self.completeQoIControlModel.dZ

        m_np = np.random.randn(1, dM)
        z_np = np.random.randn(1, dZ)
        g_np = self.completeQoIControlModel.evalGradientControlBatched(m_np, z_np)

        delta = 5e-3

        q0 = self.completeQoIControlModel.cost(m_np, z_np)[0]
        g_fd = np.zeros(dZ)
    
        # By Finite difference 
        for i in range(dZ):
            dz_np = np.zeros((1, dZ))
            dz_np[0, i] = 1.0
            q1 = self.completeQoIControlModel.cost(m_np, z_np + delta * dz_np)[0]
            g_fd[i] = (q1-q0)/delta

        print("Gradient by end to end NN control model\n", g_np)
        print("Gradient by finite difference\n", g_fd)
        RTOL = 1e-2

        ref_norm = np.linalg.norm(g_np)
        diff_norm = np.linalg.norm(g_np - g_fd)

        print("FD error: %.3e" %(diff_norm/ref_norm))
        self.assertTrue(diff_norm/ref_norm < RTOL)


if __name__ == "__main__":
    unittest.main()
