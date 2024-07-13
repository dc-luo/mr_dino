import unittest
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
sys.path.append("../../../")

from buildMultiInputNN import build_multi_input_network
from spoon import ControlNeuralOperator

class TestNeuralOperatorJacobian(unittest.TestCase):
	def setUp(self):
		N_TRAIN = 10

		N_LAYERS_BEFORE_COMBINATION = 1
		N_LAYERS_AFTER_COMBINATION = 1
		BREADTH = 6

		self.dM = 2
		self.dZ = 3
		self.dQ = 4
		m_train = np.random.randn(N_TRAIN, self.dM)
		z_train = np.random.randn(N_TRAIN, self.dZ)
		q_train = np.random.randn(N_TRAIN, self.dQ)

		self.nn = build_multi_input_network(self.dM, self.dZ, self.dQ, N_LAYERS_BEFORE_COMBINATION, N_LAYERS_AFTER_COMBINATION, BREADTH)
		self.neural_operator = ControlNeuralOperator(None, self.nn)

		self.TOL = 1e-2

	def testFullJacobian(self):
		m = np.random.randn(1, self.dM)
		z = np.random.randn(1, self.dZ)
		Jfull = self.neural_operator.Jz(m, z)[0]

		print("Check correct shape of Jacobian")
		self.assertEqual(Jfull.shape[0], self.dQ)
		self.assertEqual(Jfull.shape[1], self.dZ)

		q0 = self.neural_operator.eval(m, z)[0]

		# Finite difference check individual components
		delta = 1e-3
		for i in range(self.dZ):
			dz = np.zeros((1, self.dZ))
			dz[0,i] = 1.0
			q1 = self.neural_operator.eval(m, z + delta * dz)[0]

			fd_grad = (q1 - q0)/delta
			jac_grad = Jfull[:,i]
			diff_grad = fd_grad - jac_grad

			diff_norm = np.linalg.norm(diff_grad, 2)
			dqdz_norm = np.linalg.norm(jac_grad, 2)
			print("Component %d" %(i))
			print("Finite difference: ")
			print(fd_grad)
			print("Jacobian")
			print(jac_grad)

			self.assertTrue(diff_norm/dqdz_norm < self.TOL)

	def testJtvp(self):
		m = np.random.randn(1, self.dM)
		z = np.random.randn(1, self.dZ)
		qhat = np.random.randn(1, self.dQ)

		Jfull = self.neural_operator.Jz(m, z)[0]
		Jtqhat_full = Jfull.T @ qhat[0]
		Jtqhat_nn = self.neural_operator.Jztvp(m, z, qhat)[0]

		ref_norm = np.linalg.norm(Jtqhat_full, 2)
		diff = Jtqhat_full - Jtqhat_nn
		diff_norm = np.linalg.norm(diff, 2)

		print("Full Jacobian transpose")	
		print(Jtqhat_full)
		print("Jacobian transpose network")
		print(Jtqhat_nn)
		self.assertTrue(diff_norm/ref_norm < self.TOL)


	def testJvp(self):
		m = np.random.randn(1, self.dM)
		z = np.random.randn(1, self.dZ)
		zhat = np.random.randn(1, self.dZ)

		Jfull = self.neural_operator.Jz(m, z)[0]
		Jzhat_full = Jfull @ zhat[0]
		Jzhat_nn = self.neural_operator.Jzvp(m, z, zhat)[0]

		ref_norm = np.linalg.norm(Jzhat_full, 2)
		diff = Jzhat_full - Jzhat_nn
		diff_norm = np.linalg.norm(diff, 2)

		print("Full Jacobian action")	
		print(Jzhat_full)
		print("Jacobian action network")
		print(Jzhat_nn)
		self.assertTrue(diff_norm/ref_norm < self.TOL)


if __name__ == "__main__":
	unittest.main()
