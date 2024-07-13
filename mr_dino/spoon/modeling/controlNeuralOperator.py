import dolfin as dl
import tensorflow as tf
import numpy as np

import hippylib as hp
import soupy


def build_control_jacobian_networks(model):
	"""
	Given an input neural network, forms new networks returning
	1. Full Jacobian
	2. Jacobian vector product with input zhat
	3. Jacobian transpose vector product with input qhat
	"""
	print('Assuming that the control input is the second input!!')
	assert len(model.inputs) == 2
	assert len(model.outputs) == 1
	input_m = model.inputs[0]
	input_z = model.inputs[1]
	output_q = model.outputs[0]

	with tf.GradientTape(persistent = True) as tape:
		tape.watch(input_z)
		qout = model([input_m, input_z])

	# Full batched Jacobian
	fullJz = tape.batch_jacobian(qout,input_z)
	Jfull_model = tf.keras.models.Model([input_m,input_z],[fullJz])

	# Jacobian vector products
	dZ = model.input_shape[1][1]
	zhat = tf.keras.layers.Input(shape = (dZ, ))
	Jzhat = tf.einsum('ikj,ij->ik', fullJz, zhat)
	Jprod_model = tf.keras.Model([input_m, input_z, zhat], Jzhat)

	# Jacobian transpose vector products
	dQ = model.output_shape[1]
	qhat = tf.keras.layers.Input(shape = (dQ, ))
	qqhat = tf.einsum('ij, ij -> i', output_q, qhat)
	Jtqhat = tf.gradients(qqhat, input_z, stop_gradients=qhat, name='Jtqhat')[0]
	Jtprod_model = tf.keras.Model([input_m, input_z, qhat], Jtqhat)

	return Jfull_model, Jprod_model, Jtprod_model


class ControlNeuralOperator:
	def __init__(self, Vh, nn):
		self.Vh = Vh 
		self.nn = nn 
		self.dM = self.nn.input_shape[0][1]
		self.dZ = self.nn.input_shape[1][1]
		self.dQ = self.nn.output_shape[1]

		self.fullJz, self.JzProd, self.JztProd = build_control_jacobian_networks(self.nn)


	def generate_parameter(self):
		return dl.Function(self.Vh[soupy.PARAMETER]).vector()

	def generate_control(self):
		return dl.Function(self.Vh[soupy.CONTROL]).vector()

	def generate_state(self):
		return dl.Function(self.Vh[soupy.STATE]).vector()


	def eval(self, mi, zi):
		assert mi.shape[0] == zi.shape[0]
		return self.nn.predict([mi, zi])

	def Jztvp(self, mi, zi, qhat):
		"""
		Perform control Jacobian transpose vector product 
		"""
		assert len(mi.shape) > 1
		assert len(zi.shape) > 1
		assert len(qhat.shape) > 1
		assert mi.shape[0] == zi.shape[0]
		assert mi.shape[0] == qhat.shape[0]
		# The logic here needs to be better thought out to accomodate action on matrices
		# if mi.shape[0] != 1 or len(mi.shape) == 1:
		# 	mi = np.expand_dims(mi,axis = 0)

		# if zi.shape[0] != 1 or len(zi.shape) == 1:
		# 	zi = np.expand_dims(zi,axis = 0)

		# if qhat.shape[0] != 1 or len(mi.shape) == 1:
		# 	qhat = np.expand_dims(qhat,axis = 0)

		return self.JztProd.predict([mi, zi, qhat])

	def Jzvp(self,mi, zi, zhat,sess = None):
		"""
		Perform control Jacobian vector product
		"""
		assert len(mi.shape) > 1
		assert len(zi.shape) > 1
		assert len(zhat.shape) > 1
		assert mi.shape[0] == zi.shape[0]
		assert mi.shape[0] == zhat.shape[0]

		# Dimension checks have been temporarily removed.
		# Currently assumes that number of points and number of action directions are the same
		# Actions on multiple directions at the same point not supported 

		# The logic here needs to be better thought out to accomodate action on matrices
		# if mi.shape[0] != 1 or len(mi.shape) == 1:
		# 	mi = np.expand_dims(mi,axis = 0)

		# if zi.shape[0] != 1 or len(zi.shape) == 1:
		# 	zi = np.expand_dims(zi,axis = 0)

		# if zhat.shape[0] != 1 or len(mi.shape) == 1:
		# 	mhat = np.expand_dims(mhat,axis = 0)

		return self.JzProd.predict([mi, zi,zhat])

	def Jz(self, mi, zi):
		"""
		Evaluate the full Jacobian 
		"""
		assert len(mi.shape) > 1
		assert len(zi.shape) > 1
		assert mi.shape[0] == zi.shape[0]

		# if mi.shape[0] != 1 or len(mi.shape) == 1:
		# 	mi = np.expand_dims(mi,axis = 0)

		# if int(tf.__version__[0]) > 1:
		# 	sess = tf.compat.v1.keras.backend.get_session()
		# else:
		# 	sess = tf.keras.backend.get_session()

		# return sess.run(self.fullJz,feed_dict={self.input_m:mi})[0]

		# Currently uses this. Not sure what to do with sessions
		return self.fullJz.predict([mi, zi])
