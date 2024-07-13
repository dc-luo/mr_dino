import tensorflow as tf 
import numpy as np 
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()

def build_multi_input_network(dM, dZ, dQ, n_before_comb, n_after_comb, breadth):

	# Make input layers
	m_input = tf.keras.layers.Input(shape = (dM, ), name="m_input")
	z_input = tf.keras.layers.Input(shape = (dZ, ), name="z_input")

	# Pass through first layers 
	m_separate = m_input 
	z_separate = z_input 
	for i in range(n_before_comb):
		m_separate = tf.keras.layers.Dense(dM, activation="tanh", name="pre_layer_m%d" %(i))(m_separate)
		z_separate = tf.keras.layers.Dense(dZ, activation="tanh", name="pre_layer_z%d" %(i))(z_separate)

	z_nn = tf.keras.layers.concatenate([m_separate, z_separate], name="concat") 
	# Now combine 
	for i in range(n_after_comb):
		z_nn = tf.keras.layers.Dense(breadth, activation="tanh", name="post_layer_%d" %(i))(z_nn)

	output = tf.keras.layers.Dense(dQ, name="output")(z_nn)

	network = tf.keras.Model([m_input, z_input], output)
	return network 
