import tensorflow as tf 

def equip_model_with_full_control_jacobian(model,name_prefix = ''):
	"""
	"""
	print('Assuming that the control input is the second input!!')
	assert len(model.inputs) == 2
	assert len(model.outputs) == 1
	input_m = model.inputs[0]
	input_z = model.inputs[1]
	output_q = model.outputs[0]

	with tf.GradientTape(persistent = True) as tape:
		tape.watch(input_z)
		qout = model(model.inputs)
	# Full batched Jacobian
	fullJz = tape.batch_jacobian(qout,input_z)

	new_model = tf.keras.models.Model([input_m,input_z],[output_q,fullJz])

	return new_model