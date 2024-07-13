import time
import numpy as np 
import dolfin as dl 

import hippylib as hp
import soupy


def NNMeanVarRiskMeasureSettings(data = {}):
	# This should be a Parameter
	# data['nsamples'] = [100,'Number of Monte Carlo samples']
	data['batch_size'] = [100, 'Batch size for evaluation']
	data['beta'] = [0,'Weighting factor for variance']

	return hp.ParameterList(data)

class NNMeanVarRiskMeasure(soupy.MeanVarRiskMeasure):
	"""
	Class for memory efficient evaluation of the Mean + Variance risk measure 
	E[X] + beta Var[X]. 
	"""

	def __init__(self, nn_control_model, prior, settings = NNMeanVarRiskMeasureSettings()):
		"""
		Parameters
			- :code: `control_model` control model of problem 
			- :code: `prior` prior for uncertain parameter
			- :code: `settings` additional settings
		"""
		self.model = nn_control_model
		self.prior = prior
		self.settings = settings
		# self.settings.showMe()
		# self.n_samples = self.settings['nsamples']
		self.batch_size = self.settings['batch_size']
		self.beta = settings['beta']

		# Aggregate components for computing cost, grad, hess
		self.x = self.model.generate_vector()
		self.g = self.model.generate_vector(soupy.CONTROL)
		self.q_samples = np.array([0])

		self.q_bar = 0
		self.g_bar = self.model.generate_vector(soupy.CONTROL)
		self.qg_bar = self.model.generate_vector(soupy.CONTROL)

		# For sampling
		self.noise = dl.Vector()
		self.prior.init_vector(self.noise, "noise")


		self.dM = self.model.dM
		self.dZ = self.model.dZ


	def generate_vector(self, component="ALL"):
		return self.model.generate_vector(component)

	def computeComponents(self, z, order=0, sample_size=100, rng=None):
		"""
		Computes the components for the stochastic approximation of the cost
		Parameters:
			- :code: `z` the control variable 
			- :code: `order` the order of derivatives needed. 
					0 for cost. 1 for grad. 2 for Hessian
			- :code: `rng` rng for the sampling (optional)
		"""
		t0 = time.time()

		if order >= 1:
			self.g_bar.zero()
			self.qg_bar.zero()

		# Initialize batch arrays
		self.q_samples = np.zeros(sample_size)
		num_batches, remainder = divmod(sample_size, self.batch_size)
		print("Sample size, ", sample_size)
		# print("Batch size, ", self.batch_size)
		# print("Batches, ", num_batches)
		# print("Remainder, ", remainder)

		# Initialize the numpy arrays for inputs
		m_batch = np.zeros((self.batch_size, self.dM))
		z_batch = np.tile(z.get_local(), (self.batch_size, 1))

		for i_batch in range(num_batches):
			# Sample parameters for the batch
			for i_sample in range(self.batch_size):
				if rng is None:
					hp.parRandom.normal(1.0, self.noise)
				else:
					rng.normal(1.0, self.noise)
				self.prior.sample(self.noise, self.x[soupy.PARAMETER])
				m_batch[i_sample, :] = self.x[soupy.PARAMETER].get_local()

			# Compute batch samples 
			q_batch = self.model.cost(m_batch, z_batch)
			self.q_samples[i_batch*self.batch_size : (i_batch + 1)*self.batch_size] = q_batch[:]

			if order >= 1:
				g_batch = self.model.evalGradientControlBatched(m_batch, z_batch)
				qg_batch = (g_batch.T * q_batch).T

				self.g_bar.add_local(1/sample_size * np.sum(g_batch, axis=0))
				self.qg_bar.add_local(1/sample_size * np.sum(qg_batch, axis=0))


		# Initialize the numpy arrays for remaining inputs
		if remainder > 0:
			m_batch = np.zeros((remainder, self.dM))
			z_batch = np.tile(z.get_local(), (remainder, 1))

			# Sample parameters for the batch
			for i_sample in range(remainder):
				if rng is None:
					hp.parRandom.normal(1.0, self.noise)
				else:
					rng.normal(1.0, self.noise)
				self.prior.sample(self.noise, self.x[soupy.PARAMETER])
				m_batch[i_sample, :] = self.x[soupy.PARAMETER].get_local()

			# Compute batch samples 
			q_batch = self.model.cost(m_batch, z_batch)
			self.q_samples[num_batches*self.batch_size : ] = q_batch[:]

			if order >= 1:
				g_batch = self.model.evalGradientControlBatched(m_batch, z_batch)
				qg_batch = (g_batch.T * q_batch).T
				self.g_bar.add_local(1/sample_size * np.sum(g_batch, axis=0))
				self.qg_bar.add_local(1/sample_size * np.sum(qg_batch, axis=0))

		self.q_bar = np.mean(self.q_samples)
		t1 = time.time()
		# print("time for component evaluation: %g" %(t1-t0))

	
	def cost(self):
		"""
		Evaluates the cost given by the risk measure
		Assumes :code: `computeComponents` has been called
		"""
		return self.q_bar + self.beta * np.std(self.q_samples)**2

	def grad(self, g):
		"""
		Evaluates the gradient by the risk measure
		Assumes :code: `computeComponents` has been called with :code: `order>=1`
		Parameters
			- :code: `g` output vector for the gradient
		
		Output
			- :code: `gradnorm` norm of the gradient vector
		"""
		g.zero()
		g.axpy(1.0, self.g_bar)
		g.axpy(2*self.beta, self.qg_bar)
		g.axpy(-2*self.beta*self.q_bar, self.g_bar)
		return np.sqrt(g.inner(g))

	def hessian(self, zhat, Hzhat):
		pass
