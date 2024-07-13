import time

import numpy as np 
import dolfin as dl 

import hippylib as hp
import soupy


def NNMeanVarRiskMeasureSAASettings(data = {}):
    # This should be a Parameter
    data['sample_size'] = [100,'Number of Monte Carlo samples']
    data['beta'] = [0,'Weighting factor for variance']
    data['seed'] = [1, 'rng seed for sampling']

    return hp.ParameterList(data)

class NNMeanVarRiskMeasureSAA:
    """
    Class for memory efficient evaluation of the Mean + Variance risk measure 
    E[X] + beta Var[X]. 
    """

    def __init__(self, nn_control_model, prior, settings=NNMeanVarRiskMeasureSAASettings(), 
            m_samples=None, project_m=False):
        """
        Parameters
            - :code: `control_model` control model of problem 
            - :code: `prior` prior for uncertain parameter
            - :code: `settings` additional settings
            - :code: `m_samples` numpy array of samples to use instead of sampling within class
            - :code: `project_m` project samples into reduced subspace, applicable for reduced basis networks
        """
        self.model = nn_control_model
        self.prior = prior
        self.settings = settings
        # self.settings.showMe()
        self.sample_size = self.settings['sample_size']
        self.beta = settings['beta']

        self.dM = self.model.dM
        self.dZ = self.model.dZ

        if m_samples is not None:
            assert self.sample_size == m_samples.shape[0], "inconsistent sample sizes"
            assert self.dM == m_samples.shape[1], "inconsistent dimensions of parameter samples"

        # Aggregate components for computing cost, grad, hess
        self.x = self.model.generate_vector()
        self.g = self.model.generate_vector(soupy.CONTROL)
        self.q_samples = np.zeros(self.sample_size)

        self.q_bar = 0
        self.g_bar = self.model.generate_vector(soupy.CONTROL)
        self.qg_bar = self.model.generate_vector(soupy.CONTROL)

        if m_samples is None:
            # Draw samples from prior 
            print("Risk measure: sampling from prior")
            self.m = self.model.generate_vector(soupy.PARAMETER)
            self.noise = dl.Vector()
            self.prior.init_vector(self.noise, "noise")
            rng = hp.Random(seed=self.settings['seed'])

            self.m_data = np.zeros((self.sample_size, self.dM))
            # Generate samples for m 
            for i in range(self.sample_size):
                # hp.parRandom.normal(1.0, self.noise)
                rng.normal(1.0, self.noise)
                self.prior.sample(self.noise, self.m)
                self.m_data[i] = self.m.get_local()
        else:
            # Use input samples
            print("Risk measure: using input samples")
            self.m_data = m_samples

        self.project_m = project_m
        if project_m:
            assert hasattr(nn_control_model, 'project_m')
            self.m_data = nn_control_model.project_m(self.m_data)

        self.z_data = np.zeros((self.sample_size, self.dZ))

        self.g_samples = np.zeros((self.sample_size, self.dZ))
        self.qg_samples = np.zeros((self.sample_size, self.dZ))

    def generate_vector(self, component="ALL"):
        return self.model.generate_vector(component)

    def computeComponents(self, z, order=0):
        """
        Computes the components for the stochastic approximation of the cost
        Parameters:
            - :code: `z` the control variable 
            - :code: `order` the order of derivatives needed. 
                    0 for cost. 1 for grad. 2 for Hessian
        """
        t0 = time.time()
        z_vec = z.get_local()
        for i in range(self.sample_size):
            self.z_data[i] = z_vec

        self.q_samples = self.model.cost(self.m_data, self.z_data)

        if order >= 1:
            self.g_samples = self.model.evalGradientControlBatched(self.m_data, self.z_data)
            self.qg_samples = (self.g_samples.T * self.q_samples).T
            self.g_bar.set_local(np.mean(self.g_samples, axis=0))
            self.qg_bar.set_local(np.mean(self.qg_samples, axis=0))

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
