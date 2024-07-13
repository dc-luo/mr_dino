import time
import numpy as np
import dolfin as dl

import hippylib as hp
import soupy


def NNSuperquantileRiskMeasureSAASettings(data = {}):
    # This should be a Parameter
    data['sample_size'] = [100,'Number of Monte Carlo samples']
    data['beta'] = [0.95,'Weighting factor for variance']
    data['seed'] = [1, 'rng seed for sampling']
    data['smoothplus_type'] = ['quartic', 'approximation type for smooth plus function']
    data['epsilon'] = [0.01, 'Sharpness of smooth plus approximation']

    return hp.ParameterList(data)

class NNSuperquantileRiskMeasureSAA:
    """
    Class for superquantile risk measure, defined in terms of the alternative minimization formulation
    min J(z,t) := t + 1/(1-beta) * E[max(Q-t, 0)]
    """

    def __init__(self, nn_control_model, prior, settings = NNSuperquantileRiskMeasureSAASettings(), m_samples=None, project_m=False):
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



        if self.settings["smoothplus_type"] == "softplus":
            self.smoothplus = soupy.SmoothPlusApproximationSoftplus(self.settings["epsilon"])
        elif self.settings["smoothplus_type"] == "quartic":
            self.smoothplus = soupy.SmoothPlusApproximationQuartic(self.settings["epsilon"])
        else:
            # Default case
            self.smoothplus = soupy.SmoothPlusApproximationQuartic(self.settings["epsilon"])

        # Aggregate components for computing cost, grad, hess
        self.x = self.model.generate_vector()
        self.g = self.model.generate_vector(soupy.CONTROL)
        self.q_samples = np.zeros(self.sample_size)

        self.q_bar = 0
        self.s_bar = 0
        self.sprime_bar = 0
        self.sprime_g_bar = self.model.generate_vector(soupy.CONTROL)
        self.t = 0

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
        # self.qg_samples = np.zeros((self.sample_size, self.dZ))

    def generate_vector(self, component = "ALL"):
        if component == soupy.CONTROL:
            return soupy.AugmentedVector(self.model.generate_vector(soupy.CONTROL), copy_vector=False)
        else:
            return self.model.generate_vector(component)


    def computeComponents(self, zt, order=0, rng=None):
        """
        Computes the components for the stochastic approximation of the cost
        Parameters:
            - :code: `zt` the augmented control variable by t
            - :code: `order` the order of derivatives needed.
                    0 for cost. 1 for grad. 2 for Hessian
            - :code: `rng` rng for the sampling (optional)
        """
        t0 = time.time()
        zt_vec = zt.get_local()
        self.t = zt_vec[-1]

        for i in range(self.sample_size):
            self.z_data[i] = zt_vec[:-1]

        self.q_samples = self.model.cost(self.m_data, self.z_data)
        self.s_samples = self.smoothplus(self.q_samples - self.t)
        self.q_bar = np.mean(self.q_samples)
        self.s_bar = np.mean(self.s_samples)

        if order >= 1:
            self.g_samples = self.model.evalGradientControlBatched(self.m_data, self.z_data)
            self.sprime_samples = self.smoothplus.grad(self.q_samples - self.t)
            self.sprime_g_samples = (self.g_samples.T * self.sprime_samples).T
            self.sprime_bar = np.mean(self.sprime_samples, axis=0)
            self.sprime_g_bar = np.mean(self.sprime_g_samples, axis=0)


        t1 = time.time()
        # print("time for component evaluation: %g" %(t1-t0))


    def cost(self):
        """
        Evaluates the cost given by the risk measure
        Assumes :code: `computeComponents` has been called
        """
        return self.t + 1/(1 - self.beta) * self.s_bar

    def grad(self, gt):
        """
        Evaluates the gradient by the risk measure
        Assumes :code: `computeComponents` has been called with :code: `order>=1`
        Parameters
            - :code: `gt` output augmented vector for the gradient

        Output
            - :code: `gradnorm` norm of the gradient vector
        """
        dzJ_np = self.sprime_g_bar/(1-self.beta)
        dtJ_np = 1 - self.sprime_bar/(1-self.beta)
        dz_np = np.append(dzJ_np, dtJ_np)
        gt.set_local(dz_np)

        return np.sqrt(gt.inner(gt))

    def hessian(self, zhat, Hzhat):
        pass

    def superquantile(self):
        """
        Evaluate the superquantile using the computed samples
        """
        return soupy.sample_superquantile(self.q_samples, self.beta)

