import unittest
import dolfin as dl
import numpy as np 
import matplotlib.pyplot as plt 

import tensorflow as tf 
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import sys
import os

sys.path.append( os.environ.get('HIPPYLIB_PATH'))
sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
sys.path.append(os.environ.get('SOUPY_PATH'))
sys.path.append("../../../")

import hippylib as hp
import soupy

from buildMultiInputNN import build_multi_input_network
from spoon import ControlNeuralOperator, NNControlModel, NNSparseL2QoI, NNOperatorQoIControlModel, \
    NNMeanVarRiskMeasureSAA, NNMeanVarRiskMeasureSAASettings, UniformDistribution


class TestNNMeanVarRiskMeasureSAA(unittest.TestCase):
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
            return u**2*dl.dx + m*u*dl.dx + dl.inner(z, z) * dl.dx

        self.qoi = soupy.VariationalControlQoI(self.Vh, l2norm)
        nn = build_multi_input_network(dM, dZ, dQ, 1, 1, 50)
        self.neural_operator = ControlNeuralOperator(self.Vh, nn)
        self.control_model = NNControlModel(self.Vh, self.neural_operator, self.qoi)

        # Set up the monolithic networks 
        u_trial = dl.TrialFunction(Vh_STATE)
        u_test = dl.TestFunction(Vh_STATE)
        W = dl.assemble(u_trial * u_test * dl.dx) 
        d = dl.Function(self.Vh[hp.STATE]).vector()
        self.neural_qoi = NNSparseL2QoI(W, d)
        self.completeQoIControlModel = NNOperatorQoIControlModel(self.Vh, self.neural_operator, self.neural_qoi)

        self.delta = 1e-3
        self.fdtol = 5e-2
        self.reltol = 1e-3

    # def compareSAA(self, model):
    #     BETA = 0.5
    #     N_SAMPLE = 10
    #     N_BATCH = 100
    #     GAMMA = 0.1
    #     DELTA = 5.0
    #     STRENGTH_UPPER = 10.0
    #     STRENGTH_LOWER = -10.0

    #     np.random.seed(1)

    #     m_mean_fun = dl.Function(self.Vh[soupy.PARAMETER])
    #     m_mean_fun.interpolate(dl.Constant(0.0))
    #     prior = hp.BiLaplacianPrior(self.Vh[soupy.PARAMETER], GAMMA, DELTA,
    #                                 mean=m_mean_fun.vector(), robin_bc=True)
    #     noise = dl.Vector()
    #     prior.init_vector(noise, "noise")
    #     control_dist = UniformDistribution(self.Vh[soupy.CONTROL], STRENGTH_LOWER, STRENGTH_UPPER)

    #     rm_saa_settings = NNMeanVarRiskMeasureSAASettings()
    #     rm_saa_settings["beta"] = BETA
    #     rm_saa_settings["sample_size"] = N_SAMPLE
    #     riskSAA = NNMeanVarRiskMeasureSAA(model, prior, rm_saa_settings)

    #     rm_settings = NNMeanVarRiskMeasureSettings()
    #     rm_settings["beta"] = BETA
    #     rm_settings["batch_size"] = N_BATCH 
    #     riskSA = NNMeanVarRiskMeasure(model, prior, rm_settings)

    #     # Compute cost 
    #     z = model.generate_vector(soupy.CONTROL)
    #     control_dist.sample(z)

    #     rng = hp.Random(seed=rm_saa_settings['seed'])
    #     riskSA.computeComponents(z, order=1, sample_size=N_SAMPLE, rng=rng)
    #     riskSAA.computeComponents(z, order=1)

    #     cost_sa = riskSA.cost()
    #     cost_saa = riskSAA.cost()

    #     print("Compare cost")
    #     print("SA: %g" %cost_sa)
    #     print("SAA: %g" %cost_saa)

    #     self.assertTrue(abs((cost_sa - cost_saa)/cost_sa) < self.reltol)


    # def testSAA(self):
    #     for model in [self.control_model, self.completeQoIControlModel]:
    #         self.compareSAA(model)


    def testPresampling(self):
        n_sample = 100 
        GAMMA = 0.1
        DELTA = 5.0
        STRENGTH_UPPER = 10.0
        STRENGTH_LOWER = -10.0
        np.random.seed(1)

        m_mean_fun = dl.Function(self.Vh[soupy.PARAMETER])
        m_mean_fun.interpolate(dl.Constant(0.0))
        prior = hp.BiLaplacianPrior(self.Vh[soupy.PARAMETER], GAMMA, DELTA,
                                    mean=m_mean_fun.vector(), robin_bc=True)
        noise = dl.Vector()
        prior.init_vector(noise, "noise")

        control_dist = UniformDistribution(self.Vh[soupy.CONTROL], STRENGTH_LOWER, STRENGTH_UPPER)

        rm_settings = NNMeanVarRiskMeasureSAASettings()
        rm_settings["beta"] = 0.5
        rm_settings["sample_size"] = n_sample

        risk_no_sample = NNMeanVarRiskMeasureSAA(self.control_model, prior, rm_settings)
        risk_with_sample = NNMeanVarRiskMeasureSAA(self.control_model, prior, rm_settings, m_samples=risk_no_sample.m_data)

        # Compute cost 
        z = self.control_model.generate_vector(soupy.CONTROL)
        control_dist.sample(z)
        risk_no_sample.computeComponents(z, order=1)
        risk_with_sample.computeComponents(z, order=1)

        cost_no_sample = risk_no_sample.cost()
        cost_with_sample = risk_with_sample.cost()

        print("Compare cost")
        print("No input samples: %g" %cost_no_sample)
        print("With input samples: %g" %cost_with_sample)
        self.assertTrue(abs((cost_no_sample - cost_with_sample)/cost_no_sample) < self.reltol)


    def finiteDifferenceCheck(self, model, n_sample, presample):
        print("Test case: %d sample size" %(n_sample))
        # 2. Setting up prior
        GAMMA = 0.1
        DELTA = 5.0
        STRENGTH_UPPER = 10.0
        STRENGTH_LOWER = -10.0
        np.random.seed(1)

        m_mean_fun = dl.Function(self.Vh[soupy.PARAMETER])
        m_mean_fun.interpolate(dl.Constant(0.0))
        prior = hp.BiLaplacianPrior(self.Vh[soupy.PARAMETER], GAMMA, DELTA,
                                    mean=m_mean_fun.vector(), robin_bc=True)
        noise = dl.Vector()
        prior.init_vector(noise, "noise")

        control_dist = UniformDistribution(self.Vh[soupy.CONTROL], STRENGTH_LOWER, STRENGTH_UPPER)

        rm_settings = NNMeanVarRiskMeasureSAASettings()
        rm_settings["beta"] = 0.5
        rm_settings["sample_size"] = n_sample

        if presample:
            m_samples = np.random.randn(n_sample, len(prior.mean.get_local()))
            risk = NNMeanVarRiskMeasureSAA(model, prior, rm_settings, m_samples)
        else:
            risk = NNMeanVarRiskMeasureSAA(model, prior, rm_settings)

        z0 = model.generate_vector(soupy.CONTROL)
        dz = model.generate_vector(soupy.CONTROL)
        z1 = model.generate_vector(soupy.CONTROL)
        g0 = model.generate_vector(soupy.CONTROL)

        control_dist.sample(z0)
        control_dist.sample(dz)
        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)

        risk.computeComponents(z0, order=1)

        c0 = risk.cost()
        risk.grad(g0)

        risk.computeComponents(z1, order=0)
        c1 = risk.cost()

        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = g0.inner(dz)
        print("Initial cost: ", c0)
        print("New cost: ", c1)
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        print("Error: ", abs((dcdz_fd - dcdz_ad)/dcdz_ad))
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)

    def testFiniteDifference(self):
        do_presample = [True, False]

        for presample in do_presample:
            for model in [self.control_model, self.completeQoIControlModel]:
                for n_sample in [1, 10, 100]:
                    self.finiteDifferenceCheck(model, n_sample, presample)


if __name__ == "__main__":
    unittest.main()
