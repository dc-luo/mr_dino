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

from spoon import ControlNeuralOperator, NNControlModel, \
    NNSuperquantileRiskMeasureSAASettings, NNSuperquantileRiskMeasureSAA, \
    NNSparseL2QoI, NNOperatorQoIControlModel


import scipy.stats

def standardNormalSuperquantile(beta):
    quantile = scipy.stats.norm.ppf(beta)
    return np.exp(-quantile**2/2)/(1-beta)/np.sqrt(2*np.pi)

class TestNNMeanVarRiskMeasure(unittest.TestCase):
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


    def finiteDifferenceCheck(self, model, n_sample, presample):
        print("Test case: %d sample size" %(n_sample))
        # 2. Setting up prior
        GAMMA = 0.1
        DELTA = 5.0
        STRENGTH_UPPER = 3.0
        STRENGTH_LOWER = -3.0
        np.random.seed(1)

        m_mean_fun = dl.Function(self.Vh[soupy.PARAMETER])
        m_mean_fun.interpolate(dl.Constant(0.0))
        prior = hp.BiLaplacianPrior(self.Vh[soupy.PARAMETER], GAMMA, DELTA,
                                    mean=m_mean_fun.vector(), robin_bc=True)
        noise = dl.Vector()
        prior.init_vector(noise, "noise")


        rm_settings = NNSuperquantileRiskMeasureSAASettings()
        rm_settings["beta"] = 0.9
        rm_settings["sample_size"] = n_sample

        if presample:
            m_samples = np.random.randn(n_sample, len(prior.mean.get_local()))
            risk = NNSuperquantileRiskMeasureSAA(model, prior, rm_settings, m_samples)
        else:
            risk = NNSuperquantileRiskMeasureSAA(model, prior, rm_settings)

        zt0 = risk.generate_vector(soupy.CONTROL)
        dzt = risk.generate_vector(soupy.CONTROL)
        zt1 = risk.generate_vector(soupy.CONTROL)
        gt0 = risk.generate_vector(soupy.CONTROL)
        
        zt0.set_local(np.random.randn(len(zt0.get_local())))
        dzt.set_local(np.random.randn(len(zt0.get_local())))


        zt1.axpy(1.0, zt0)
        zt1.axpy(self.delta, dzt)


        risk.computeComponents(zt0, order=1)

        c0 = risk.cost()
        risk.grad(gt0)

        risk.computeComponents(zt1, order=0)
        c1 = risk.cost()

        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = gt0.inner(dzt)
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



    def computeSuperquantileValue(self, sample_size, beta):
        GAMMA = 0.1
        DELTA = 5.0
        STRENGTH_UPPER = 3.0
        STRENGTH_LOWER = -3.0
        np.random.seed(1)

        m_mean_fun = dl.Function(self.Vh[soupy.PARAMETER])
        m_mean_fun.interpolate(dl.Constant(0.0))
        prior = hp.BiLaplacianPrior(self.Vh[soupy.PARAMETER], GAMMA, DELTA,
                                    mean=m_mean_fun.vector(), robin_bc=True)
        noise = dl.Vector()
        prior.init_vector(noise, "noise")


        rm_settings = NNSuperquantileRiskMeasureSAASettings()
        rm_settings["beta"] = beta
        rm_settings["sample_size"] = sample_size
        risk = NNSuperquantileRiskMeasureSAA(self.completeQoIControlModel, prior, rm_settings)
        np.random.seed(1)
        risk.q_samples = np.random.randn(len(risk.q_samples))
        sq = risk.superquantile()
        return sq
    

    def testSuperquantileValue(self):
        sample_size = 10000
        beta = 0.2
        sq_normal = standardNormalSuperquantile(beta)
        sq = self.computeSuperquantileValue(sample_size, beta)
        print("Computed superquantile: ", sq)
        print("Analytic superquantile: ", sq_normal)
        tol = 1e-2
        self.assertTrue(np.abs(sq_normal - sq) < tol)


if __name__ == "__main__":
    unittest.main()
