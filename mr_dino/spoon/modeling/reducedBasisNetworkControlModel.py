import tensorflow as tf
import numpy as np
import dolfin as dl
import scipy.sparse as sp

import soupy
from .nnCompleteQoI import build_complete_qoi_network
from .controlNeuralOperator import ControlNeuralOperator
from .nnCompleteQoIControlModel import NNCompleteQoIControlModel


class ReducedBasisNetworkQoIControlModel(NNCompleteQoIControlModel):
    """
    Interfaces with dolfin and hippylib
    """
    def __init__(self, Vh, rbnet, rbqoi):
        self.rbnet = rbnet
        self.reduced_neural_operator = ControlNeuralOperator(Vh, self.rbnet.reduced_network)
        self.rbqoi = rbqoi
        self.nn_qoi, self.nn_grad = build_complete_qoi_network(self.reduced_neural_operator, self.rbqoi)

        self.Vh = Vh
        self.dQ = self.Vh[soupy.STATE].dim()
        self.dM = self.Vh[soupy.PARAMETER].dim()
        self.dZ = self.Vh[soupy.CONTROL].dim()


    def generate_state(self):
        return dl.Function(self.Vh[soupy.STATE]).vector()

    def generate_parameter(self):
        return dl.Function(self.Vh[soupy.PARAMETER]).vector()

    def generate_control(self):
        return dl.Function(self.Vh[soupy.CONTROL]).vector()

    def generate_vector(self, component = "ALL"):
        """
        By default, return the list :code:`[u,m,p,z]` where:

            - :code:`u` is any object that describes the state variable
            - :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
            (Needs to support linear algebra operations)
            - :code:`p` is any object that describes the adjoint variable
            - :code:`z` is any object that describes the control variable

        If :code:`component = STATE` return only :code:`u`

        If :code:`component = PARAMETER` return only :code:`m`

        If :code:`component = ADJOINT` return only :code:`p`
        If :code:`component = CONTROL` return only :code:`z`
        """
        if component == "ALL":
            x = [self.generate_state(),
                 self.generate_parameter(),
                 self.generate_state(),
                 self.generate_control()]
        elif component == soupy.STATE:
            x = self.generate_state()
        elif component == soupy.PARAMETER:
            x = self.generate_parameter()
        elif component == soupy.ADJOINT:
            x = self.generate_state()
        elif component == soupy.CONTROL:
            x = self.generate_control()
        return x


    def cost(self, reduced_m_np, z_np):
        return self.nn_qoi.predict([reduced_m_np, z_np])[:,0]


    def solveFwd(self, m_np, z_np):
        u_np = self.rbnet.full_network.predict([m_np, z_np])
        return u_np

    def project_m(self, m_np):
        return self.rbnet.project_m(m_np)

    def evalGradientControlBatched(self, reduced_m_np, z_np):
        """
        m_np, z_np are numpy arrays
        """
        return self.nn_grad.predict([reduced_m_np, z_np])

    def evalGradientControl(self, m, z, out):
        """
        Evaluates the control gradient given dl.Vector m, z
        """
        m_np = np.expand_dims(m.get_local(), axis=0)
        reduced_m_np = self.project_m(m_np)
        z_np = np.expand_dims(z.get_local(), axis=0)
        grad_np = self.nn_grad.predict([reduced_m_np, z_np])[:,0]
        out.set_local(grad_np)