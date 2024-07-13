
import tensorflow as tf 
import numpy as np 
import dolfin as dl

import hippylib as hp
import soupy

from .nnCompleteQoI import build_complete_qoi_network


def build_control_gradient_network(nn_qoi):
    """
    Build the control gradient neural network based on an input scalar neural network 
    """

    input_m = nn_qoi.inputs[0]
    input_z = nn_qoi.inputs[1]
    output_qoi = nn_qoi([input_m, input_z])
    qoi_model = tf.keras.Model(inputs=[input_m, input_z], outputs=output_qoi)

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(input_z)
        qout = qoi_model([input_m, input_z])

    # Full batched gradient 
    dqdz = tape.batch_jacobian(qout,input_z)
    dqdz_model = tf.keras.models.Model([input_m,input_z],[dqdz])
    return dqdz_model


class NNCompleteQoIControlModel:
    """
    Interfaces with dolfin and hippylib 
    """
    def __init__(self, Vh, nn_qoi):
        self.nn_qoi = nn_qoi
        self.nn_grad = build_control_gradient_network(nn_qoi)

        self.Vh = Vh 
        self.dM = Vh[soupy.PARAMETER].dim()
        self.dZ = Vh[soupy.CONTROL].dim() 
        self.dQ = Vh[soupy.STATE].dim()

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


    def cost(self, m_np, z_np):
        return self.nn_qoi.predict([m_np, z_np])[:,0]

    def evalGradientControlBatched(self, m_np, z_np):
        """
        m_np, z_np are numpy arrays
        """
        # grads = self.nn_grad.predict([m_np, z_np])
        return self.nn_grad.predict([m_np, z_np])[:,0,:]

    def evalGradientControl(self, m, z, out):
        """
        Evaluates the control gradient given dl.Vector m, z
        """
        m_np = np.expand_dims(m.get_local(), axis=0)
        z_np = np.expand_dims(z.get_local(), axis=0)
        grad_np = self.nn_grad.predict([m_np, z_np])[:,0]
        out.set_local(grad_np)



class NNOperatorQoIControlModel(NNCompleteQoIControlModel):
    """
    Interfaces with dolfin and hippylib 
    """
    def __init__(self, Vh, neural_operator, neural_qoi):
        self.neural_operator = neural_operator
        self.neural_qoi = neural_qoi 
        self.nn_qoi, self.nn_grad =  build_complete_qoi_network(self.neural_operator, self.neural_qoi)

        self.dM = self.neural_operator.dM
        self.dQ = self.neural_operator.dQ
        self.dZ = self.neural_operator.dZ
        self.Vh = Vh 

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
            x = [self.neural_operator.generate_state(),
                 self.neural_operator.generate_parameter(),
                 self.neural_operator.generate_state(),
                 self.neural_operator.generate_control()]
        elif component == soupy.STATE:
            x = self.neural_operator.generate_state()
        elif component == soupy.PARAMETER:
            x = self.neural_operator.generate_parameter()
        elif component == soupy.ADJOINT:
            x = self.neural_operator.generate_state()
        elif component == soupy.CONTROL:
            x = self.neural_operator.generate_control()
        return x


    def cost(self, m_np, z_np):
        return self.nn_qoi.predict([m_np, z_np])[:,0]


    def solveFwd(self, m_np, z_np):
        return self.neural_operator.eval(m_np, z_np)


    def evalGradientControlBatched(self, m_np, z_np):
        """
        m_np, z_np are numpy arrays
        """
        return self.nn_grad.predict([m_np, z_np])

    def evalGradientControl(self, m, z, out):
        """
        Evaluates the control gradient given dl.Vector m, z
        """
        m_np = np.expand_dims(m.get_local(), axis=0)
        z_np = np.expand_dims(z.get_local(), axis=0)
        grad_np = self.nn_grad.predict([m_np, z_np])[:,0]
        out.set_local(grad_np)
