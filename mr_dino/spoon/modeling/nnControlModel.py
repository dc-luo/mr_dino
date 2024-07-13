import tensorflow as tf 
import numpy as np 
import dolfin as dl

import hippylib as hp
import soupy

class NNQoIWrapperGeneric:
    def __init__(self, Vh, qoi):
        self.Vh = Vh 
        self.qoi = qoi 
        self.m = dl.Function(Vh[soupy.PARAMETER]).vector()
        self.u = dl.Function(Vh[soupy.STATE]).vector()
        self.z = dl.Function(Vh[soupy.CONTROL]).vector()

        self.x = [self.u, self.m, None, self.z]
        self.dQ = self.u.get_local().shape[0]
        self.dM = self.m.get_local().shape[0]
        self.dZ = self.z.get_local().shape[0]

        self.dims = [self.dQ, None, self.dM, self.dZ]
    
    # def eval(self, x):
        # return self.qoi.eval(x)

    def eval_np(self, u_np, m_np, z_np):
        """
        If u_np, m_np, z_np are single numpy arrays, returns one evaluation 
        of the qoi 

        If u_np, m_np, z_np are numpy arrays with (N, dQ), (N, dM), (N, dZ),
        returns a vector q_np with shape (N, ) of evaluations
        """

        # Some checks to be figured out for distinguishing single computation and batch

        if len(u_np.shape) == 1 and len(m_np.shape) == 1 and len(z_np.shape) == 1:
            # u_np, m_np, z_np are single numpy arrays
            self.u.set_local(u_np)
            self.m.set_local(m_np)
            self.z.set_local(z_np)
            return self.qoi.cost(self.x)

        else:
            N = u_np.shape[0]
            q_np = np.zeros(N)
            for i in range(N):
                self.u.set_local(u_np[i,:])
                self.m.set_local(m_np[i,:])
                self.z.set_local(z_np[i,:])
                q_np[i] = self.qoi.cost(self.x)
            return q_np 

    def grad_np(self, ind, u_np, m_np, z_np):
        """
        If u_np, m_np, z_np are single numpy arrays, returns one evaluation 
        of the qoi gradient shaped (dQ, )

        If u_np, m_np, z_np are numpy arrays with (N, dQ), (N, dM), (N, dZ),
        returns a vector qoi gradients shaped (N, dQ)
        """

        # Some checks to be figured out for distinguishing single computation and batch
        g = dl.Function(self.Vh[ind]).vector()

        if len(u_np.shape) == 1 and len(m_np.shape) == 1 and len(z_np.shape) == 1:
            self.u.set_local(u_np)
            self.m.set_local(m_np)
            self.z.set_local(z_np)
            self.qoi.grad(ind, self.x, g)
            return g.get_local()

        else:
            N_batch = u_np.shape[0]
            hat_np = np.zeros((N_batch, self.dims[ind]))

            for i in range(N_batch):
                # Set vectors and evaluate rhs 
                self.u.set_local(u_np[i,:])
                self.m.set_local(m_np[i,:])
                self.z.set_local(z_np[i,:])
                self.qoi.grad(ind, self.x, g)
                hat_np[i,:] = g.get_local()
            return hat_np

    # def eval_np_batched(self, u_np, m_np, z_np):
    #     """
    #     u_np, m_np, z_np are numpy arrays analogous to inputs to NNs
    #     """
    #     N = u_np.shape[0]
    #     q_np = np.zeros(N)
    #     for i in range(N):
    #         q_np[i] = self.eval_np(u_np[i,:], m_np[i,:], z_np[i,:])
    #     return q_np 


    # def grad_np_batched(self, ind, u_np, m_np, z_np):
    #     N_batch = u_np.shape[0]
    #     hat_np = np.zeros((N_batch, self.dQ))

    #     for i in range(N_batch):
    #         # Set vectors and evaluate rhs 
    #         self.u.set_local(u_np[i,:])
    #         self.m.set_local(m_np[i,:])
    #         self.z.set_local(z_np[i,:])
    #         self.qoi.grad(ind, self.x, self.qoi_grad)
    #         hat_np[i,:] = self.qoi_grad.get_local()
    #     return hat_np



class NNControlModel:
    """
    Interfaces with dolfin and hippylib 
    """
    def __init__(self, Vh, neural_operator, qoi):
        self.neural_operator = neural_operator
        self.qoi = qoi 
        self.nn_qoi_wrapper = NNQoIWrapperGeneric(Vh, qoi)
        # self.qoi_grad = dl.Function(Vh[hp.STATE]).vector()

        self.dM = self.neural_operator.dM
        self.dQ = self.neural_operator.dQ
        self.dZ = self.neural_operator.dZ
        self.Vh = Vh 

        # self.m = dl.Function(Vh[hp.PARAMETER]).vector()
        # self.u = dl.Function(Vh[hp.STATE]).vector()
        # self.z = dl.Function(Vh[soupy.CONTROL])
        # self.x = [self.u, self.m, None, self.z]

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
        # Need to deal with what types of input are allowed
        # Should we always just stick to the correct kind of input?
        u_np = self.neural_operator.eval(m_np, z_np)
        return self.nn_qoi_wrapper.eval_np(u_np, m_np, z_np)


    def solveFwd(self, m_np, z_np):
        return self.neural_operator.eval(m_np, z_np)


    def evalGradientControlBatched(self, m_np, z_np):
        """
        m_np, z_np are numpy arrays
        """
        N_batch = m_np.shape[0]
        u_np = self.neural_operator.eval(m_np, z_np)
        qhat_np = self.nn_qoi_wrapper.grad_np(hp.STATE, u_np, m_np, z_np)
        du_np = self.neural_operator.Jztvp(m_np, z_np, qhat_np)

        # also need gradient w.r.t. z. How do we check to bypass this 
        dz_np = self.nn_qoi_wrapper.grad_np(soupy.CONTROL, u_np, m_np, z_np)
        
        return du_np + dz_np 

    # def cost(self, x):
    #     return self.qoi.eval(x)

    # def cost(self, m, z):
    #     m_np = np.expand_dims(m.get_local(), axis=0)
    #     z_np = np.expand_dims(z.get_local(), axis=0)
    #     u_np = self.neural_operator.eval(m_np, z_np)
    #     return self.qoi.eval

    def evalGradientControl(self, m, z, out):
        """
        Evaluates the control gradient given dl.Vector m, z
        """
        m_np = np.expand_dims(m.get_local(), axis=0)
        z_np = np.expand_dims(z.get_local(), axis=0)
        u_np = self.neural_operator.eval(m_np, z_np)
        self.u.set_local(u_np[0,:])
        self.m.set_local(m_np[0,:])
        self.z.set_local(z_np[0,:])

        # Form dQ/du. Maybe need a mass matrix in here? 
        self.qoi.grad(hp.STATE, self.x, self.qoi_grad)
        grad_np = self.neural_operator.Jztvp(m_np, z_np, self.qoi_grad.get_local())
        out.set_local(grad_np)
