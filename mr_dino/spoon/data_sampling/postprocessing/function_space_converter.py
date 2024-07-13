import dolfin as dl 
import numpy as np 
import matplotlib.pyplot as plt 
import hippylib as hp 
import soupy 

class FunctionExpression(dl.UserExpression):
    def __init__(self, u, **kwargs):
        self.u = u 
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[:] = self.u(x)

    def value_shape(self):
        return self.u.value_shape()

class FunctionSpaceConverter:
    def __init__(self, Vh_old, Vh_new, method='mumps'):
        self.Vh_old = Vh_old 
        self.Vh_new = Vh_new 
        self.u_old = dl.Function(self.Vh_old)
        self.u_old_exp  = FunctionExpression(self.u_old)
        self.u_new = dl.Function(self.Vh_new)
        self.comm_mesh = self.Vh_new.mesh().mpi_comm()

        self.new_trial = dl.TrialFunction(Vh_new)
        self.new_test = dl.TestFunction(Vh_new)
        self.M_new = dl.assemble(dl.inner(self.new_trial, self.new_test) * dl.dx) 
        self.b_new = dl.assemble(dl.inner(self.u_new, self.new_test)*dl.dx)
        self.M_solver = hp.PETScLUSolver(self.comm_mesh, method=method)
        self.M_solver.set_operator(self.M_new)

        self.dim_new = self.Vh_new.dim()
    
    def convert_batch(self, data_old):
        n_data, dim_old = data_old.shape
        data_new = np.zeros((n_data, self.dim_new)) 
        self.u_old.set_allow_extrapolation(True)

        for i in range(n_data):
            self.u_old.vector().set_local(data_old[i])
        
            dl.assemble( dl.inner(self.u_old_exp, self.new_test) * dl.dx, tensor=self.b_new ) 
            self.M_solver.solve(self.u_new.vector(), self.b_new)
            data_new[i] = self.u_new.vector().get_local() 
        return data_new 


    def convert_jacobian(self, jacobian_old):
        dim_old, dim_input = jacobian_old.shape 
        jacobian_new = np.zeros((self.dim_new, dim_input))

        for i_input in range(dim_input):
            self.u_old.vector().set_local(jacobian_old[:, i_input])
        
            dl.assemble( dl.inner(self.u_old_exp, self.new_test) * dl.dx, tensor=self.b_new ) 
            self.M_solver.solve(self.u_new.vector(), self.b_new)
            jacobian_new[:, i_input] = self.u_new.vector().get_local() 

        return jacobian_new 


    def convert_batch_jacobian(self, jacobian_old):
        n_data, dim_old, dim_input = jacobian_old.shape 
        jacobian_new = np.zeros((n_data, self.dim_new, dim_input))

        for i_data in range(n_data):
            for i_input in range(dim_input):
                self.u_old.vector().set_local(jacobian_old[i_data, :, i_input])
            
                dl.assemble( dl.inner(self.u_old_exp, self.new_test) * dl.dx, tensor=self.b_new ) 
                self.M_solver.solve(self.u_new.vector(), self.b_new)
                jacobian_new[i_data, :, i_input] = self.u_new.vector().get_local() 

        return jacobian_new 


def convert_data_to_new_mesh(Vh_old_all, Vh_new_all, data_loader, new_dir_handler):
    m_data, z_data, u_data = data_loader.load_state_data()
    parameter_converter = FunctionSpaceConverter(Vh_old_all[soupy.PARAMETER], Vh_new_all[soupy.PARAMETER])
    state_converter = FunctionSpaceConverter(Vh_old_all[soupy.STATE], Vh_new_all[soupy.STATE])

    m_new = parameter_converter.convert_batch(m_data) 
    z_new = z_data 
    u_new = state_converter.convert_batch(u_data) 

    np.save(new_dir_handler.data_dir + '/m_data.npy', m_new)
    np.save(new_dir_handler.data_dir + '/z_data.npy', z_new)
    np.save(new_dir_handler.data_dir + '/q_data.npy', u_new)
    
    n_data = m_data.shape[0]   

    for i_data in range(n_data):
        jacobian_old = data_loader.load_full_jacobian_data(i_data)
        jacobian_new = state_converter.convert_jacobian(jacobian_old)
        np.save(new_dir_handler.jacobian_dir + '/Jz_data%d.npy' %(i_data), jacobian_new)

def compare_data_on_meshes(data_loader1, data_loader2, sampler1, sampler2, 
                           data_inds=[0,1,10], 
                           jacobian_inds=[0, 1, 5]):
    
    m_fun1 = dl.Function(sampler1.Vh[soupy.PARAMETER])
    z_fun1 = dl.Function(sampler1.Vh[soupy.CONTROL])
    u_fun1 = dl.Function(sampler1.Vh[soupy.STATE])
    m1, z1, u1  = data_loader1.load_state_data()

    m_fun2 = dl.Function(sampler2.Vh[soupy.PARAMETER])
    z_fun2 = dl.Function(sampler2.Vh[soupy.CONTROL])
    u_fun2 = dl.Function(sampler2.Vh[soupy.STATE])
    m2, z2, u2  = data_loader2.load_state_data()

    for data_ind in data_inds:
        m_fun1.vector().set_local(m1[data_ind])
        m_fun2.vector().set_local(m2[data_ind])

        z_fun1.vector().set_local(z1[data_ind])
        z_fun2.vector().set_local(z2[data_ind])

        u_fun1.vector().set_local(u1[data_ind])
        u_fun2.vector().set_local(u2[data_ind])

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        tri = dl.plot(m_fun1)
        plt.colorbar(tri)
        plt.title('Parameter %d' %(data_ind))

        plt.subplot(122)
        tri = dl.plot(m_fun2)
        plt.colorbar(tri)
        plt.title('Parameter %d' %(data_ind))

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.plot(z_fun1.vector().get_local())
        plt.title('Control %d' %(data_ind))

        plt.subplot(122)
        plt.plot(z_fun2.vector().get_local())
        plt.title('Control %d' %(data_ind))


        plt.figure(figsize=(12,6))
        plt.subplot(121)
        tri = dl.plot(u_fun1)
        plt.colorbar(tri)
        plt.title('State %d' %(data_ind))

        plt.subplot(122)
        tri = dl.plot(u_fun2)
        plt.colorbar(tri)
        plt.title('State %d' %(data_ind))

        Jz_1 = data_loader1.load_full_jacobian_data(data_ind)
        Jz_2 = data_loader2.load_full_jacobian_data(data_ind)


        for jacobian_ind in jacobian_inds:
            u_fun1.vector().set_local(Jz_1[:, jacobian_ind])
            u_fun2.vector().set_local(Jz_2[:, jacobian_ind])

            plt.figure(figsize=(12,6))
            plt.subplot(121)
            tri = dl.plot(u_fun1)
            plt.colorbar(tri)
            plt.title('Jacobian col %d sample %d' %(jacobian_ind, data_ind))

            plt.subplot(122)
            tri = dl.plot(u_fun2)
            plt.colorbar(tri)
            plt.title('Jaobian col %d sample  %d' %(jacobian_ind, data_ind))




