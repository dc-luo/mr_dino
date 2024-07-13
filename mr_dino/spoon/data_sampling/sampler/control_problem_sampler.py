import pickle
import time 
import dolfin as dl 
import numpy as np 
import hippylib as hp 
import hippyflow as hf 
import soupy 

from ..postprocessing import build_mass_matrix_csr, BoundaryRestrictedKLEConstructor

class ControlProblemSampler:
    def __init__(self, Vh, observable, parameter_sampler, control_sampler):
        self.Vh = Vh 
        self.mesh_comm = self.Vh[soupy.STATE].mesh().mpi_comm()
        assert self.mesh_comm.Get_size() == 1
        self.observable = observable
        self.parameter_sampler = parameter_sampler 
        self.control_sampler = control_sampler 

        self.parameter_jacobian = hf.ObservableJacobian(observable)
        self.control_jacobian = hf.ObservableControlJacobian(observable)

        self.x = self.observable.generate_vector()

        self.u = self.x[soupy.STATE]
        self.m = self.x[soupy.PARAMETER]
        self.p = self.x[soupy.ADJOINT]
        self.z = self.x[soupy.CONTROL]

        self.Bu = dl.Vector(self.mesh_comm)
        self.observable.B.init_vector(self.Bu, 0)

        self.M_csr = build_mass_matrix_csr(self.Vh[soupy.STATE])

    
    def sample_data(self, n_data, data_dir, checkpoint_every=100):
        """
        Sample the observable data 
        """
        observable_data = np.zeros((n_data, self.Bu.get_local().shape[0]))
        m_data = np.zeros((n_data, self.m.get_local().shape[0]))
        z_data = np.zeros((n_data, self.z.get_local().shape[0]))

        with open(data_dir + '/M_csr.p', 'wb') as mass_file:
            pickle.dump(self.M_csr, mass_file)

        if hasattr(self.observable.problem, "converged"):
            track_convergence = True 
            converged_data = np.zeros(n_data, dtype=bool)
        else:
            track_convergence = False 

        for i in range(n_data):
            # Sample data
            self.parameter_sampler.sample(self.m)
            self.control_sampler.sample(self.z)
            
            # Solve forward problem 
            t0 = time.time()
            self.observable.solveFwd(self.u, self.x)
            self.observable.applyB(self.u, self.Bu)
            t1 = time.time()
            
            # Save data 
            observable_data[i] = self.Bu.get_local()
            m_data[i] = self.m.get_local()
            z_data[i] = self.z.get_local()
            if track_convergence:
                converged_data[i] = self.observable.problem.converged
            
            if i % checkpoint_every == 0:
                np.save(data_dir + '/m_data.npy', m_data)
                np.save(data_dir + '/z_data.npy', z_data)
                np.save(data_dir + '/q_data.npy', observable_data)
                if track_convergence:
                    np.save(data_dir + '/converged_data.npy', converged_data)

                with open(data_dir + '/checkpoint.txt', 'w') as logfile:
                    print("Saved to sample %d" %(i+1), file=logfile)
            
            print("Sample data %d generated (took %g seconds)" %(i+1, t1 - t0))
        
        # Final save 
        np.save(data_dir + '/m_data.npy', m_data)
        np.save(data_dir + '/z_data.npy', z_data)
        np.save(data_dir + '/q_data.npy', observable_data)
        if track_convergence:
            np.save(data_dir + '/converged_data.npy', converged_data)

        with open(data_dir + '/checkpoint.txt', 'w') as logfile:
            print("Saved to sample %d" %(i+1), file=logfile)



        return m_data, z_data, observable_data  
    

    def load_state_data(self, data_dir):
        m_data = np.load(data_dir + '/m_data.npy')
        z_data = np.load(data_dir + '/z_data.npy')
        q_data = np.load(data_dir + '/q_data.npy')
        return m_data, z_data, q_data 


    def sample_control_jacobian_at_data(self, jacobian_dir, m_data, z_data, u_data):
        n_data = m_data.shape[0] 
        dZ = z_data.shape[1]
        dU = u_data.shape[1] 
        ej = self.observable.generate_vector(soupy.CONTROL)
        du = self.observable.generate_vector(soupy.STATE)

        Jz_data = np.zeros((dU, dZ))

        # Being sampling loop 
        for i in range(n_data):
            self.m.set_local(m_data[i])
            self.z.set_local(z_data[i])
            self.u.set_local(u_data[i])
            
            self.observable.setLinearizationPoint(self.x) 
            for j in range(dZ):
                t0 = time.time()
                ej_np = np.zeros(dZ)
                ej_np[j] = 1.0 
                ej.set_local(ej_np)
                self.control_jacobian.mult(ej, du)
                Jz_data[:,j] = du.get_local()
                t1 = time.time()
                print("Jz in direction %d took %g seconds" %(j, t1 - t0)) 

            np.save(jacobian_dir + '/Jz_data%d.npy' %(i), Jz_data)
            print("Sampled jacobian %d" %(i))


    def sample_mode_jacobian_at_data(self, jacobian_dir, m_data, z_data, u_data, u_basis_np,
            m_basis_np=None, z_basis_np=None,
            control_jacobian=True, parameter_jacobian=True):
        """
        Sample the gradient in direction of u_basis 
        """
        assert control_jacobian or parameter_jacobian 
        n_data = m_data.shape[0] 
        u_basis = hf.dense_to_mv_local(u_basis_np, self.Bu)
        
        JmT_u_basis = None
        JzT_u_basis = None
        
        Bt_u_basis = hp.MultiVector(self.u, u_basis.nvec())
        hp.MatMvTranspmult(self.observable.B, u_basis, Bt_u_basis)
        Ainvt_Bt_u_basis = hp.MultiVector(self.u, u_basis.nvec())

        if parameter_jacobian:
            JmT_u_basis = hp.MultiVector(self.m, u_basis.nvec())

        if control_jacobian:
            JzT_u_basis = hp.MultiVector(self.z, u_basis.nvec())


        # Being sampling loop 
        for i in range(n_data):
            self.m.set_local(m_data[i])
            self.z.set_local(z_data[i])
            self.u.set_local(u_data[i])
            
            self.observable.setLinearizationPoint(self.x) 
            t0 = time.time()
            for i_vec in range(u_basis.nvec()):
                # Solve adjoint incrementals for each rhs 
                self.observable.solveAdjIncremental(Ainvt_Bt_u_basis[i_vec], -1.0 * Bt_u_basis[i_vec])
            t1 = time.time()
            print("Adjoint solves took %g" %(t1 - t0))


            if control_jacobian:
                t1 = time.time()
                hp.MatMvTranspmult(self.observable.problem.Cz, Ainvt_Bt_u_basis, JzT_u_basis)
                JzT_u_basis_np = hf.mv_to_dense(JzT_u_basis)

                if z_basis_np is not None:
                    JzT_u_basis_np = JzT_u_basis_np.T @ z_basis_np
                else:
                    JzT_u_basis_np = JzT_u_basis_np.T
                t2 = time.time()
                print("Control Jacobian postprocessing took %g s" %(t2 - t1))
                np.save(jacobian_dir + '/Jz_data%d.npy' %(i), JzT_u_basis_np)

            if parameter_jacobian:
                t1 = time.time()
                hp.MatMvTranspmult(self.observable.problem.C, Ainvt_Bt_u_basis, JmT_u_basis)

                JmT_u_basis_np = hf.mv_to_dense(JmT_u_basis)
                if m_basis_np is not None:
                    JmT_u_basis_np = JmT_u_basis_np.T @ m_basis_np

                t2 = time.time()
                print("Parameter Jacobian postprocessing took %g s" %(t2 - t1))

                np.save(jacobian_dir + '/Jm_data%d.npy' %(i), JmT_u_basis_np)
            
            print("Sampled Jacobian %d of %d" %(i+1, n_data))


    def time_sampling(self, n_data):
        """
        Times the forward solve, jacobian factorization, and jacobian solve 
        """
        t_fwd = np.zeros(n_data)
        t_jacobian_lu = np.zeros(n_data)
        t_jacobian_apply = np.zeros(n_data)
    
        dz = self.observable.generate_vector(soupy.CONTROL)
        drdz = self.observable.generate_vector(soupy.STATE)
        dudz = self.observable.generate_vector(soupy.STATE)

        dim_control = len(dz.get_local())

        for i in range(n_data):
            # Sample data
            self.parameter_sampler.sample(self.m)
            self.control_sampler.sample(self.z)
            
            # Solve forward problem 
            t_fwd_0 = time.time()
            self.observable.solveFwd(self.u, self.x)
            t_fwd_1 = time.time()

            t_fwd[i] = t_fwd_1 - t_fwd_0 

            t0_linearize = time.time()
            self.observable.setLinearizationPoint(self.x)
            t1_linearize = time.time()
            
            for j in range(dim_control):
                ej = np.zeros(dim_control)
                ej[j] = 1.0 
                dz.set_local(ej)

                t0_jac = time.time()
                self.observable.applyCz(dz, drdz)
                self.observable.solveFwdIncremental(dudz, drdz)
                t1_jac = time.time()

                if j == 0:
                    t_jacobian_lu[i] = t1_jac - t0_jac  + t1_linearize - t0_linearize
                else:
                    t_jacobian_apply[i] += (t1_jac - t0_jac)/(dim_control-1)
            
            print("Sample %d: Solve time %g, Jacobian LU %g, Jacobian apply %g" %(i, t_fwd[i], t_jacobian_lu[i], t_jacobian_apply[i]))

        return t_fwd, t_jacobian_lu, t_jacobian_apply

    def compute_kle_basis(self, rank, basis_dir, orthogonality):
        kle_parameters = hf.KLEParameterList()
        kle_parameters["rank"] = rank
        kle_parameters["output_directory"] = basis_dir + '/'
        kle_constructor = hf.KLEProjector(self.parameter_sampler.prior, 
                                        self.mesh_comm, collective=None, 
                                        parameters=kle_parameters)
        kle_eig, kle_basis, kle_projector = kle_constructor.construct_input_subspace(orthogonality=orthogonality)

        kle_basis = hf.mv_to_dense(kle_basis)
        kle_projector = hf.mv_to_dense(kle_projector)
        kle_shift = self.parameter_sampler.prior.mean.get_local()

        np.save(basis_dir + '/KLE_eigenvalues.npy', kle_eig)
        np.save(basis_dir + '/KLE_basis.npy', kle_basis)
        np.save(basis_dir + '/KLE_projector.npy', kle_projector)
        np.save(basis_dir + '/KLE_shift.npy', kle_shift)


    def compute_pod_basis(self, q_data, rank, shifted, basis_dir, orthogonality='mass'):
        assert orthogonality == 'mass', "Only supports mass weighted for now"
        pod_projector = hf.PODProjectorFromData(self.Vh)
        d, pod_basis, pod_projector, pod_shift = pod_projector.construct_subspace(q_data, rank, shifted=shifted, method='hep')

        n_sample = q_data.shape[0]

        if shifted:
            pod_name = f'POD_shifted_n{n_sample}_r{rank}'
        else:
            pod_name = f'POD_n{n_sample}_r{rank}'


        np.save(basis_dir + f'/{pod_name}_eigenvalues.npy', d)
        np.save(basis_dir + f'/{pod_name}_basis.npy', pod_basis)
        np.save(basis_dir + f'/{pod_name}_projector.npy', pod_projector)
        if shifted:
            np.save(basis_dir + f'/{pod_name}_shift.npy', pod_shift)


    def collect_and_project_jacobian(self, n_pod, rank, shifted, 
                                     data_dir, jacobian_dir, basis_dir):
        if shifted:
            pod_name = f'POD_shifted_n{n_pod}_r{rank}'
        else:
            pod_name = f'POD_n{n_pod}_r{rank}'
        pod_projector = np.load(basis_dir + f'/{pod_name}_projector.npy')
        _, z_data, _ = self.load_state_data(data_dir)
        n_data, dZ = z_data.shape

        projected_jacobian_all = np.zeros((n_data, rank, dZ))

        for i in range(n_data):
            Jz = np.load(jacobian_dir + f'/Jz_data{i}.npy')
            projected_jacobian_all[i] = pod_projector.T @ Jz
        
        np.save(data_dir + f'/{pod_name}_projected_Jz_data.npy', projected_jacobian_all)


class ControlProblemSamplerWithBoundaryKLE(ControlProblemSampler):
    def __init__(self, Vh, observable, parameter_sampler, control_sampler, ds):
        super().__init__(Vh, observable, parameter_sampler, control_sampler)
        self.ds = ds 

    def compute_kle_basis(self, rank, basis_dir, orthogonality):
        assert orthogonality == 'mass' 
        oversampling = 10 
        prior = self.parameter_sampler.prior 
        Vh = self.Vh[soupy.PARAMETER]

        kle_constructor = BoundaryRestrictedKLEConstructor(prior, Vh, self.ds)
        kle_eig, kle_basis, kle_projector = kle_constructor.construct_input_subspace(rank, oversampling, s=1, as_numpy=True)
        kle_shift = self.parameter_sampler.prior.mean.get_local()

        np.save(basis_dir + '/KLE_eigenvalues.npy', kle_eig)
        np.save(basis_dir + '/KLE_basis.npy', kle_basis)
        np.save(basis_dir + '/KLE_projector.npy', kle_projector)
        np.save(basis_dir + '/KLE_shift.npy', kle_shift)
