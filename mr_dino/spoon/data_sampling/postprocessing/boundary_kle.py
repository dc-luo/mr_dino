import dolfin as dl 
import numpy as np
import hippylib as hp 
import hippyflow as hf 


def make_boundary_restricted_mass_matrix(Vh, ds, fill_nullspace=False):
    Vh = Vh 
    ds = ds 
    u_trial = dl.TrialFunction(Vh)
    u_test = dl.TestFunction(Vh)
    boundary_mass = dl.assemble(u_trial*u_test*ds)


    if fill_nullspace:
        diag = dl.Function(Vh).vector()
        boundary_mass.get_diagonal(diag)

        new_diag = np.isclose(diag.get_local(), 0.0)
        new_diag = new_diag.astype(np.float64)

        diag.set_local(new_diag)
        diag.apply('')

        nullspace_identity = dl.assemble(u_trial * u_test * dl.dx)
        nullspace_identity.zero()
        nullspace_identity.set_diagonal(diag)

        boundary_mass += nullspace_identity

    return boundary_mass  

class MassPreconditionedCovarianceOperator:
	def __init__(self, C, M):
		"""
		Linear operator representing the mass matrix preconditioned
		covariance matrix :math:`M C M`
		"""
		self.C = C 
		self.M = M 
		self.mpi_comm = self.M.mpi_comm()


		self.Mx = dl.Vector(self.mpi_comm)
		self.CMx = dl.Vector(self.mpi_comm)
		self.M.init_vector(self.Mx, 0)
		self.M.init_vector(self.CMx, 0)

	def init_vector(self,x,dim):
		self.M.init_vector(x,dim)

	def mult(self, x, y):
		self.M.mult(x, self.Mx)
		self.C.mult(self.Mx, self.CMx)
		self.M.mult(self.CMx, y)


class BoundaryRestrictedKLEConstructor:
    def __init__(self, prior, Vh, ds):
        self.prior = prior 
        self.Vh = Vh 
        self.ds = ds 

        self.M = make_boundary_restricted_mass_matrix(Vh, ds, fill_nullspace=False)
        self.B = make_boundary_restricted_mass_matrix(Vh, ds, fill_nullspace=True)

        self.covariance = hp.Solver2Operator(self.prior.Rsolver)
        self.mass_preconditioned_covariance = MassPreconditionedCovarianceOperator(self.covariance, self.M)

        self.Bsolver = hp.PETScLUSolver(comm=self.Vh.mesh().mpi_comm(), method='mumps') 
        self.Bsolver.set_operator(self.B)

    def construct_input_subspace(self, rank, oversampling, s=1, as_numpy=True):
        m_dummy = dl.Function(self.Vh).vector()
        Omega = hp.MultiVector(m_dummy, rank + oversampling)
        hp.parRandom.normal(1.0, Omega)

        d_KLE, V_KLE = hp.doublePassG(self.mass_preconditioned_covariance, self.B, self.Bsolver, Omega, rank, s=s)

        MV_KLE = hp.MultiVector(m_dummy, rank)
        hp.MatMvMult(self.M, V_KLE, MV_KLE)

        if as_numpy:
            V_np = hf.mv_to_dense(V_KLE)
            MV_np = hf.mv_to_dense(MV_KLE)
            return d_KLE, V_np, MV_np
        else:
            return d_KLE, V_KLE, MV_KLE


