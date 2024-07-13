import sys, os
import math 
import dolfin as dl 
import numpy as np 
from mpi4py import MPI

import hippylib as hp
import soupy 
import mr_dino.spoon as spoon


def BiLaplacian2D(Vh_parameter,gamma = 0.1,delta = 0.1, theta0 = 2.0, theta1 = 0.5,\
             alpha = np.pi/4,mean = None,robin_bc = False):
    """
    Return 2D BiLaplacian prior given function space and coefficients for Matern covariance
    """
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.theta0 = theta0
    anis_diff.theta1 = theta1
    anis_diff.alpha = alpha

    return hp.BiLaplacianPrior(Vh_parameter, gamma, delta, anis_diff,mean = mean,robin_bc = robin_bc)


def poisson_control_settings():
	settings = {}
	settings['nx'] = 64
	settings['ny'] = 64
	settings['state_order'] = 1
	settings['parameter_order'] = 1

	settings['STRENGTH_UPPER'] = 4.
	settings['STRENGTH_LOWER'] = -4.
	settings['LINEAR'] = False


	settings['N_WELLS_PER_SIDE'] = 7
	settings['LOC_LOWER'] = 0.1
	settings['LOC_UPPER'] = 0.9
	settings['WELL_WIDTH'] = 0.08

	settings['GAMMA'] = 0.1
	settings['DELTA'] = 5.0
	settings['THETA0'] = 2.0
	settings['THETA1'] = 0.5
	settings['ALPHA'] = math.pi/4

	return settings


class PoissonVarfHandler:
	"""
	"""
	def __init__(self,Vh, settings):
		"""
		"""
		self.linear = settings['LINEAR']

		# Set up the control right hand side
		well_grid = np.linspace(settings['LOC_LOWER'],settings['LOC_UPPER'],settings['N_WELLS_PER_SIDE'])
		well_grid_x, well_grid_y = np.meshgrid(well_grid, well_grid)
		mollifier_list = [] 

		for i in range(settings['N_WELLS_PER_SIDE']):
			for j in range(settings['N_WELLS_PER_SIDE']):
				mollifier_list.append(
						dl.interpolate(dl.Expression("a*exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(b,2)))", 
							xi=well_grid[i], 
							yj=well_grid[j], 
							a=1/(2*math.pi*settings['WELL_WIDTH']**2),
							b=settings['WELL_WIDTH'],
							degree=2),
                            Vh[soupy.STATE])
						)

		self.mollifiers = dl.as_vector(mollifier_list)
		assert Vh[soupy.CONTROL].dim() == len(mollifier_list)

	def __call__(self,u,m,p,z):
		if self.linear:
			return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx - dl.inner(self.mollifiers,z)*p*dl.dx
		else:
			return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx + u**3*p*dl.dx  - dl.inner(self.mollifiers,z)*p*dl.dx



class PoissonPDEProblemForTiming(soupy.PDEVariationalControlProblem):
    """
    Poisson PDE problem for timing --- more light weight 
    """
    def __init__(self, Vh, varf_handler, bc, bc0, is_fwd_linear = False, lu_method="default"):
        super().__init__(Vh, varf_handler, bc, bc0, is_fwd_linear=is_fwd_linear, lu_method=lu_method)
        print("Using this for timing")

    def setLinearizationPoint(self, x, gauss_newton):
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """
            
        x_fun = [hp.vector2Function(x[i], self.Vh[i]) for i in range(4)]
        
        f_form = self.varf_handler(*x_fun)
        g_form = [None, None, None, None] 
        g_form[soupy.ADJOINT] = dl.derivative(f_form, x_fun[soupy.ADJOINT])
            
        self.A, dummy = dl.assemble_system(dl.derivative(g_form[soupy.ADJOINT],x_fun[soupy.STATE]), g_form[soupy.ADJOINT], self.bc0)
        self.Cz = dl.assemble(dl.derivative(g_form[soupy.ADJOINT],x_fun[soupy.CONTROL]))
        [bc.zero(self.Cz) for bc in self.bc0]
                
        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
        self.solver_fwd_inc.set_operator(self.A)



def build_poisson_problem(settings, comm_mesh=MPI.COMM_WORLD, timing=False):
    assert comm_mesh.Get_size() == 1
    mesh = dl.UnitSquareMesh(comm_mesh, settings['nx'], settings['ny'])
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', settings["state_order"])
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', settings["parameter_order"])
    n_control = settings['N_WELLS_PER_SIDE']**2
    Vh_CONTROL = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=n_control)

    Vh = [Vh2, Vh1, Vh2, Vh_CONTROL]

    def u_boundary(x, on_boundary):
        return on_boundary 

    u_bdr = dl.Constant(0.0)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[soupy.STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[soupy.STATE], u_bdr0, u_boundary)
    pde_varf = PoissonVarfHandler(Vh, settings = settings)

    if timing:
        pde = PoissonPDEProblemForTiming(Vh, pde_varf, bc, bc0, is_fwd_linear=settings['LINEAR'], lu_method="default")
    else:
        pde = soupy.PDEVariationalControlProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=settings['LINEAR'], lu_method="default")

    m_mean_fun = dl.Function(Vh[hp.PARAMETER])
    m_mean_fun.interpolate(dl.Constant(-1.0))
    prior = BiLaplacian2D(Vh[hp.PARAMETER], mean = m_mean_fun.vector(),
        gamma = settings['GAMMA'],
        delta = settings['DELTA'],
        theta0 = 2.0, theta1 = 0.5,alpha = np.pi/4)

    # Add the control distribution here
    control_distribution = spoon.UniformDistribution(Vh[soupy.CONTROL],
        settings['STRENGTH_LOWER'],settings['STRENGTH_UPPER'])

    return mesh, pde, Vh, prior, control_distribution




