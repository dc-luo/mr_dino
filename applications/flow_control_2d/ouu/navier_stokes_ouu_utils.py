import time
import os, sys
import pickle
import argparse

import numpy as np 
import matplotlib.pyplot as plt
import dolfin as dl

import hippylib as hp
import soupy



def L2VelocityPenalization(Vh, basis_all, geometry, alpha):
    basis = dl.as_vector(basis_all)
    z_trial = dl.TrialFunction(Vh[soupy.CONTROL])
    z_test = dl.TestFunction(Vh[soupy.CONTROL])

    M_form = dl.inner(dl.dot(z_trial, basis), dl.dot(z_test, basis)) * geometry.ds(geometry.OBSTACLE)
    M = dl.assemble(M_form)
    return soupy.WeightedL2Penalization(Vh, M, alpha)


def observation_box(mesh, geo_specs, observation_type):
    if observation_type == "full_back":
        body_end = geo_specs['center'][0] + geo_specs['radius']
        chi = dl.Expression("x[0] > d", d=body_end, degree=1, mpi_comm=mesh.mpi_comm())
        print("full back observation")

    elif observation_type == "windowed_back":
        body_end = geo_specs['center'][0] + geo_specs['radius']
        box_top = geo_specs['center'][1] + 3 * geo_specs['radius']
        box_bot = geo_specs['center'][1] - 3 * geo_specs['radius']
        chi = dl.Expression("(x[0] > d) && (x[1] > yl) && (x[1] < yu) ", d=body_end, yl=box_bot, yu=box_top, degree=1,
                mpi_comm=mesh.mpi_comm())

        print("windowed back observation")
    elif observation_type == "windowed_body":
        body_front = geo_specs['center'][0] - 3 * geo_specs['radius']
        box_top = geo_specs['center'][1] + 3 * geo_specs['radius']
        box_bot = geo_specs['center'][1] - 3 * geo_specs['radius']
        chi = dl.Expression("(x[0] > d) && (x[1] > yl) && (x[1] < yu) ", d=body_front, yl=box_bot, yu=box_top, degree=1,
                mpi_comm=mesh.mpi_comm())

        print("windowed body observation")
    else:
        chi = dl.Constant(1.0)
        print("full observation")

    return chi


def setup_qoi(qoi_type, observation_type, mesh, Vh, geo_specs, settings):
    if qoi_type== "dissipation":
        chi = observation_box(mesh, geo_specs, observation_type)
        qoi = StrainEnergyQoI(Vh, settings["nu"], chi=chi)
    elif qoi_type== "tracking":
        v_target = [1.0, 0.0]
        chi = observation_box(mesh, geo_specs, observation_type)
        qoi = VelocityTrackingQoI(Vh, v_target=v_target, chi=chi)

    elif qoi_type == "tracking_up":
        v_target = [1.0, 0.5]
        chi = observation_box(mesh, geo_specs, observation_type)
        qoi = VelocityTrackingQoI(Vh, v_target=v_target, chi=chi)
    else:
        raise ValueError("no qoi type of this kind implemented")
    return qoi


class StrainEnergyQoI(soupy.ControlQoI):
    """
    Defines the strain energy qoi integrated over the domain 
    """
    def __init__(self, Vh, nu, chi=None):
        """
        Constructor.
        INPUTS:
        - mesh: the mesh
        - Vh: the finite element space for [state, parameter, adjoint, optimization] variable
            Note that the state space is the joint space for velocity and pressure, (v, p)
        """
        self.nu = nu 
        self.Vh = Vh
        self.chi = chi 
        self.x = [dl.Function(Vh[soupy.STATE]).vector(), dl.Function(Vh[soupy.PARAMETER]).vector(),
                  dl.Function(Vh[soupy.ADJOINT]).vector(), dl.Function(Vh[soupy.CONTROL]).vector()]
        self.x_test = [dl.TestFunction(Vh[soupy.STATE]), dl.TestFunction(Vh[soupy.PARAMETER]),
                       dl.TestFunction(Vh[soupy.ADJOINT]), dl.TestFunction(Vh[soupy.CONTROL])]

        self.u = dl.Function(Vh[soupy.STATE]).vector()
        self.Ku = dl.Function(Vh[soupy.STATE]).vector()

        u_trial = dl.TrialFunction(Vh[soupy.STATE])
        u_test = dl.TestFunction(Vh[soupy.STATE])
        v_trial, p_trial = dl.split(u_trial)
        v_test, p_test = dl.split(u_test)
        if self.chi is None:
            self.K = dl.assemble(dl.Constant(2 * self.nu) * dl.inner(self._strain(v_trial), self._strain(v_test))*dl.dx)
        else:
            self.K = dl.assemble(dl.Constant(2 * self.nu) * chi * dl.inner(self._strain(v_trial), self._strain(v_test))*dl.dx)

    def _strain(self, v_fun):
        return dl.sym(dl.grad(v_fun))

    def cost(self, x):
        """
        evaluate the qoi at given x
        :param x: [state, parameter, adjoint, optimization] variable
        :return: qoi(x)
        """
        self.K.mult(x[soupy.STATE], self.Ku)
        return self.Ku.inner(x[soupy.STATE])

    def adj_rhs(self, x, rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
        with respect to the state u).
        INPUTS:
        - x coefficient vector of all variables
        - rhs: FEniCS vector to store the rhs for the adjoint problem.
        """
        self.grad(soupy.STATE, x, rhs)
        rhs *= -1

    def grad(self, i, x, out):
        out.zero()
        if i == soupy.STATE:
            self.K.mult(x[soupy.STATE], self.Ku)
            out.axpy(2.0, self.Ku)


    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER,CONTROL) of the q.o.i. in direction dir.
        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1, CONTROL=3) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.
        NOTE: setLinearizationPoint must be called before calling this method.
        """
        out.zero()
        if i == soupy.STATE and j == soupy.STATE:
            self.K.mult(dir, self.Ku)
            out.axpy(2.0, self.Ku)


    def apply_ijk(self,i,j,k,dir1,dir2, out):
        """
        Apply the third order variation of the q.o.i. w.r.t. ijk in direction dir1, dir2 for j and k
        :param i: STATE or PARAMETER or CONTROL
        :param j:
        :param k:
        :param dir1:
        :param dir2:
        :param out:
        :return: out
        """
        out.zero()


    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.
        INPUTS:
        - x = [u,m,p,z] is a list of the state u, parameter m, and adjoint variable p
        """
        pass 


class VelocityTrackingQoI(soupy.ControlQoI):
    """
    Defines the strain energy qoi integrated over the domain 
    """
    def __init__(self, Vh, v_target=np.array([1.0, 0.0]), chi=None):
        """
        Constructor.
        INPUTS:
        - Vh: the finite element space for [state, parameter, adjoint, optimization] variable
            Note that the state space is the joint space for velocity and pressure, (v, p)
        """
        self.Vh = Vh
        self.v_target = v_target


        self.chi = chi 
        self.x = [dl.Function(Vh[soupy.STATE]).vector(), dl.Function(Vh[soupy.PARAMETER]).vector(),
                  dl.Function(Vh[soupy.ADJOINT]).vector(), dl.Function(Vh[soupy.CONTROL]).vector()]
        self.x_test = [dl.TestFunction(Vh[soupy.STATE]), dl.TestFunction(Vh[soupy.PARAMETER]),
                       dl.TestFunction(Vh[soupy.ADJOINT]), dl.TestFunction(Vh[soupy.CONTROL])]

        self.u = dl.Function(Vh[soupy.STATE]).vector()
        self.Ku = dl.Function(Vh[soupy.STATE]).vector()

        u_trial = dl.TrialFunction(Vh[soupy.STATE])
        u_test = dl.TestFunction(Vh[soupy.STATE])
        v_trial, p_trial = dl.split(u_trial)
        v_test, p_test = dl.split(u_test)

        if self.chi is None:
            self.K = dl.assemble(dl.inner(v_trial, v_test) * dl.dx)
        else:
            self.K = dl.assemble(self.chi * dl.inner(v_trial, v_test) * dl.dx)

        self.d = dl.interpolate(dl.Constant(np.append(v_target, 0)), self.Vh[soupy.STATE]).vector()


    def cost(self, x):
        """
        evaluate the qoi at given x
        :param x: [state, parameter, adjoint, optimization] variable
        :return: qoi(x)
        """
        self.u.zero()
        self.u.axpy(1.0, x[soupy.STATE])
        self.u.axpy(-1.0, self.d)
        self.K.mult(self.u, self.Ku)
        return self.Ku.inner(self.u)

    def adj_rhs(self, x, rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
        with respect to the state u).
        INPUTS:
        - x coefficient vector of all variables
        - rhs: FEniCS vector to store the rhs for the adjoint problem.
        """
        self.grad(soupy.STATE, x, rhs)
        rhs *= -1

    def grad(self, i, x, out):
        out.zero()
        if i == soupy.STATE:
            self.u.zero()
            self.u.axpy(1.0, x[soupy.STATE])
            self.u.axpy(-1.0, self.d)
            self.K.mult(self.u, self.Ku)
            out.axpy(2.0, self.Ku)


    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER,CONTROL) of the q.o.i. in direction dir.
        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1, CONTROL=3) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.
        NOTE: setLinearizationPoint must be called before calling this method.
        """
        out.zero()
        if i == soupy.STATE and j == soupy.STATE:
            self.K.mult(dir, self.Ku)
            out.axpy(2.0, self.Ku)


    def apply_ijk(self,i,j,k,dir1,dir2, out):
        """
        Apply the third order variation of the q.o.i. w.r.t. ijk in direction dir1, dir2 for j and k
        :param i: STATE or PARAMETER or CONTROL
        :param j:
        :param k:
        :param dir1:
        :param dir2:
        :param out:
        :return: out
        """
        out.zero()


    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.
        INPUTS:
        - x = [u,m,p,z] is a list of the state u, parameter m, and adjoint variable p
        """
        pass 
