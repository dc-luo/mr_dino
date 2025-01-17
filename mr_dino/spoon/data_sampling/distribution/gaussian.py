# Copyright (c) 2023, The University of Texas at Austin 
# & Georgia Institute of Technology
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the SOUPy package. For more information see
# https://github.com/hippylib/soupy/
#
# SOUPy is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 1991.

import math
import hippylib as hp 
import dolfin as dl 
import numpy as np


class FiniteIndependentGaussian:
    """
    Class for sampling from an Gaussian distribution with independent components to `dl.Vector`
    """
    def __init__(self, Vh, mu=0, sigma=1):
        """ 
        Constructor:
            :code: `Vh`: Function space for sample vectors
            :code: `mu`: Scalar or numpy vector for mean 
            :code: `sigma`: Scalar or numpy vector for standard deviation
            :code: `ndim`: Dimension of sample vectors
        """
        self.Vh = Vh
        self.mu = mu
        self.sigma = sigma
        self.dim = self.Vh.dim()
        self._dummy = dl.Function(Vh).vector()

    def init_vector(self, v):
        v.init( self._dummy.local_range() )

    def sample(self, out):
        assert out.mpi_comm().Get_size() == 1
        v = np.random.randn(self.dim) * self.sigma + self.mu
        out.set_local(v)

class ParFiniteIndependentGaussian:
    def __init__(self, Vh, mu=0.0, sigma=1.0,  rng=None):
        self.Vh = Vh 
        self.rng = rng 
        self.mu = mu 
        self.sigma = sigma 
        self.noise = dl.Function(Vh).vector()

        self.mean_vector = self.noise + mu 

        if self.rng is None:
            self.rng = hp.parRandom

    def init_vector(self, v):
        v.init( self.noise.local_range() )

    def sample(self, out):
        self.rng.normal(self.sigma, out)
        if math.isclose(self.mu, 0.0):
            out.axpy(1.0, self.mean_vector)
