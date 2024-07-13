import dolfin as dl 
import hippylib as hp 

class BiLaplacianSampler:
    def __init__(self, prior, rng=None):
        self.prior = prior 
        self.Vh = prior.Vh 
        self.mesh_comm = self.Vh.mesh().mpi_comm()
        self.noise = dl.Vector(self.mesh_comm)
        self.prior.init_vector(self.noise, "noise")
        self.rng = rng 
        if self.rng is None:
            self.rng = hp.parRandom
        
    def init_vector(self, v):
        self.prior.init_vector(v, 0)

    def sample(self, out):
        self.rng.normal(1.0, self.noise)
        self.prior.sample(self.noise, out)
