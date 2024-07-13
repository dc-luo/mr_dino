import os 
import pickle 
import soupy 
import hippyflow as hf 
import numpy as np 
from .settings_handler import SettingsHandler


def standard_data_generation_settings():
    settings = dict()
    settings['n_data'] =16384
    settings['n_pod'] = 512
    settings['pod_rank'] = 512
    settings['pod_orth'] = 'mass'
    settings['pod_shift'] = True
    settings['kle_rank'] = 512
    settings['kle_orth'] = 'mass'
    return SettingsHandler(settings)


class StandardDataLoader:
    def __init__(self, dir_handler, settings):
        self.dir_handler = dir_handler 
        self.settings = settings 
    
    def load_state_data(self, return_sizes=False):
        m_data = np.load(self.dir_handler.data_dir + '/m_data.npy')
        z_data = np.load(self.dir_handler.data_dir + '/z_data.npy')
        q_data = np.load(self.dir_handler.data_dir + '/q_data.npy')
        if return_sizes:
            return m_data, z_data, q_data, m_data.shape[1], z_data.shape[1], q_data.shape[1]
        else:
            return m_data, z_data, q_data 

    def load_kle(self, rank=None):
        kle_basis = np.load(self.dir_handler.basis_dir + f'/KLE_basis.npy')
        kle_projector = np.load(self.dir_handler.basis_dir + f'/KLE_projector.npy')
        kle_eig = np.load(self.dir_handler.basis_dir + f'/KLE_eigenvalues.npy')
        kle_shift = np.load(self.dir_handler.basis_dir + f'/KLE_shift.npy')
        if rank is not None:
            kle_basis = kle_basis[:, :rank]
            kle_projector = kle_projector[:, :rank]

        return kle_eig, kle_basis, kle_projector, kle_shift

    def load_full_jacobian_data(self, i):
        Jz_data = np.load(self.dir_handler.jacobian_dir + f'/Jz_data{i}.npy')
        return Jz_data


    def load_pod(self, rank=None):
        pod_name = self._make_pod_name()
        pod_eig = np.load(self.dir_handler.basis_dir + f'/{pod_name}_eigenvalues.npy')
        pod_basis = np.load(self.dir_handler.basis_dir + f'/{pod_name}_basis.npy')
        pod_projector = np.load(self.dir_handler.basis_dir + f'/{pod_name}_projector.npy')
        if self.settings['pod_shift']: 
            pod_shift = np.load(self.dir_handler.basis_dir + f'/{pod_name}_shift.npy')
        else:
            pod_shift = pod_basis[:,0] * 0 

        if rank is not None:
            pod_basis = pod_basis[:, :rank]
            pod_projector = pod_projector[:, :rank]

        return pod_eig, pod_basis, pod_projector, pod_shift

    def load_projected_jacobian_data(self, rank=None):
        pod_name = self._make_pod_name()
        projected_Jz = np.load(self.dir_handler.data_dir + f'/{pod_name}_projected_Jz_data.npy')
        if rank is not None:
            projected_Jz = projected_Jz[:, :rank, :]
        return projected_Jz

    def _make_pod_name(self):
        if self.settings['pod_shift']:
            pod_name = f"POD_shifted_n{self.settings['n_pod']}_r{self.settings['pod_rank']}"
        else:
            pod_name = f"POD_n{self.settings['n_pod']}_r{self.settings['pod_rank']}"
        return pod_name

    def load_mass_csr(self):
        pass


class StandardDataGenerationDriver:
    def __init__(self, observable_sampler, dir_handler, settings):
        self.observable_sampler = observable_sampler
        self.dir_handler = dir_handler
        self.settings = settings



    def sample_state_data(self):
        """
        Sample the input/output data tuples
        """
        self.observable_sampler.sample_data(self.settings['n_data'], self.dir_handler.data_dir)

    def load_state_data(self):
        """
        Load the sampled data for this process
        """
        m_data, z_data, q_data = self.observable_sampler.load_state_data(self.dir_handler.data_dir)
        return m_data, z_data, q_data 


    def sample_control_jacobian_at_data(self):
        """
        Sample control jacobian data
        """
        m_data, z_data, q_data = self.load_state_data()
        self.observable_sampler.sample_control_jacobian_at_data(self.dir_handler.jacobian_dir, m_data, z_data, q_data)


    def compute_kle_basis(self):
        """
        Compute the KLE basis 
        """
        self.observable_sampler.compute_kle_basis(self.settings['kle_rank'], 
                                                  self.dir_handler.basis_dir, 
                                                  orthogonality=self.settings['kle_orth'])

    def compute_pod_basis(self):
        """
        Compute POD basis from saved output data 
        """
        m_data, z_data, q_data = self.observable_sampler.load_state_data(self.dir_handler.data_dir)
        n_data = self.settings['n_pod'] 
        q_pod = q_data[:n_data]
        self.observable_sampler.compute_pod_basis(q_pod, self.settings['pod_rank'], self.settings['pod_shift'], self.dir_handler.basis_dir, self.settings['pod_orth'])


    def collect_and_project_jacobian(self):
        """
        Load the sampled Jacobians, project into POD basis, 
        and save as a collected array to data directory
        """
        self.observable_sampler.collect_and_project_jacobian(self.settings['n_pod'],
                                                             self.settings['pod_rank'],
                                                             self.settings['pod_shift'], 
                                                             self.dir_handler.data_dir, 
                                                             self.dir_handler.jacobian_dir, 
                                                             self.dir_handler.basis_dir)


    def run_all(self):
        print("Sampling data")
        self.sample_state_data()
        print("Sampling jacobian data")
        self.sample_control_jacobian_at_data()
        print("Computing bases")
        self.compute_pod_basis()
        self.compute_kle_basis()
        print("Projecting jacobian data and saving to data directory")
        self.collect_and_project_jacobian()




