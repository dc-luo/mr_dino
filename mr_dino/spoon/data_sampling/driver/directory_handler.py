import os 

class DirectoryHandler:
    def __init__(self, save_dir):
        self.save_dir = save_dir 
        self.data_dir = save_dir + '/data'
        self.jacobian_dir = save_dir + '/jacobian'
        self.basis_dir = save_dir + '/basis'
        self.figures_dir = save_dir + '/figures'
        self.networks_dir = save_dir + '/networks'

    def make_all_directories(self, networks=False):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.jacobian_dir, exist_ok=True)
        os.makedirs(self.basis_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.networks_dir, exist_ok=True)