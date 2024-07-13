import pickle 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

def data_name_for_mesh_size(base_dir, mesh_size):
    return f"{base_dir}/data{mesh_size}"

def load_all_seeds(data_dir, base_name, seeds):
    acc_dir = data_dir + '/accuracy_on_fine'
    state_acc = np.zeros(len(seeds))
    jacobian_acc = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        nn_dir = acc_dir + f'/{base_name}_run{seed}.npy'
        acc = np.load(nn_dir)
        state_acc[i] = acc[0]
        jacobian_acc[i] = acc[1]
    return state_acc, jacobian_acc  



def make_error_dict(base_data_dir, base_network_name_l2, base_network_name_h1, mesh_sizes, n_seeds):

    seeds = range(1,n_seeds+1)

    accuracies = dict()
    accuracies["mrdino_state"] = np.zeros(len(mesh_sizes))
    accuracies["mrdino_jacobian"] = np.zeros(len(mesh_sizes))
    accuracies["mrno_state"] = np.zeros(len(mesh_sizes))
    accuracies["mrno_jacobian"] = np.zeros(len(mesh_sizes))
    accuracies["mesh_sizes"] = mesh_sizes
    accuracies["n_dofs"] = (mesh_sizes + 1)**2


    for i, mesh_size in enumerate(mesh_sizes):
        data_dir = data_name_for_mesh_size(base_data_dir, mesh_size)
        state_acc, jacobian_acc = load_all_seeds(data_dir, base_network_name_l2, seeds)
        accuracies["mrno_state"][i] = np.mean(state_acc)
        accuracies["mrno_jacobian"][i] = np.mean(jacobian_acc)
        print(state_acc, jacobian_acc)

    for i, mesh_size in enumerate(mesh_sizes):
        data_dir = data_name_for_mesh_size(base_data_dir, mesh_size)
        state_acc, jacobian_acc = load_all_seeds(data_dir, base_network_name_h1, seeds)
        accuracies["mrdino_state"][i] = np.mean(state_acc)
        accuracies["mrdino_jacobian"][i] = np.mean(jacobian_acc)


    return accuracies 

if __name__ == "__main__":

    base_data_dir = "/storage/dingcheng/repos/github/mr_dino/applications/poisson"
    mesh_sizes = np.array([24, 32, 48, 64])
    n_seeds = 8
    base_network_name_l2 = "Srbnet_data1024_l2_only_kle-pod_m100u300_400_400_scheduler"
    base_network_name_h1 = "Srbnet_data1024_kle-pod_m100u300_400_400_scheduler"

    base_network_name_l2_small = "Srbnet_data1024_l2_only_kle-pod_m50u100_200_200_scheduler"
    base_network_name_h1_small = "Srbnet_data1024_kle-pod_m50u100_200_200_scheduler"

    results_dir = base_data_dir + "/mesh_independence"


    acc_large = make_error_dict(base_data_dir, base_network_name_l2, base_network_name_h1, mesh_sizes, n_seeds)
    acc_small = make_error_dict(base_data_dir, base_network_name_l2_small, base_network_name_h1_small, mesh_sizes, n_seeds)
    accuracies = {'small' : acc_small, 'large' : acc_large}

    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/results_on_fine.p', 'wb') as rfile:
        pickle.dump(accuracies, rfile)

    plt.figure(figsize=(8,6))
    plt.loglog(acc_small["n_dofs"], acc_small["mrno_state"], '--ob', label="MR-NO")
    plt.loglog(acc_small["n_dofs"], acc_small["mrdino_state"], '--sr', label="MR-DINO")
    plt.loglog(acc_large["n_dofs"], acc_large["mrno_state"], '-ob', label="MR-NO")
    plt.loglog(acc_large["n_dofs"], acc_large["mrdino_state"], '-sr', label="MR-DINO")
    plt.legend()
    plt.title("State accuracy")
    plt.savefig(results_dir + '/state_acc.png')

    plt.figure(figsize=(8,6))
    plt.loglog(acc_small["n_dofs"], acc_small["mrno_jacobian"], '--ob', label="MR-NO")
    plt.loglog(acc_small["n_dofs"], acc_small["mrdino_jacobian"], '--sr', label="MR-DINO")
    plt.loglog(acc_large["n_dofs"], acc_large["mrno_jacobian"], '-ob', label="MR-NO")
    plt.loglog(acc_large["n_dofs"], acc_large["mrdino_jacobian"], '-sr', label="MR-DINO")
    plt.legend()
    plt.title("Jacobian accuracy")

    plt.savefig(results_dir + '/jacobian_acc.png')
