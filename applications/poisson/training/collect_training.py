import pickle 
import os 
import numpy as np 
import matplotlib.pyplot as plt 


def load_all_seeds(data_dir, base_name, seeds):
    acc_dir = data_dir + '/accuracy'
    state_acc = np.zeros(len(seeds))
    jacobian_acc = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        nn_dir = acc_dir + f'/{base_name}_run{seed}.npy'
        acc = np.load(nn_dir)
        state_acc[i] = acc[0]
        jacobian_acc[i] = acc[1]
    return state_acc, jacobian_acc  



def make_error_dict(base_data_dir, n_samples, n_seeds, small=False):
    seeds = range(1,n_seeds+1)

    accuracies = dict()
    accuracies["mrdino_state"] = np.zeros(len(n_samples))
    accuracies["mrdino_jacobian"] = np.zeros(len(n_samples))
    accuracies["mrno_state"] = np.zeros(len(n_samples))
    accuracies["mrno_jacobian"] = np.zeros(len(n_samples))
    accuracies["n_samples"] = n_samples


    for i, n_sample in enumerate(n_samples):
        base_network_name_l2, base_network_name_h1 = base_network_name(n_sample, small)
        state_acc, jacobian_acc = load_all_seeds(data_dir, base_network_name_l2, seeds)
        accuracies["mrno_state"][i] = np.mean(state_acc)
        accuracies["mrno_jacobian"][i] = np.mean(jacobian_acc)
        print(state_acc, jacobian_acc)

        state_acc, jacobian_acc = load_all_seeds(data_dir, base_network_name_h1, seeds)
        accuracies["mrdino_state"][i] = np.mean(state_acc)
        accuracies["mrdino_jacobian"][i] = np.mean(jacobian_acc)
        print(state_acc, jacobian_acc)
    return accuracies 


def base_network_name(n_data, small=False):
    if small:
        network_name_l2 = f"Srbnet_data{n_data}_l2_only_kle-pod_m50u100_200_200_scheduler"
        network_name_h1 = f"Srbnet_data{n_data}_kle-pod_m50u100_200_200_scheduler"
    else:
        network_name_l2 = f"Srbnet_data{n_data}_l2_only_kle-pod_m100u300_400_400_scheduler"
        network_name_h1 = f"Srbnet_data{n_data}_kle-pod_m100u300_400_400_scheduler"
    return network_name_l2, network_name_h1


if __name__ == "__main__":

    data_dir = "/storage/dingcheng/repos/github/mr_dino/applications/poisson/data64"
    training_data_sizes = np.array([512, 1024, 2048, 4096])
    n_seeds = 9

    results_dir = data_dir + "/training_results"


    acc_large = make_error_dict(data_dir, training_data_sizes, n_seeds, small=False)
    acc_small = make_error_dict(data_dir, training_data_sizes, n_seeds, small=True)
    accuracies = {'small' : acc_small, 'large' : acc_large}

    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/results.p', 'wb') as rfile:
        pickle.dump(accuracies, rfile)

    plt.figure(figsize=(8,6))
    plt.loglog(acc_small["n_samples"], acc_small["mrno_state"], '--ob', label="MR-NO")
    plt.loglog(acc_small["n_samples"], acc_small["mrdino_state"], '--sr', label="MR-DINO")
    plt.loglog(acc_large["n_samples"], acc_large["mrno_state"], '-ob', label="MR-NO")
    plt.loglog(acc_large["n_samples"], acc_large["mrdino_state"], '-sr', label="MR-DINO")
    plt.legend()
    plt.title("State accuracy")
    plt.savefig(results_dir + '/state_acc.png')

    plt.figure(figsize=(8,6))
    plt.loglog(acc_small["n_samples"], acc_small["mrno_jacobian"], '--ob', label="MR-NO")
    plt.loglog(acc_small["n_samples"], acc_small["mrdino_jacobian"], '--sr', label="MR-DINO")
    plt.loglog(acc_large["n_samples"], acc_large["mrno_jacobian"], '-ob', label="MR-NO")
    plt.loglog(acc_large["n_samples"], acc_large["mrdino_jacobian"], '-sr', label="MR-DINO")
    plt.legend()
    plt.title("Jacobian accuracy")

    plt.savefig(results_dir + '/jacobian_acc.png')
