import pickle 
import numpy as np 
import matplotlib.pyplot as plt 

def evaluate_prediction_error(model, test_input, test_output, weight_matrix=None):
    pred = model.predict(test_input)
    diff = pred - test_output

    if weight_matrix is None:
        errors = np.linalg.norm(diff, axis=1)
        norms = np.linalg.norm(test_output, axis=1)
        # same 
        rel_errors = errors/norms
        mean_rel_errors = np.mean(rel_errors)
        print("Mean relative l2 error: %.4f" %(mean_rel_errors))
    else:
        errors = weighted_l2_norm(diff, weight_matrix)
        norms = weighted_l2_norm(pred, weight_matrix)
        # same 
        rel_errors = errors/norms
        mean_rel_errors = np.mean(rel_errors)
        print("Mean relative weighted l2 error: %.4f" %(mean_rel_errors))

    return mean_rel_errors


def weighted_l2_norm(x, W):
    Wx = x @ W 
    norm2 = np.einsum('ij,ij->i', Wx, x)
    return np.sqrt(norm2)


def split_train_test_arrays(data_arrays, n_train, n_test, shuffle=False, seed=1, n_max=10000):
    n_total = data_arrays[0].shape[0]
    assert n_total - n_train > n_test, "Not enough data for testing"
    train_arrays = []
    test_arrays = []

    if shuffle:
        # Choose a random shuffling of indices for the first `n_max` elements
        # This allows the remaining `n_total`- `n_max` samples to be safely
        # used as test samples
        print("Random shuffling of training data")
        n_max = min(n_max, n_total)
        indices_all = np.arange(n_max)
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(indices_all)
        train_indices = indices_all[:n_train]
        # test_indices = indices_all[-n_test:]
        for data_array in data_arrays:
            train_arrays.append(data_array[train_indices])
            test_arrays.append(data_array[-n_test:])
    else:
        # Just use the first/last continguous chunks as train/test samples
        print("Deterministic selection of training data")
        for data_array in data_arrays:
            train_arrays.append(data_array[:n_train])
            test_arrays.append(data_array[-n_test:])

    return train_arrays, test_arrays


def plot_and_save_history_all(history, save_dir):
    """
    Plot the history for all losses/metrics and save plots and history data
    - :code: `history` a tf History object
    - :code: `save_dir` directory for saving plots and history 
    """
    history_data = history.history
    keys = history_data.keys()
    for key in history_data.keys():
        plt.figure()
        if 'loss' in key:
            plt.semilogy(history_data[key])
        else:
            plt.plot(history_data[key])
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(key)
        plt.savefig(save_dir + '/%s.png' %(key))
        plt.close()

    with open(save_dir + '/history.p', 'wb') as outfile:
        pickle.dump(history_data, outfile)
