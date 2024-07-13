import numpy as np 

def relative_l2_prediction_error(pred_output, test_output, weight_matrix=None):
    """
    Computes the mean relative l2 prediction errors, with optional weight matrix 
        rel error = ||xi - xi_pred||_W/||xi||_W

    - :code: `pred_output` numpy array of predict output with shape (n_data, dim)
    - :code: `test_output` numpy array of test data output with shape (n_data, dim)
    - :code: `weight_matrix` Optional weighting matrix defining the l2 norm. Default is
        simply identity 

    returns the mean of the relative errors 
    - :code: `mean_rel_errors`
    """

    diff = pred_output - test_output

    if weight_matrix is None:
        errors = np.linalg.norm(diff, axis=1)
        norms = np.linalg.norm(test_output, axis=1)
        # same 
        rel_errors = errors/norms
        mean_rel_errors = np.mean(rel_errors)
        print("Mean relative l2 error: %.4f" %(mean_rel_errors))
    else:
        errors = weighted_l2_norm(diff, weight_matrix)
        norms = weighted_l2_norm(pred_output, weight_matrix)
        # same 
        rel_errors = errors/norms
        mean_rel_errors = np.mean(rel_errors)
        print("Mean relative weighted l2 error: %.4f" %(mean_rel_errors))

    return mean_rel_errors


def relative_fro_prediction_error(pred_output, test_output, weight_matrix=None):
    """
    Computes the mean relative Frobenius norm prediction errors, with optional weight matrix 
        rel error = ||Xi - Xi_pred||_W/||Xi||_W
    where The Frobenius norm can be weighted by the output inner product.
        ||X||_W = ||W^{1/2} X ||_F

    - :code: `pred_output` numpy array of predict output with shape (n_data, dim)
    - :code: `test_output` numpy array of test data output with shape (n_data, dim)
    - :code: `weight_matrix` Optional weighting matrix defining the l2 norm. Default is
        simply identity 

    returns the mean of the relative errors 
    - :code: `mean_rel_errors`
    """
    diff = pred_output - test_output

    n_data = test_output.shape[0] 
    errors = np.zeros(n_data)
    norms = np.zeros(n_data)
    rel_errors = np.zeros(n_data)

    if weight_matrix is None:
        for i_data in range(n_data):
            errors[i_data] = np.linalg.norm(diff[i_data], 'fro')
            norms[i_data] = np.linalg.norm(test_output[i_data], 'fro')

        rel_errors = errors/norms
        mean_rel_errors = np.mean(rel_errors)
        print("Mean relative Frobenius error: %.4f" %(mean_rel_errors))
    else:
        for i_data in range(n_data):
            errors[i_data] = weighted_fro_norm(diff[i_data], weight_matrix)
            norms[i_data] = weighted_fro_norm(test_output[i_data], weight_matrix)
        rel_errors = errors/norms
        mean_rel_errors = np.mean(rel_errors)
        print("Mean relative weighted Frobenius error: %.4f" %(mean_rel_errors))

    return mean_rel_errors

def weighted_fro_norm(X, W):
    input_dim = X.shape[1] 
    WX = W @ X 
    norm2 = np.einsum('ij,ij->i', WX, X)
    return np.sqrt(np.mean(norm2))


def weighted_l2_norm(x, W):
    Wx = x @ W 
    norm2 = np.einsum('ij,ij->i', Wx, x)
    return np.sqrt(norm2)
