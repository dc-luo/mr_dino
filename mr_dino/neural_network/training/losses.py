import tensorflow as tf 


def dense_weighted_L2_error(weight):
    assert len(weight.shape.as_list()) == 2, 'Only worked out for rank 2 tensors'
    def dense_weighted_loss(y_true,y_pred):
        error = y_pred - y_true
        errorTWerror = tf.einsum('ij,ij->i',error,tf.einsum('ij,kj->ik',error,weight))
        return tf.reduce_mean(errorTWerror)
    return dense_weighted_loss

def dense_weighted_L2_accuracy(weight):
    assert len(weight.shape.as_list()) == 2, 'Only worked out for rank 2 tensors'
    def dense_weighted_acc(y_true,y_pred):
        error = y_pred - y_true
        errorTWerror = tf.einsum('ij,ij->i',error,tf.einsum('ij,kj->ik',error,weight))
        y_trueTWy_true = tf.einsum('ij,ij->i',y_true,tf.einsum('ij,kj->ik',y_true,weight))
        normalized_squared_difference = tf.reduce_mean(errorTWerror,axis=-1)\
                                        /tf.reduce_mean(y_trueTWy_true,axis =-1)
        return 1. - tf.sqrt(tf.reduce_mean(normalized_squared_difference))
    return dense_weighted_acc


def sparse_tensor_weighted_L2_error(sparse_tensor):
    assert len(sparse_tensor.shape.as_list()) == 2, 'Only worked out for rank 2 tensors'
    def sparse_tensor_weighted_loss(y_true,y_pred):
        error = y_pred - y_true
        errorTWerror = tf.einsum('ij,ij->i',error,tf.transpose(tf.sparse.sparse_dense_matmul(sparse_tensor, tf.transpose(error))))
        return tf.reduce_mean(errorTWerror)
    return sparse_tensor_weighted_loss

def sparse_tensor_weighted_L2_accuracy(sparse_tensor):
    assert len(sparse_tensor.shape.as_list()) == 2, 'Only worked out for rank 2 tensors'
    def sparse_tensor_weighted_acc(y_true,y_pred):
        error = y_pred - y_true
        errorTWerror = tf.einsum('ij,ij->i',error,tf.transpose(tf.sparse.sparse_dense_matmul(sparse_tensor, tf.transpose(error))))
        y_trueTWy_true = tf.einsum('ij,ij->i',y_true,tf.transpose(tf.sparse.sparse_dense_matmul(sparse_tensor, tf.transpose(y_true))))
        normalized_squared_difference = tf.reduce_mean(errorTWerror,axis=-1)\
                                        /tf.reduce_mean(y_trueTWy_true,axis =-1)
        return 1. - tf.sqrt(tf.reduce_mean(normalized_squared_difference))
    return sparse_tensor_weighted_acc

def normalized_mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    normalized_squared_difference = tf.reduce_mean(squared_difference,axis=-1)\
                                    /(tf.reduce_mean(tf.square(y_true),axis =-1))
    return tf.reduce_mean(normalized_squared_difference)

def normalized_mse_matrix(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    normalized_squared_difference = tf.reduce_sum(tf.reduce_sum(squared_difference,axis=-1),axis=-1)\
                            /tf.reduce_sum(tf.reduce_sum(tf.square(y_true),axis =-1),axis = -1)
    return tf.reduce_mean(normalized_squared_difference)

def l2_accuracy(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    normalized_squared_difference = tf.reduce_mean(squared_difference,axis=-1)\
                                    /tf.reduce_mean(tf.square(y_true),axis =-1)
    return 1. - tf.sqrt(tf.reduce_mean(normalized_squared_difference))

def f_accuracy_matrix(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    normalized_squared_difference = tf.reduce_sum(tf.reduce_sum(squared_difference,axis=-1),axis=-1)\
                                /tf.reduce_sum(tf.reduce_sum(tf.square(y_true),axis =-1),axis = -1)
    return 1. - tf.sqrt(tf.reduce_mean(normalized_squared_difference))



