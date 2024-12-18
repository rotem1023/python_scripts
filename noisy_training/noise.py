import numpy as np
from numpy.testing import assert_array_almost_equal
from tqdm import tqdm 
 
# basic function
def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """ 

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y



# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print (P)

    return y_train, actual_noise,P

def noisify_multiclass_asymmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.eye(nb_classes)

    if noise > 0.0:
        P = np.random.uniform(low=0.1, high=1., size=(nb_classes, nb_classes))
        for i in range(nb_classes):
            P[i, i] = 1 - noise
            sum = P[i].sum() - P[i, i]
            for j in range(nb_classes):
                if i != j:
                    P[i, j] = P[i, j] / sum * noise


        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise,P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    return noisify_multiclass_symmetric_diag_and_uniform(y_train, noise, random_state=random_state, nb_classes=nb_classes) 


# def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
#     """mistakes:
#         flip in the symmetric way
#     """
#     P = np.ones((nb_classes, nb_classes))
#     n = noise
#     P = (n / (nb_classes - 1)) * P
    
#     if n > 0.0:
#         # 0 -> 1
#         P[0, 0] = 1. - n
#         for i in range(1, nb_classes-1):
#             P[i, i] = 1. - n
#         P[nb_classes-1, nb_classes-1] = 1. - n
#         y_train_noisy = multiclass_noisify(y_train, P=P,
#                                            random_state=random_state)
#         actual_noise = (y_train_noisy != y_train).mean()
#         assert actual_noise > 0.0
#         y_train = y_train_noisy
#     return y_train, actual_noise,P


"""
(1-epsilon) {i=j} + (epsilon / n_classes)
"""
def noisify_multiclass_symmetric_diag_and_uniform(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes)) * P
    
    if n > 0.0:
        # 0 -> 1
        P[0, 0] +=  1. - n
        for i in range(1, nb_classes-1):
            P[i, i] += 1. - n
        P[nb_classes-1, nb_classes-1] += 1. - n

        # validate:
        column_sums = np.sum(P, axis=0)
        row_sums = np.sum(P, axis=1)

        # Check if all sums are equal to 1
        columns_sum_to_one = np.allclose(column_sums, np.ones(P.shape[1]))
        rows_sum_to_one = np.allclose(row_sums, np.ones(P.shape[0]))
        # print (columns_sum_to_one, rows_sum_to_one)
        
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy 

    return y_train, actual_noise,P



"""
(1-epsilon) {i=j} + (epsilon / n_classes)
"""
def noisify_multiclass_custom_noise(y_train, noise_matrix, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = noise_matrix

    # validate:
    column_sums = np.sum(P, axis=0)
    row_sums = np.sum(P, axis=1)

    # Check if all sums are equal to 1
    columns_sum_to_one = np.allclose(column_sums, np.ones(P.shape[1]))
    rows_sum_to_one = np.allclose(row_sums, np.ones(P.shape[0]))
    # print (columns_sum_to_one, rows_sum_to_one)
    
    y_train_noisy = multiclass_noisify(y_train, P=P,
                                        random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    y_train = y_train_noisy 

    return y_train, actual_noise,P


def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=1):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate, t = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate, t = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)

    if noise_type == 'asymmetric':
        train_noisy_labels, actual_noise_rate, t = noisify_multiclass_asymmetric(train_labels, noise_rate,
                                                                             random_state=random_state, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate
