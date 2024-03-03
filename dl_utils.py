import numpy as np

"""_summary_
Train-Test splitting based on percentage or exact count of test data size.
"""

def train_test_split(X: np.array, y: np.array, test_size: int | float = 0.2):
    m = X.shape[0]
    if test_size < 1:
        train_size = int((1 - test_size)*m)
    else:
        assert (test_size < m), "Test data size should be smaller than total data size."
        train_size = int(m-test_size)

    train_indices = np.random.choice(m, size=train_size, replace=False)
    test_indices = np.setdiff1d(ar1=np.arange(0, m, 1), ar2=train_indices)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X[train_indices, :], X[test_indices, :], y[train_indices, :], y[test_indices, :]


