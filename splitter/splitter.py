import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.model_selection import LeavePOut as SKLeavePOut

# ------------------------
# K-Fold
# ------------------------
def KFold(X, n_splits=5, max_splits=None):
    fold_size = len(X) // n_splits
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i != n_splits - 1 else len(X)
        test_indices = list(range(start, end))
        train_indices = list(range(0, start)) + list(range(end, len(X)))
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break

# ------------------------
# Stratified K-Fold
# ------------------------
def StratKFoldSplit(X, y, n_splits=5, max_splits=None):
    label_indices = defaultdict(list)
    for idx, label in enumerate(y):
        label_indices[label].append(idx)

    folds = [[] for _ in range(n_splits)]
    for label, indices in label_indices.items():
        np.random.shuffle(indices)
        fold_size = len(indices) // n_splits
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i != n_splits - 1 else len(indices)
            folds[i].extend(indices[start:end])

    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [idx for j in range(n_splits) if j != i for idx in folds[j]]
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break

# ------------------------
# Random Subsampling
# ------------------------
def RandomSubsampling(X, n_splits=5, test_size=0.2, max_splits=None):
    if n_splits is None:
        n_splits = 5
    n_samples = len(X)
    test_size = int(n_samples * test_size)
    for i in range(n_splits):
        indices = np.random.permutation(n_samples)
        test_indices = indices[:test_size].tolist()
        train_indices = indices[test_size:].tolist()
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break

# ------------------------
# Hold-Out
# ------------------------
def HoldOut(X, test_size=0.2):
    n_samples = len(X)
    test_size = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_size].tolist()
    train_indices = indices[test_size:].tolist()
    return train_indices, test_indices

# ------------------------
# Leave-One-Out
# ------------------------
def LeaveOneOut(X, *args, max_splits=None, **kwargs):
    n_samples = len(X)
    for i in range(n_samples):
        test_indices = [i]
        train_indices = list(range(0, i)) + list(range(i + 1, n_samples))
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break


# ------------------------
# Leave-P-Out
# ------------------------
def LeavePOut(X, p=2, max_splits=None):
    lpo = SKLeavePOut(p)
    for i, (train_idx, test_idx) in enumerate(lpo.split(X)):
        yield train_idx.tolist(), test_idx.tolist()
        if max_splits is not None and i + 1 >= max_splits:
            break

# ------------------------
# Bootstrap
# ------------------------
def Bootstrap(X, n_splits=5, max_splits=None):
    n_samples = len(X)
    for i in range(n_splits):
        train_indices = np.random.choice(n_samples, size=n_samples, replace=True).tolist()
        test_indices = list(set(range(n_samples)) - set(train_indices))
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break

# ------------------------
# Custom Split
# ------------------------
def CustomSplit(X, custom_indices):
    for train_indices, test_indices in custom_indices:
        yield train_indices, test_indices
