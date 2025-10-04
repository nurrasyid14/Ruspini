import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_nn(X, y, train_idx, test_idx, n_neighbors=3):
    from sklearn.neighbors import KNeighborsClassifier
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    labels = sorted(set(y))   # FIX: always include all classes
    cm = confusion_matrix(y_test, preds, labels=labels)
    report = classification_report(y_test, preds, labels=labels, output_dict=True)

    return preds, acc, cm, report



def explore(X, y, splitter_fn, n_neighbors=3, **kwargs):
    """
    Run kNN exploration with a given splitter function.
    splitter_fn: one of the functions from splitter (KFold, StratKFoldSplit, etc.)
    kwargs: parameters for the splitter function (e.g., n_splits, test_size, p, etc.)
    """
    results = []

    # Some splitters require y (like StratKFoldSplit), others don't
    try:
        splits = splitter_fn(X, y, **kwargs)  # try with y
    except TypeError:
        splits = splitter_fn(X, **kwargs)     # fallback without y

    for i, (train_idx, test_idx) in enumerate(splits):
        preds, acc, cm, report = run_nn(X, y, train_idx, test_idx, n_neighbors)
        results.append({
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "train_indices": list(map(int, train_idx)),
        "test_indices": list(map(int, test_idx))
        })


    return results

