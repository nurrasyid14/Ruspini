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

    # NEW: distances and neighbor indices
    distances, indices = model.kneighbors(X_test)

    # convert neighbor indices to original global indices
    global_indices = [[train_idx[i] for i in row] for row in indices]

    acc = accuracy_score(y_test, preds)
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, preds, labels=labels)
    report = classification_report(y_test, preds, labels=labels, output_dict=True)

    return preds, acc, cm, report, distances, global_indices



def explore(X, y, splitter_fn, n_neighbors=3, **kwargs):
    results = []

    # Ensure splitter produces (train_idx, test_idx)
    try:
        splits = splitter_fn(X, y, **kwargs)
    except TypeError:
        splits = splitter_fn(X, **kwargs)

    for train_idx, test_idx in splits:

        preds, acc, cm, report, distances, neighbors = run_nn(
            X, y, train_idx, test_idx, n_neighbors
        )

        results.append({
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
            "report": report,
            "train_indices": list(map(int, train_idx)),
            "test_indices": list(map(int, test_idx)),
            "preds": preds.tolist(),
            "distances": distances.tolist(),
            "neighbors": neighbors
        })

    return results


