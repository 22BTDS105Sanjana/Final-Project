import numpy as np

def conformal_interval(y_true, y_pred, alpha=0.1):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    errors = np.abs(y_true - y_pred)

    q = np.quantile(errors, 1 - alpha)

    lower = y_pred - q
    upper = y_pred + q

    return lower, upper, q