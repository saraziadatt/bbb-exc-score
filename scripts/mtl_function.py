# Light GBM MTL: 

import numpy as np


# Smooth sigmoid version of class prediction: sigmoid(-k*(y_pred + 1))
def sigmoid(x, k=10):
    return 1 / (1 + np.exp(-k * x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def custom_multi_loss(y_train_class, y_reg_w_nans,  alpha, beta, margin=-1, margin_buffer = 0.1):
    """
    alpha: weight for regression loss (MSE)
    beta: weight for classification loss (BCE)
    """

    def objective(preds, train_data):

        y_pred = preds
        y_true = train_data.get_label()
        y_class = y_train_class

        # Regression loss (MSE)
        is_reg_available = ~np.isnan(y_reg_w_nans)
        grad_reg = np.zeros_like(y_pred)
        hess_reg = np.zeros_like(y_pred)

        grad_reg[is_reg_available] = alpha * (y_pred[is_reg_available] - y_reg_w_nans[is_reg_available])
        hess_reg[is_reg_available] = alpha

        # Classification loss
        grad_class = np.zeros_like(y_pred)
        hess_class = np.zeros_like(y_pred)

        buffer = margin_buffer 
        lower_bound = margin + buffer
        upper_bound = margin - buffer

        n_0 = np.sum(y_class == 0)
        n_1 = np.sum(y_class == 1)
        w0 = 1.0 / n_0 if n_0 > 0 else 0.0
        w1 = 1.0 / n_1 if n_1 > 0 else 0.0

        # Penalize class 1 
        idx_wrong_1 = (y_class == 1) & (y_pred > upper_bound)
        diff_1 = y_pred[idx_wrong_1] - upper_bound
        grad_class[idx_wrong_1] = beta * w1 * 2 * diff_1
        hess_class[idx_wrong_1] = beta * w1 * 2

        # Penalize class 0
        idx_wrong_0 = (y_class == 0) & (y_pred < lower_bound)
        diff_0 = y_pred[idx_wrong_0] - lower_bound
        grad_class[idx_wrong_0] = beta * w0 * 2 * diff_0
        hess_class[idx_wrong_0] = beta * w0 * 2

        
        # Combine
        grad = grad_reg + grad_class
        hess = hess_reg + hess_class


        return grad, hess

    return objective
