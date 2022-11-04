import time
import numpy as np
from common import calc_gradient, obj_func


def GM_algorithm(y_l, W_l, W_u, lr, max_iter, eps):
    loss_stat = [] # loss statistics
    time_stat = [] # time statistics
    y_pred = np.zeros(len(W_u))
    print(f"Initial loss: {obj_func(y_pred, y_l, W_l, W_u)}")
    for i in range(max_iter):
        start_grad = time.time()
        gradient = calc_gradient(y_pred, y_l, W_l, W_u)
        end_grad = time.time()
        time_stat.append(end_grad - start_grad)

        y_pred += -lr * gradient

        stop_cond = np.linalg.norm(gradient) * abs(lr)
        loss = obj_func(y_pred, y_l, W_l, W_u)
        loss_stat.append(loss)
        print(f"Iteration {i+1}: loss {loss}, gradient norm {stop_cond}")
        if stop_cond < eps:
            break
    return y_pred, loss_stat, time_stat