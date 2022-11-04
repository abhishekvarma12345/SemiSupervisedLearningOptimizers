import time
import numpy as np
from common import calc_gradient, obj_func

def cycl_BCGD(y_l, W_l, W_u, lr, num_blocks, max_iter, eps):
    loss_stat = [] # loss statistics
    time_stat = [] # time statistics
    y_pred = np.zeros(len(W_u))
    print(f"Initial loss: {obj_func(y_pred, y_l, W_l, W_u)}")

    # calculate the partition size for unlabeled data
    p_size_u = int(len(y_pred) / num_blocks)
    
    for iter in range(max_iter):
        y_old = y_pred.copy()
        grad_time = 0
        for i in range(num_blocks):
            # slice the blocks
            y_k = y_pred[(i*p_size_u):((i+1)*p_size_u)]
            W_l_k = W_l[:, (i*p_size_u):((i+1)*p_size_u)]
            W_u_k = W_u[(i*p_size_u):((i+1)*p_size_u), (i*p_size_u):((i+1)*p_size_u)]

            # calculate the gradient
            start = time.time()
            grad = calc_gradient(y_k, y_l, W_l_k, W_u_k)
            end = time.time()
            grad_time += end - start

            # update predictions
            y_k -= lr * grad
            y_pred[(i*p_size_u):((i+1)*p_size_u)] = y_k
        
        time_stat.append(grad_time)
        loss = obj_func(y_pred, y_l, W_l, W_u)
        loss_stat.append(loss)

        stop_cond = np.linalg.norm(y_old - y_pred)
        print(f"Interation {iter}, loss:  {obj_func(y_pred, y_l, W_l, W_u)}, norm: {stop_cond}")
        if stop_cond < eps:
          break

    return y_pred, loss_stat, time_stat 