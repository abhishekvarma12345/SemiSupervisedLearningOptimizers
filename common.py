import yaml
import numpy as np
import matplotlib.pyplot as plt

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content

def calc_gradient(y, y_l, W_l, W_u):
    nl = len(y_l) # number of labeled points
    nu = len(y) # number of unlabeled points
    part1 = np.outer(np.ones(nl), y) - np.outer(y_l, np.ones(nu))
    part1 = np.multiply(W_l, part1)
    part1 = np.matmul(part1.T, np.ones(nl))

    part2 = np.outer(np.ones(nu), y) - np.outer(y, np.ones(nu))
    part2 = np.multiply(W_u, part2)
    part2 = np.matmul(part2.T, np.ones(nu))
    return 2 * (part1 + part2)

def obj_func(y, y_l, W_l, W_u):
    nl = len(y_l) # number of labeled points
    nu = len(y) # number of unlabeled points
    part1 = np.outer(np.ones(nl), y) - np.outer(y_l, np.ones(nu))
    part1 = np.multiply(part1, part1)
    part1 = np.multiply(W_l, part1)

    part2 = np.outer(np.ones(nu), y) - np.outer(y, np.ones(nu))
    part2 = np.multiply(part2, part2)
    part2 = np.multiply(W_u, part2)
    return np.sum(part1) + (0.5 * np.sum(part2))

def calc_Lipshitz(W_l, W_u):
    hes = W_u * (-2)
    diag_values = (W_l.sum(axis=0) + W_u.sum(axis=0))
    np.fill_diagonal(hes, diag_values)
    eigenvalues = np.linalg.eigvalsh(hes)
    L = np.max(eigenvalues)
    return L

def plot_data(X,Y):
    plt.scatter(X.T[0], X.T[1], c=Y)
    plt.show()

def graph1(loss_stat_GM, loss_stat_RB,loss_stat_CB):
    # plotting loss and gradient computation time
    # for BCGD the time for each gradient calculation in the number of blocks was summed
    plt.figure(figsize=(16,7))
    plt.subplot(1, 2, 1)
    plt.plot(loss_stat_GM, label = "Gradient Method")
    plt.plot(loss_stat_RB, label = "Randomized BCGD")
    plt.plot(loss_stat_CB, label = "Cyclic BCGD")
    plt.title("Loss at each iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    # plot the loss after n iterations
    n = 75
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(loss_stat_GM)))[n:], loss_stat_GM[n:], label = "Gradient Method")
    plt.plot(list(range(len(loss_stat_RB)))[n:], loss_stat_RB[n:], label = "Randomized BCGD")
    plt.plot(list(range(len(loss_stat_CB)))[n:], loss_stat_CB[n:], label = "Cyclic BCGD")
    plt.title(f"Loss after {n} iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

def graph2(time_stat_GM, time_stat_RB, time_stat_CB):
    plt.figure(figsize=(8,7))
    plt.plot(time_stat_GM, label = "Gradient Method")
    plt.plot(time_stat_RB, label = "Randomized BCGD")
    plt.plot(time_stat_CB, label = "Cyclic BCGD")
    plt.title("Gradient computation time per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("CPU time")
    plt.legend()
    plt.show()

def graph3(time_stat_GM, time_stat_RB, time_stat_CB):
    # total and average gradient computation time
    total_GM = np.array(time_stat_GM).sum()
    total_RB = np.array(time_stat_RB).sum()
    total_CB = np.array(time_stat_CB).sum()

    avg_per_iter_GM = total_GM / len(time_stat_GM)
    avg_per_iter_RB = total_RB / len(time_stat_RB)
    avg_per_iter_CB = total_CB / len(time_stat_CB)

    x_ticks = np.array(['GM', 'Rand BCGD', 'Cycl BCGD'])
    total_time = np.array([total_GM, total_RB, total_CB])
    avg_time = np.array([avg_per_iter_GM, avg_per_iter_RB, avg_per_iter_CB])

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.bar(x_ticks,total_time)
    plt.title("Total time for gradient computation")

    plt.subplot(1, 2, 2)
    plt.bar(x_ticks,avg_time)
    plt.title("Average time per iteration")
    plt.show()


