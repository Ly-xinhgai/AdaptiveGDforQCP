import numpy as np
from autograd import grad
import autograd.numpy as np1
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from scipy.optimize import BFGS, SR1
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

# RK4 method
def ode_solve_G(z0, G):
    n_steps = 500
    z = np.array(z0).reshape(20, 1)
    h = np.array([0.05])
    for i_step in range(n_steps):
        k1 = h * G(z)
        k2 = h * (G((z + h/2)))
        k3 = h * (G((z + h/2)))
        k4 = h * (G((z + h)))
        k = (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        z = z + k
    return z

def f(x):
    tmp = 20 * [1]
    q = np1.array(tmp)
    return -np1.exp(-np1.sum((x**2) / (q**2)))

def g_i(x, i):
    start_idx = 10 * (i - 1)
    end_idx = start_idx + 10
    squared_terms = x[start_idx:end_idx]**2
    g_x_i = np1.sum(squared_terms) - 20
    return g_x_i

def derivative_g_i(x, i):
    grad = np.zeros_like(x)
    start_idx = 10 * (i - 1)
    end_idx = start_idx + 10
    grad[start_idx:end_idx] = 2 * x[start_idx:end_idx]
    return grad

def g3(x):
    x = np.array(x)
    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
    b = np.array([[16]])
    return (A @ (x.T) - b.T).tolist()[0][0]

f_dx = grad(f)
cons = ({'type': 'eq',
         'fun': lambda x: np.array([g3(x)]),
         'jac': lambda x: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])},
        {'type': 'ineq', 'fun': lambda x: np.array([-g_i(x, 1)]), 'jac': lambda x: np.array([-derivative_g_i(x, 1)])},
        {'type': 'ineq', 'fun': lambda x: np.array([-g_i(x, 2)]), 'jac': lambda x: np.array([-derivative_g_i(x, 2)])})

def rosen(x, y):
    return np.sqrt(np.sum((x - y)**2))

def find_min(y, n):
    x = np.random.rand(1, n).tolist()[0]
    res = minimize(rosen, x, args=(y), jac="2-point", hess=BFGS(),
                   constraints=cons, method='trust-constr', options={'disp': False})
    return res.x

def run_nonsmooth1(x, max_iters, f, f_dx, n, alpha, mu0):
    res = []
    val = []
    lda = 1
    sigma = 0.1
    mut = mu0
    K = np.random.rand(1, 1)
    res.append(x)
    val.append(f(x))
    x_pre = x
    for t in range(max_iters):
        y = x - lda * f_dx(x)
        x_pre = x.copy()
        x = find_min(y, n)
        if f(x) - f(x_pre) + sigma * (np.dot(f_dx(x_pre).T, x_pre - x)) <= 0:
            lda = lda
        else:
            lda = K * lda
        res.append(x)
        val.append(f(x))
    return res, val

def Phi(s):
    if s > 0:
        return 1
    elif s == 0:
        return np.random.rand(1)
    return 0

A = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
b = np.array([[16]])
def G(x):
    gx = [g_i(x, 1), g_i(x, 2)]
    g_dx = [grad(lambda x: g_i(x, 1)), grad(lambda x: g_i(x, 2))]
    c_xt = 1.
    Px = np.zeros((20, 1))
    for (i, j) in zip(gx, g_dx):
        c_xt *= (1 - Phi(i))
        Px += np.array([Phi(i) * j(x)]).reshape(20, 1)
    c_xt *= (1 - Phi(np.abs(A @ (x) - b)))
    eq_constr_dx = ((2 * Phi(A @ (x) - b) - 1) * A.T)
    return np.array([-c_xt * f_dx(x)]).reshape(20, 1) - Px - eq_constr_dx

def run_nonsmooth(x0, max_iters):
    xt = x0
    res = []
    res.append(xt)
    for t in range(max_iters):
        xt = ode_solve_G(xt, G)
        res.append(xt.reshape(20,))
    return res, np.array(res[-1]).reshape(1, 20)

def plot_x(sol_all, count, max_iters):
    t = np.arange(max_iters + 1)
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i in range(count):
        for j in range(20):
            label = f'$x_{{{j+1}}}(t)$' if i == 0 else None
            y = np.array(sol_all[i])[:max_iters + 1, j] if len(sol_all[i]) > max_iters + 1 else np.array(sol_all[i])[:, j]
            plt.plot(t, y, color=colors[j], label=label, linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('x(t)')
    plt.title('Trajectory of x(t) for GDA and RNN')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def main_GDA(num, max_iters, n, alpha, mu0):
    sol_all1 = []
    start_time = time.time()
    for i in range(num):
        x0 = np.random.rand(1, n)
        x0 = find_min(x0, n)
        res, _ = run_nonsmooth1(x0, max_iters, f, f_dx, n, alpha, mu0)
        xt = res[-1]
        print("Result GDA - x*:", xt)
        print("Value -ln(-f(x*)) of GDA:", -np.log(-f(xt)))
        sol_all1.append(np.array(res))
    end_time = time.time()
    print(f"Time for GDA: {end_time - start_time} seconds")
    return sol_all1

def main_RNN(num, max_iters, n):
    sol_all = []
    start_time = time.time()
    for i in range(num):
        x0 = np.random.rand(1, n)
        x0 = find_min(x0, n)
        res, _ = run_nonsmooth(x0, max_iters)
        xt = res[-1]
        print("Result RNN - x*:", xt)
        print("Value -ln(-f(x*)) of RNN:", -np.log(-f(xt)))
        sol_all.append(np.array(res))
    end_time = time.time()
    print(f"Time for RNN: {end_time - start_time} seconds")
    return sol_all

if __name__ == '__main__':
    num = 1
    max_iters_gda = 10  # 10 iterations for GDA
    max_iters_rnn = 1000  # 1000 iterations for RNN
    n = 20
    alpha = np.random.rand(1)
    mu0 = np.random.rand(1)

    result_GDA = main_GDA(num, max_iters_gda, n, alpha, mu0)
    result_RNN = main_RNN(num, max_iters_rnn, n)  # Fix: Remove alpha, mu0

    sol_all1 = result_GDA + result_RNN
    plot_x(sol_all1, 2, max(max_iters_gda, max_iters_rnn))