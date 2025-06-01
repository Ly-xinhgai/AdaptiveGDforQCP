import numpy as np
from autograd import grad
import autograd.numpy as anp
from scipy.optimize import minimize, BFGS
import time

# Objective function
def f(x, q=None):
    if q is None:
        q = anp.ones_like(x) * 10
    x_clipped = anp.clip(x, -1e6, 1e6)
    q_clipped = anp.clip(q, 1e-6, 1e6)
    return -anp.exp(-anp.sum((x_clipped**2) / (q_clipped**2)))

f_dx = grad(f)

# Inequality constraint g_i(x)
def g_i(x, i=1):
    indices = slice(10 * (i - 1), min(10 * i, len(x)))
    return np.sum(x[indices]**2) - 20

def derivative_g_i(x, i=1):
    grad_vec = np.zeros_like(x)
    indices = slice(10 * (i - 1), min(10 * i, len(x)))
    grad_vec[indices] = 2 * x[indices]
    return grad_vec

# Equality constraint g3(x)
def g3(x):
    A = np.ones(len(x))
    if len(x) >= 10:
        A[5:] *= 3
    b = 16
    return np.dot(A, x) - b

def derivative_g3(x):
    A = np.ones(len(x))
    if len(x) >= 10:
        A[5:] *= 3
    return A

# Constraints for scipy minimize
cons = [
    {'type': 'eq', 'fun': g3, 'jac': derivative_g3},
    {'type': 'ineq', 'fun': lambda x: -g_i(x, 1), 'jac': lambda x: -derivative_g_i(x, 1)}
]

def ode_solve_G(z0, G, h=0.01, n_steps=2000):
    z = z0.copy()
    for _ in range(n_steps):
        k1 = h * G(z).flatten()
        k2 = h * G(z + 0.5 * k1).flatten()
        k3 = h * G(z + 0.5 * k2).flatten()
        k4 = h * G(z + k3).flatten()
        z = z + (k1 + 2*k2 + 2*k3 + k4) / 6
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            raise ValueError("ODE solver encountered NaN or Inf")
    return z.flatten()

def find_min(y, n):
    x0 = np.clip(y, 1e-6, 1e6)
    res = minimize(
        lambda x: np.sum((x - y)**2),
        x0,
        method='trust-constr',
        jac=lambda x: 2*(x - y),
        hess=BFGS(),
        constraints=cons,
        options={'disp': False, 'maxiter': 2000}
    )
    if not res.success:
        print(f"Optimization failed: {res.message}")
        return x0
    return res.x

def run_nonsmooth1(x0, max_iters, f, f_dx, n, alpha, mu0):
    res, val = [x0], [f(x0)]
    lda, sigma = 1.0, 0.1
    x_pre = x0.copy()

    for _ in range(max_iters):
        grad_val = f_dx(x_pre)
        grad_norm = np.linalg.norm(grad_val)
        if grad_norm < 1e-3:
            print(f"Stopped due to small gradient norm: {grad_norm:.6f}")
            break

        y = x_pre - lda * grad_val
        x = find_min(y, n)

        # Armijo condition
        if f(x) - f(x_pre) + sigma * np.dot(grad_val, x_pre - x) <= 0:
            pass
        else:
            lda *= 0.9  # Decrease step size

        res.append(x)
        val.append(f(x))
        if np.linalg.norm(x - x_pre) < 1e-8 or lda < 1e-10:
            break
        x_pre = x.copy()

    return res, x

def Phi(s):
    # Smooth approx of Heaviside using tanh and clip for stability
    s_scaled = np.clip(s * 10, -50, 50)
    return 0.5 * (1 + np.tanh(s_scaled))

def G(x):
    x = np.array(x).flatten()
    gx = [g_i(x, 1)]
    g_dx = [derivative_g_i(x, 1)]
    c_xt = 1.0

    Px = np.zeros((len(x), 1))
    for g, g_d in zip(gx, g_dx):
        p = Phi(g)
        c_xt *= (1 - p)
        Px += (p * g_d.reshape(-1,1))

    A = np.ones(len(x))
    if len(x) >= 10:
        A[5:] *= 3
    eq_constr = abs(np.dot(A, x) - 16)
    p_eq = Phi(eq_constr)
    c_xt *= (1 - p_eq)
    eq_constr_dx = (2 * p_eq - 1) * A.reshape(-1, 1)

    val = -c_xt * f_dx(x).reshape(-1, 1) - Px - eq_constr_dx
    if np.any(np.isnan(val)) or np.any(np.isinf(val)):
        raise ValueError("G(x) contains NaN or Inf")
    return val

def run_nonsmooth(x0, max_iters):
    res = [x0.copy()]
    xt = x0.copy()
    for _ in range(max_iters):
        xt = ode_solve_G(xt, G)
        res.append(xt.copy())
        if np.linalg.norm(xt - res[-2]) < 1e-8:
            break
    return res, xt

def main_GDA(num, max_iters, n, alpha, mu0):
    sol_all = []
    for _ in range(num):
        x0 = np.random.rand(n)
        x0 = find_min(x0, n)
        _, xt = run_nonsmooth1(x0, max_iters, f, f_dx, n, alpha, mu0)
        print("Result GDA - x*:", xt)
        print("Value -ln(-f(x*)) of GDA:", -np.log(-f(xt)))
        print(f"Equality constraint g3(x*): {g3(xt):.6f}")
        print(f"Inequality constraint g_i(x*): {g_i(xt):.6f}")
        sol_all.append(xt)
    return sol_all

def main_RNN(num, max_iters, n):
    sol_all = []
    for _ in range(num):
        x0 = np.random.rand(n)
        x0 = find_min(x0, n)
        _, xt = run_nonsmooth(x0, max_iters)
        print("Result RNN - x*:", xt)
        print("Value -ln(-f(x*)) of RNN:", -np.log(-f(xt)))
        print(f"Equality constraint g3(x*): {g3(xt):.6f}")
        print(f"Inequality constraint g_i(x*): {g_i(xt):.6f}")
        sol_all.append(xt)
    return sol_all

if __name__ == '__main__':
    num = 1
    max_iters = 100
    n = 20
    alpha = 0.1
    mu0 = 0.1

    start_time = time.time()
    result_GDA = main_GDA(num, max_iters, n, alpha, mu0)
    print(f"GDA Time: {time.time() - start_time:.4f}s")

    start_time = time.time()
    result_RNN = main_RNN(num, max_iters, n)
    print(f"RNN Time: {time.time() - start_time:.4f}s")
