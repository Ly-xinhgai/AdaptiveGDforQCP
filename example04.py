
import numpy as np
from autograd import grad
import autograd.numpy as np1
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import time
from scipy.optimize import BFGS,SR1
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

# Global variable for current n
current_n = 10

# RK4 method
def ode_solve_G(z0, G):
    n_steps = 500
    z = z0
    # print(z)
    h = np.array([0.05])
    for i_step in range(n_steps):
        k1 = h*G(z)

        k2 = h * (G((z+h/2)))
        k3 = h * (G((z+h/2)))
        k4 = h * (G((z+h)))
        k = (1/6)*(k1+2*k2+2*k3+k4)
        #k = k.reshape(1,current_n)
        z = np.array([z0]).reshape(current_n,1)
        z = z + k
        #print("z;",z.shape)
    return z
def f(x):
    n = len(x)  # Tính từ kích thước x
    tmp = n*[1]
    q = np1.array(tmp)
    return -np1.exp(-np1.sum((x**2) / (q**2)))

def g_i(x, i):
    start = 10*(i-1)
    end = start + 10
    squared_terms = x[start:end]**2
    return np1.sum(squared_terms) - 20
def derivative_g_i(x, i):
    grad = np.zeros_like(x)
    start = 10*(i-1)
    end = start + 10
    grad[start:end] = 2 * x[start:end]
    return grad
def create_constraints(n):
    num_ineq = n // 10
    constraints = []
    
    # Equality constraint
    constraints.append({
        'type': 'eq',
        'fun': lambda x: np.array([g3(x)]),
        'jac': lambda x: np.array([1]*(n//2) + [3]*(n//2))
    })
    
    # Multiple inequality constraints
    for i in range(1, num_ineq + 1):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=i: np.array([-g_i(x, idx)]),
            'jac': lambda x, idx=i: np.array([-derivative_g_i(x, idx)])
        })
    
    return constraints
def g3(x):
    x = np.array(x)
    A = np.array([[1]*(current_n//2) + [3]*(current_n//2)])
    b = np.array([[16]])
    return (A@(x.T) - b.T).tolist()[0][0] # 
# g1_dx = grad(g1)
# g2_dx = grad(g2)
# g3_dx = grad(g3)
# g_dx = [g1_dx,g2_dx]
f_dx = grad(f)
# bounds = Bounds([0,0],[np.inf,np.inf])
# cons = ({'type': 'eq',
#           'fun' : lambda x: np.array([g3(x)]),
#           'jac' : lambda x: np.array([1]*(current_n//2) + [3]*(current_n//2))},
#         {'type': 'ineq',
#           'fun' : lambda x: np.array([-g_i(x)]),
#           'jac' : lambda x: np.array([-grad(g_i)(x)])})
def rosen(x,y):
    return np.sqrt(np.sum((x-y)**2))
def find_min(y,n):
    x = np.random.rand(1,n).tolist()[0]
    constraints = create_constraints(n)  # Tạo constraints cho n
    res = minimize(rosen, x, args=(y), jac="2-point",hess=BFGS(),
                constraints=constraints,method='trust-constr', options={'disp': False})
    return res.x
def run_gda(x, max_iters, f, f_dx,n,alpha,mu0):
    res = []
    val = []
    lda = 1 #1e9
    sigma = 0.1 #100
    mut = mu0
    K = np.random.rand(1,1)
    res.append(x)
    val.append(f(x))
    x_pre = x
    for t in range(max_iters):
        if t % 5 == 0:  # Log mỗi 5 iterations
            print(f"    GDA iter {t}/{max_iters}", end="\r", flush=True)
        y = x - lda*f_dx(x)
        x_pre = x.copy()
        x = find_min(y,n)
        if f(x) - f(x_pre) + sigma*(np.dot(f_dx(x_pre).T,x_pre - x)) <= 0:
            lda = lda
        else:
            lda = K*lda
        res.append(x)
        val.append(f(x))
        if np.allclose(x, x_pre, rtol=1e-5, atol=1e-8):
            break
    print(f"    GDA completed {max_iters} iterations")
    return res,x
def Phi(s):
    if s > 0:
        return 1
    elif s == 0:
        return np.random.rand(1)
    return 0
# Neural network
A = np.array([[1]*(current_n//2) + [3]*(current_n//2)])
b = np.array([[16]])
def G(x):
    n = len(x)
    num_ineq = n // 10
    
    # Tính tất cả constraints
    gx = [g_i(x, i) for i in range(1, num_ineq + 1)]
    g_dx = [lambda x, idx=i: derivative_g_i(x, idx) for i in range(1, num_ineq + 1)]
    
    c_xt = 1.
    Px = np.zeros((n, 1))

    for (gi_val, gi_grad) in zip(gx, g_dx):
        c_xt *= (1-Phi(gi_val))
        Px += np.array([Phi(gi_val)*gi_grad(x)]).reshape(n,1)
    
    # Phần equality constraint
    A = np.array([[1]*(n//2) + [3]*(n//2)])
    b = np.array([[16]])
    c_xt *= (1-Phi(np.abs(A@(x) - b)))
    
    eq_constr_dx = ((2*Phi(A@(x)-b)-1)*A.T)
    
    return np.array([-c_xt*f_dx(x)]).reshape(n,1) - Px - eq_constr_dx
def run_rnn(x0, max_iters):
    xt = x0
    res = []
    res.append(xt.tolist())
    for t in range(max_iters):
        if t % 100 == 0:  # Log mỗi 100 iterations
            print(f"    RNN iter {t}/{max_iters}", end="\r", flush=True)
        xt = ode_solve_G(xt,G)
        res.append(xt.reshape(1,current_n).tolist()[0])
    print(f"    RNN completed {max_iters} iterations")
    # print(xt)
    # print(f(xt))
    return res,xt.reshape(1,current_n)

def plot_x(sol_all,count,max_iters):
    t = [i for i in range(max_iters+1)]
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    for i in range(count):
        if i ==0:
            text_color = 'red'
            text_label = r'$x_{1}(t)$'
        else:
            text_color = 'green'
            text_label = r'$x_{2}(t)$'
        for j in range(current_n):
            plt.plot(t, sol_all[i][:,j],color=text_color,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('x(t)')
    plt.legend([r'$x_{1}(t)$',r'$x_{2}(t)$'])
    plt.legend()
    plt.show()

def main_GDA(num, max_iters, n, alpha, mu0):
    start_time = time.time()
    sol_all1 = []
    for i in range(num):
        x0 = np.random.rand(1, n)
        x0 = find_min(x0, n)  # Start point
        _, xt = run_gda(x0, max_iters, f, f_dx, n, alpha, mu0)
        sol_all1.append(xt)
    end_time = time.time()
    execution_time = end_time - start_time
    value = -np.log(-f(xt))
    return sol_all1, execution_time, value

def main_RNN(num, max_iters, n):
    start_time = time.time()
    sol_all = []
    for i in range(num):
        x0 = np.random.rand(1, n)
        x0 = find_min(x0, n)  # Start point
        _, xt = run_rnn(x0, max_iters)
        sol_all.append(xt)
    end_time = time.time()
    execution_time = end_time - start_time
    value = -np.log(-f(xt))
    return sol_all, execution_time, value

if __name__ == '__main__':
    # Danh sách các giá trị n cần test
    n_values = [10, 20, 50, 100, 200, 400, 600]
    
    # Tạo danh sách lưu kết quả
    results = []
    
    for idx, n in enumerate(n_values):
        print(f"[{idx+1}/{len(n_values)}] Running n={n}...")
        
        # Cập nhật global variable
        current_n = n
        A = np.array([[1]*(current_n//2) + [3]*(current_n//2)])
        
        # Main parameters
        num = 1  # Number of starting points
        max_iters = 10  # Maximum number of iterations
        max_iters2 = 1000
        alpha = np.random.rand(1)  # Alpha parameter
        mu0 = np.random.rand(1)  # Mu0 parameter

        # Run the main function for GDA
        print(f"  Running GDA...")
        _, time_gda, value_gda = main_GDA(num, max_iters, n, alpha, mu0)
        print(f"    GDA done: {time_gda:.4f}s, value={value_gda:.6f}")

        # Run the main function for RNN
        print(f"  Running RNN...")
        _, time_rnn, value_rnn = main_RNN(num, max_iters2, n)
        print(f"    RNN done: {time_rnn:.4f}s, value={value_rnn:.6f}")
        
        # Lưu kết quả
        results.append({
            'n': n,
            'gda_iters': max_iters,
            'rnn_iters': max_iters2,
            'gda_time': time_gda,
            'rnn_time': time_rnn,
            'gda_value': value_gda,
            'rnn_value': value_rnn
        })
        
        # In kết quả ngay sau khi hoàn thành
        print(f"  Results for n={n}:")
        print(f"    GDA: {max_iters} iters, {time_gda:.4f}s, value={value_gda:.6f}")
        print(f"    RNN: {max_iters2} iters, {time_rnn:.4f}s, value={value_rnn:.6f}")
        print(f"  Completed n={n}\n")
    
    # In bảng kết quả
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'n':<6} {'GDA Iters':<10} {'RNN Iters':<10} {'GDA Time':<10} {'RNN Time':<10} {'GDA Value':<12} {'RNN Value':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['n']:<6} {result['gda_iters']:<10} {result['rnn_iters']:<10} "
              f"{result['gda_time']:<10.4f} {result['rnn_time']:<10.4f} "
              f"{result['gda_value']:<12.6f} {result['rnn_value']:<12.6f}")
    
    print("="*80)