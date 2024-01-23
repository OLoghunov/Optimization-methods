import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv
from numpy import identity as I


# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)

    return v


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def goldensectionsearch(f, interval, tol):
    a, b = interval
    Phi = (1 + np.sqrt(5)) / 2
    L = b - a
    x1 = b - L / Phi
    x2 = a + L / Phi
    xmin = x1

    # main loop
    while np.abs(L) > tol:
        if f(x1) > f(x2):
            a = x1
            xmin = x2
            x1 = x2
            L = b - a
            x2 = a + L / Phi
        else:
            b = x2
            xmin = x1
            x2 = x1
            L = b - a
            x1 = b - L / Phi

    return xmin


def pparam(pU, pB, tau):
    return tau * pU if tau <= 1 else pU + (tau - 1) * (pB - pU)


def doglegsearch(mod, g0, B0, Delta, tol):
    pU = sum(g0 * g0) / np.dot(np.dot(g0.transpose(), B0), g0) * g0
    pB = np.dot(inv(- B0), g0)
    pB *= goldensectionsearch(lambda alpha: mod(alpha * pB), [-Delta / norm(pB), Delta / norm(pB)], tol)
    
    pmin = pparam(pU, pB, goldensectionsearch(lambda tau: mod(pparam(pU, pB, tau)), [0, 2], tol))
    
    return pmin if norm(pmin) <= Delta else pmin * Delta / norm(pmin)


def trustreg(f, df, x0, tol):
# TRUSTREG searches for minimum using trust region method
# 	answer_ = trustreg(f, df, x0, tol)
#   INPUT ARGUMENTS
#   f  - objective function
#   df - gradient
# 	x0 - start point
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords, radii]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of statistics
#   radii - array of trust regions radii

    coords = []
    radii = []
    neval = 0
    
    kmax = 1000
    xmin = x0
    Dk = 1
    Bk = np.identity(2)
    gk = df(xmin)
    coords.append(xmin)
    radii.append(Dk)
    dx = tol + 1
    
    m = lambda p: f(xmin) + sum(p * gk) + sum(p * np.dot(Bk, p)) / 2

    while(norm(dx) >= tol) and (neval < kmax):
        pk = doglegsearch(m, gk, Bk, Dk, tol) 
        rho = (f(xmin) - f(xmin + pk)) / (f(xmin) - m(pk))
        
        if rho > 0.1:
            dx = pk
            xmin = xmin + dx
            dg = df(xmin) - gk
            gk = df(xmin)
            
            Bk = Bk + np.dot(dg, dg.T)/np.dot(dg.T, dx) - (np.dot(Bk,dx).dot(np.dot(dx.T,Bk.T)))/(np.dot(dx.T,Bk).dot(dx))
            
        coords.append(xmin)
        radii.append(Dk)
        neval += 1
        
        if rho < 0.25:
            Dk = Dk * 0.25
        elif rho > 0.75 and norm(pk) == Dk:
            Dk = min([2 * Dk, 1])

    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords, radii]
    return answer_



def main():
    
    print("Himmelblau function:")
    x0 = np.array([[2.0], [1.0]])
    tol = 1e-3
    [xmin, f, neval, coords, rad] = trustreg(fH, dfH, x0, tol)  # h - функция Химмельблау
    print(xmin, f, neval)

    print("Rosenbrock function:")
    x0 = np.array([[-2], [0]])
    tol = 1e-3
    [xmin, f, neval, coords, rad] = trustreg(fR, dfR, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)


if __name__ == '__main__':
    main()
