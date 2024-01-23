import numpy as np
import sys
from numpy.linalg import norm
np.seterr(all='warn')

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
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
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)

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
    v = (1 - x)**2 + 100*(y - x**2)**2
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
    v[0] = -2 * (1 - x) + 200 * (y - x**2)*(- 2 * x)
    v[1] = 200 * (y - x**2)
    return v

def H(X, tol, df):
    ddf = np.zeros((2, 2))
    deltaX = 0.1 * tol
    
    ddf[0][0] = ((df([X[0] + deltaX, X[1]]) - df([X[0] - deltaX, X[1]])) / (deltaX * 2))[0]
    ddf[1][1] = ((df([X[0], X[1] + deltaX]) - df([X[0], X[1] - deltaX])) / (deltaX * 2))[1]
    ddf[1][0] = ((df([X[0] + deltaX, X[1]]) - df([X[0] - deltaX, X[1]])) / (deltaX * 2))[1]
    ddf[0][1] = ddf[1][0]
    
    return ddf


def nsearch(f, df, x0, tol):
# NSEARCH searches for minimum using Newton method
# 	answer_ = nsearch(f, df, x0, tol)
#   INPUT ARGUMENTS
#   f  - objective function
#   df - gradient
# 	x0 - start point
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of statistics

    neval = 1
    coords = []
    kmax = 1000
    
    while (neval < kmax) :
        xmin = x0 + np.linalg.lstsq(-H(x0, tol, df), df(x0))[0]
        coords.append(xmin)
        if (norm(xmin - x0) < tol) :
            break
        neval += 1
        x0 = xmin
  
    fmin = f(xmin)
  
    answer_ = [xmin, fmin, neval, coords]
    return answer_



def main():
    print("Himmelblau function:")
    x0 = np.array([[-2.0], [-2.0]])
    tol = 1e-3
    
    
    [xmin, f, neval,  coords] = nsearch(fH, dfH, x0, tol) # h - функция Химмельблау
    print(xmin, f, neval)
    
    print("Rosenbrock function:")
    x0 = np.array([[-1.0], [-1.0]])
    tol = 1e-9
    [xmin, f, neval,  coords] = nsearch(fR, dfR, x0, tol) # r - функция Розенброка
    print(xmin, f, neval)


if __name__ == '__main__':
    main()