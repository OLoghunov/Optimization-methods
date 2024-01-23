import numpy as np
import sys
from numpy.linalg import norm
np.seterr(divide='ignore', invalid='ignore')


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


def nagsearch(f, df, x0, tol):
    
# NAGSEARCH searches for minimum using the Nesterov accelerated gradient method
# 	answer_ = nagsearch(f, df, x0, tol)
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

    al = 0.05
    nu = al / 10
    gamma = 0.75
    neval = 1
    coords = []
    kmax = 1000
    y0 = x0

    while (neval < kmax) :
        xmin = y0 - nu * df(y0)
        y0 = xmin + gamma * (xmin - x0)
        coords.append(xmin)
        if (norm(df(xmin)) < tol) : 
            break
        neval += 1
        x0 = xmin
    
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_
    
# def nagsearch(f, df, x0, tol):
#     neval = 0
#     coords = list()
#     xmin, ymin = x0, x0
    
#     while neval < 1000 and norm(df(xmin)) >= tol:
#         x0 = xmin
#         xmin = ymin - 5e-3 * df(ymin)
#         ymin = xmin + 0.75 * (xmin - x0)
#         neval += 1
#         coords.append(xmin)
    
#     return [xmin, f(xmin), neval, coords]



def main():
    print("Himmelblau function:")
    x0 = np.array([[0.0], [1.0]])
    tol = 1e-3
    [xmin, f, neval, coords] = nagsearch(fH, dfH, x0, tol)  # функция Химмельблау
    print(xmin, f, neval)


if __name__ == '__main__':
    main()