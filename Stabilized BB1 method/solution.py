import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
   
    phi = (1 + np.sqrt(5)) / 2
    a = interval[0]
    b = interval[1]
    L = b - a
    la = b - L / phi
    mu = a + L / phi
    fla = f(la)
    fmu = f(mu)
    
    while (L > tol) :
        if (fla > fmu) :
            a = la
            xmin = mu
            la = mu
            L = b - a
            mu = a + L / phi
            fla = fmu
            fmu = f(mu) 
        else :
            b = mu
            xmin = la
            mu = la
            L = b - a
            la = b - L / phi
            fmu = fla
            fla = f(la)
    
    
    answer_ = [xmin]
    return answer_



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


def bbsearch(f, df, x0, tol):

# BBSEARCH searches for minimum using stabilized BB1 method
# 	answer_ = bbsearch(f, df, x0, tol)
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
    f1dim = lambda al : f(x0 - al * df(x0))
    al = goldensectionsearch(f1dim, [0, 1], tol)
    D = 0.1

    while (neval < kmax) :
        xmin = x0 - al * df(x0)
        deltaX = abs(xmin - x0)
        deltaG = abs(df(xmin) - df(x0))

        al = np.dot(np.transpose(deltaX), deltaX) / np.dot(np.transpose(deltaX), deltaG)
        
        alStab = D/norm(df(xmin))
        al = min(al, alStab)
        
        coords.append(xmin)
        if (norm(xmin - x0) < tol) : 
            break
        neval += 1
        x0 = xmin
    
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_




def main():
    print("Rosenbrock function:")
    x0 = np.array([[2], [-1]])
    tol = 1e-9
    [xmin, f, neval, coords] = bbsearch(fR, dfR, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)
    #contourPlot()
    #bbdDraw(coords, neval)
    # Уточнить разницу в результатах - matlab - xmin 1.00000
    #                                                1.00000
    #                                           fmin 2.8911e-20
    #                                           neval 90
    # и Питон- [[1.]
    #           [1.]] [2.93450702e-20] 90
    # вероятно связано с разницей хранения знаков и округлении
    # В таком случае тесты делать на xmin или fmin?
    
if __name__ == '__main__':
    main()