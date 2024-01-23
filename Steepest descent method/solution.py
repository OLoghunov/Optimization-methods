import numpy as np
import sys
from numpy.linalg import norm

def scpr(xmin, a, df):
    der = df(xmin)
    return sum(der * -df(xmin - a * der))

def bsearch(df, interval, tol):
# searches for minimum using bisection method
# arguments: bisectionsearch(f,df,interval,tol)
# f - an objective function
# df -  an objective function derivative
# interval = [a, b] - search interval
#tol - tolerance for both range and function value
# output: [xmin, fmin, neval, coords]
# xmin - value of x in fmin
# fmin - minimul value of f
# neval - number of function evaluations
# coords - array of x values found during optimization
    
    a = interval[0]
    b = interval[1]
    coords = []
    neval = 1
    
    xmin = a
    
    while ((np.abs(a - b) > tol) and (abs(df(a)) > tol)) :
        xmin = (a + b) / 2
        coords.append(xmin)
        if (df(xmin) > 0) : 
            b = xmin
        else :
            a = xmin
        neval += 1
    
    answer_ = xmin
    return answer_


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
    
    answer_ = xmin
    return answer_


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

def sdsearch(f, df, x0, tol):

# SDSEARCH searches for minimum using steepest descent method
# 	answer_ = sdsearch(f, df, x0, tol)
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

    xmin = x0
    
    def f1dim(al):
        return f(xmin - al * df(x0))
    
    fidim = lambda al : f(xmin - al * df(x0))
    df1dim = lambda alpha: scpr(xmin, alpha, df)

    interval = [0, 1]
    neval = 1
    coords = []
    kmax = 1000
    
    while (neval < kmax) :
        al = goldensectionsearch(f1dim, interval, tol)
        xmin = x0 - al * df(x0)
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
    x0 = np.array([[1.3], [2.0]])
    tol = 1e-3
    [xmin, f, neval, coords] = sdsearch(fH, dfH, x0, tol)  # h - функция Химмельблау
    print(xmin, f, neval)


    print("Rosenbrock function:")
    x0 = np.array([[1.0], [-2.0]])
    tol = 1e-7
    [xmin, f, neval, coords] = sdsearch(fR, dfR, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)


if __name__ == '__main__':
    main()