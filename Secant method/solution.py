import numpy as np

def f(x): return x**2 -  10*np.cos(0.3*np.pi*x) - 20
def df(x): return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)

def ssearch(interval,tol):
# SSEARCH searches for minimum using secant method
# 	answer_ = ssearch(interval,tol)
#   INPUT ARGUMENTS
# 	interval = [a, b] - search interval
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of x values found during optimization    

    a = interval[0]
    b = interval[1]
    coords = []
    neval = 1
    
    while (np.abs(b - a) > tol) :
        xmin = b - df(b) * (b - a) / (df(b) - df(a))
        coords.append([xmin, a, b])
        neval += 1
        if (df(xmin) > 0) :
            b = xmin
        else :
            a = xmin

    fmin = f(xmin)
    
    answer_ = [xmin, fmin, neval, coords]
    return answer_



def main():
    print("Find:")
    interval = [-2, 5]
    tol = 1e-6
    [xmin, f, neval, coords] = ssearch(interval,tol)
    print([xmin, f, neval])


if __name__ == '__main__':
    main()