import numpy as np

def f(x): return x**2 - 10*np.cos(0.3*np.pi*x) - 20
def df(x): return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)
def ddf(x): return 2 + 0.9*(np.pi**2)*np.cos(0.3*np.pi*x)

def nsearch(tol, x0):
# NSEARCH searches for minimum using Newton method
# 	answer_ = nsearch(tol,x0)
#   INPUT ARGUMENTS
# 	tol - set for bot range and function value
#	x0 - starting point
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of x values found during optimization    

    neval = 1
    coords = []
    
    while True :
        xmin = x0 - df(x0) / ddf(x0)
        coords.append(xmin)
        neval += 3
        x0 = xmin
        if (np.abs(df(xmin)) <= tol) :
            break
        
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_


def main():
    print("Find:")
    tol = 0.01
    [xmin, f, neval, coords] = nsearch(tol, 1.3)
    print([xmin, f, neval])


if __name__ == '__main__':
    main()