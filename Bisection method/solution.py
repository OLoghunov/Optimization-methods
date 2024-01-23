import numpy as np
def f(x): return 2 * (x ** 2) - 9 * x - 31
def df(x): return 4 * x - 9

def bsearch(interval,tol):
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
    
    while ((np.abs(a - b) > tol) and (np.abs(df(a)) > tol)) :
        xmin = (a + b) / 2
        coords.append(xmin)
        if (df(xmin) > 0) : 
            b = xmin
        else :
            a = xmin
        neval += 1
        
    fmin = f(xmin)
    
    answer_ = [xmin, fmin, neval, coords]
    return answer_



def main():
    print("Find:")
    interval = [-2, 10]
    tol = 1e-10
    [xmin, f, neval, coords] = bsearch(interval,tol)
    print([xmin, f, neval])

if __name__ == '__main__':
    main()