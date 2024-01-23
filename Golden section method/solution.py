import numpy as np
def f(x): return (x - 3)**2- 3*x + x**2 - 40

def gsearch(interval,tol):
# GOLDENSECTIONSEARCH searches for minimum using golden section
# 	[xmin, fmin, neval] = GOLDENSECTIONSEARCH(f,interval,tol)
#   INPUT ARGUMENTS
# 	f is a function
# 	interval = [a, b] - search interval
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of statistics,  coord[i][:] =  [x1,x2, a, b]

    phi = (1 + np.sqrt(5)) / 2
    neval = 1
    coords = []
    a = interval[0]
    b = interval[1]
    L = b - a
    la = b - L / phi
    mu = a + L / phi
    fla = f(la)
    fmu = f(mu)
    
    while (L > tol) :
        coords.append([mu, la, a, b])
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
        neval += 1
    
    fmin = f(xmin)
    
    answer_ = [xmin, fmin, neval, coords]
    return answer_

def main():
    print("Find:")
    interval = [-2, 10]
    tol = 1e-10
    [xmin, fmin, neval, coords] = gsearch(interval,tol)
    print([xmin, fmin, neval])

if __name__ == '__main__':
    main()