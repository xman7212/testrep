import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from numpy.linalg import inv 
from numpy.linalg import norm



"""--------------------------------------------------------"""
def newton_method_nd(f,Jf,x0,tol,nmax):

    
    xn = x0
    rn = x0
    Fn = f(xn)
    n=0
    nf=1; nJ=0
    normPn=1

    while normPn>tol and n<=nmax:
        
        Jn = Jf(xn)
        nJ+=1

        
        pn = -np.linalg.solve(Jn,Fn)
        xn = xn + pn
        normPn = np.linalg.norm(pn)

        n+=1
        rn = np.vstack((rn,xn))
        Fn = f(xn)
        nf+=1

    r=xn
    return (r,rn,nf,nJ);

def SteepestDescent(x,tol,Nmax):
    
    for i in range(Nmax):
        g1 = G2(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = G2(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = G2(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = G2(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = G2(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]


