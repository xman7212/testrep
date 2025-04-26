## module householder
''' d,c = householder(a).
    Householder similarity transformation of matrix [a] to 
    the tridiagonal form [c\d\c].

    p = computeP(a).
    Computes the acccumulated transformation matrix [p]
    after calling householder(a).
'''    
from numpy import dot,diagonal,outer
from math import sqrt

def householder(a): 
    n = len(a)
    
    for i in range(n-2):
        u = a[i+1:n,i]
        print(u)
        uMag = sqrt(dot(u,u))
        
        u[0] = u[0] + uMag
        h = dot(u,u)/2
        v = dot(a[i+1:n,i+1:n],u)/h
        g = dot(u,v)/(2*h)
        v = v - g*u
        a[i+1:n,i+1:n] = a[i+1:n,i+1:n] - outer(v,u) - outer(u,v)
        a[i,i+1] = -uMag
    return diagonal(a),diagonal(a,1)


      
                

