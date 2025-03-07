import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def eval_lagrange(xeval,xint,yint,N):
    
    wj = np.ones(N+1)
    
    phi=1

    peval = 0
    
    for n in range(N+1):
        phi=phi*(xeval-xint[n])
    #print("phi = ", phi)


    for j in range(N+1):
        
        for i in range(N+1):
            if (i != j):
                wj[j] = wj[j] *(1/(xint[j]-xint[i]))

        peval = peval + phi*wj[j]*yint[j]/(xeval-xint[j])
                
    #print("peval =",peval)
    yeval = 0.
    
  
    return(peval)
  