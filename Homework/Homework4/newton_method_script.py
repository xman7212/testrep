################################################################################
# This python script presents examples regarding the newton method and its
# application to 1D nonlinear root-finding, as presented in class.
# APPM 4650 Fall 2021
################################################################################
# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;

# First, we define a function we will test the Newton method with. For each
# function we define, we also define its derivative.
# Our test function from previous sections
def fun(x):
    return x + np.cos(x)-3;
def dfun(x):
    return 1 - np.sin(x);

################################################################################
# We now implement the Newton method
def newton_method(f,df,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0;
    r=xn
    rn=np.array([x0]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            print('\n derivative at initial guess is near 0, try different x0 \n');
    else:
        n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)));

            pn = - fn/dfn; #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn;

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            dfn=df(xn);
            fn=f(xn);
            nfun+=2;

        r=xn;

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)
################################################################################
# Now, we apply this method to our test function
def mod_newton_method(f,df,x0,tol,nmax,verb=False):
    
    r=x0
    rn=np.array([x0])
    rnm=rn
    nfunm=2
    print(f(r))
    print(f(rn[-1]))
    r,rn,nfun = newton_method(f,df,x0,tol,nmax,verb)
    fnew = lambda x: f(x)/(x-r)
    f = fnew
    rnm = np.append(rnm,rn)
    nfunm = nfunm+nfun
    
    print(fnew(r))
    print(fnew(rn[-1]))
    
    while fnew(r)-fnew(rn[-1]) != 0 :
        
        r,rn, nfun = newton_method(f,df,x0,tol,nmax,verb)
        fnew = lambda x: f(x)/(x-r)
        f = fnew
        rnm = np.append(rnm,rn)
        nfunm = nfunm+nfun
    return (r,rnm,nfunm)
