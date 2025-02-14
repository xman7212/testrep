################################################################################
# This python script presents examples regarding the secant method and its
# application to 1D nonlinear root-finding, as presented in class.
# APPM 4650 Fall 2021
################################################################################
# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;

# First, we define a function we will test the secant method with.
# Our test function from previous sections
def fun(x):
    return x + np.cos(x)-3;
def dfun(x):
    return 1 - np.sin(x);

################################################################################
# We now implement the Lazy Newton (Chord) method
def lazynewton_method(f,df,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0;
    rn=np.array([x0]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            fprintf('\n derivative at initial guess is near 0, try different x0 \n');
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
            #dfn=df(xn);
            fn=f(xn);
            nfun+=1;

        r=xn;

        if n>=nmax:
            print("Lazy Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Lazy Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)
################################################################################
################################################################################
# We now implement the Secant method
def secant_method(f,x0,x1,tol,nmax,verb=False):
    #secant (quasi-newton) method to find root of f starting with guesses x0 and x1

    #Initialize iterates and iterate list
    xnm=x0; xn=x1;
    rn=np.array([x1]);
    # function evaluations
    fn=f(xn); fnm=f(xnm);
    msec = (fn-fnm)/(xn-xnm);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if np.abs(msec)<dtol:
        #If slope of secant is too small, secant will fail. Error message is
        #displayed and code terminates.
        if verb:
            fprintf('\n slope of secant at initial guess is near 0, try different x0,x1 \n');
    else:
        n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|msec|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(msec)));

            pn = - fn/msec; #Secant step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step, update xn-1
            xnm = xn; #xn-1 is now xn
            xn = xn + pn; #xn is now xn+pn

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            fnm = fn; #Note we can re-use this function evaluation
            fn=f(xn); #So, only one extra evaluation is needed per iteration
            msec = (fn-fnm)/(xn-xnm); # New slope of secant line
            nfun+=1;

        r=xn;

        if n>=nmax:
            print("Secant method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Secant method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)
################################################################################
################################################################################
# We now implement the Newton method
def newton_method(f,df,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0;
    rn=np.array([x0]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            fprintf('\n derivative at initial guess is near 0, try different x0 \n');
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
