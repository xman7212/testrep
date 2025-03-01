import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

def driver():

    #Example 1: Contractive map at origin + non-linear perturbation
    M1 = np.array([[0.8,0],[0,0.4]]);

    def F1(x):
        return M1@x - x + 0.1*np.array([x[1]**3 + 0.1*np.sin(17*x[0]),x[0]**3]);
    def JF1(x):
        return M1 - np.eye(2) + 0.1*np.array([[1.7*np.cos(17*x[0]),3*x[1]**2],[3*x[1]**2,0]]);

    def G1(x):
        return x + F1(x);
    def JG1(x):
        return JF1(x) + np.eye(2);

    x0 = np.array([1,1]);
    (r,rn,nf)=fixed_point_method_nd(G1,JG1,x0,1e-15,1000,verb=True);
    print(r)

    plt.plot(rn[:,0],rn[:,1],'-.',label="FPI (x1_n,x2_n)")
    plt.title('Fixed Point Iteration: Contractive');
    plt.legend();
    plt.show();

    numN = rn.shape[0];
    err = np.log10(np.max(np.abs(rn[0:(numN-1)]-r),1));
    plt.plot(np.arange(numN-1),err,label="Log Error FPI");
    plt.title('Log Error vs n Fixed Point Iteration: Contractive');
    plt.legend();
    plt.show();

    ###############################################################
    # Non contractive

    M2 = np.array([[1.01,0],[0,0.5]]);
    def F2(x):
        return M2@x - x + 0.1*np.array([x[1]**3,x[0]**3]);
    def JF2(x):
        return M2 - np.eye(2) + 0.1*np.array([[0,3*x[1]**2],[3*x[1]**2,0]]);
    def G2(x):
        return x + F2(x);
    def JG2(x):
        return JF2(x) + np.eye(2);

    x0 = np.array([0.1,0.1]);
    (r,rn,nf)=fixed_point_method_nd(G2,JG2,x0,1e-15,275,verb=True);
    print(r)
    r = np.array([0.0,0.0]); #true solution

    plt.plot(np.log10(rn[:,0]),np.log10(rn[:,1]),'-.',label="FPI (x1_n,x2_n)")
    plt.title('Fixed Point Iteration: Non Contractive');
    plt.legend();
    plt.show();

    numN = rn.shape[0];
    err = np.log10(np.max(np.abs(rn[0:(numN-1)]-r),1));
    plt.plot(np.arange(numN-1),err,label="Log Error FPI");
    plt.title('Log Error vs n Fixed Point Iteration: Non Contractive');
    plt.legend();
    plt.show();

    ##################################################################
    # Cycling behavior

    #M3 = np.array([[1/np.sqrt(2),-1/np.sqrt(2)],[1/np.sqrt(2),1/np.sqrt(2)]]);
    th0 = np.pi/4;
    M3 = np.array([[np.cos(th0),-np.sin(th0)],[np.sin(th0),np.cos(th0)]]);

    def F3(x):
        return M3@x - x + 0.1*np.array([x[1]**3,x[0]**3]);
    def JF3(x):
        return M3 - np.eye(2) + 0.1*np.array([[0,3*x[1]**2],[3*x[1]**2,0]]);
    def G3(x):
        return x + F3(x);
    def JG3(x):
        return JF3(x) + np.eye(2);

    x0 = np.array([0.1,0.1]);
    (r,rn,nf)=fixed_point_method_nd(G3,JG3,x0,1e-15,500,verb=True);
    print(r)
    r = np.array([0.0,0.0]); #true solution

    plt.plot(rn[:,0],rn[:,1],'-.',label="FPI (x1_n,x2_n)")
    plt.title('Fixed Point Iteration: Orbit');
    plt.legend();
    plt.show();

    numN = rn.shape[0];
    err = np.log10(np.max(np.abs(rn[0:(numN-1)]-r),1));
    plt.plot(np.arange(numN-1),err,label="Log Error FPI");
    plt.title('Log Error vs n Fixed Point Iteration: Orbit');
    plt.legend();
    plt.show();

    #######################################################################
    # Spiral in

    M4 = np.array([[0.9,0],[0,0.8]])@M3;
    def F4(x):
        return M4@x - x + 0.1*np.array([x[1]**3,x[0]**3]);
    def JF4(x):
        return M4 - np.eye(2) + 0.1*np.array([[0,3*x[1]**2],[3*x[1]**2,0]]);
    def G4(x):
        return x + F4(x);
    def JG4(x):
        return JF4(x) + np.eye(2);

    x0 = np.array([1,1]);
    (r,rn,nf)=fixed_point_method_nd(G4,JG4,x0,1e-15,500,verb=True);
    print(r)
    r = np.array([0.0,0.0]); #true solution

    plt.plot(rn[:,0],rn[:,1],'-.',label="FPI (x1_n,x2_n)")
    plt.title('Fixed Point Iteration: Spiral In');
    plt.legend();
    plt.show();

    numN = rn.shape[0];
    err = np.log10(np.max(np.abs(rn[0:(numN-1)]-r),1));
    plt.plot(np.arange(numN-1),err,label="Log Error FPI");
    plt.title('Log Error vs n Fixed Point Iteration: Spiral In');
    plt.legend();
    plt.show();

    #############################################################
    # Examples system of 2 equations (circles)

    def fun(x):
        v = np.array([(x[0]-1)**2 + x[1]**2 - 1,
                 -(x[0]-2)**2 - (x[1]-1)**2 + 1])
        return v;
    def Jfun(x):
        M = np.array([[2*(x[0]-1),2*x[1]],[-2*(x[0]-2),-2*(x[1]-1)]]);
        return M;

    s = np.array([[0,-0.25],[-0.25,0]]);

    def gfun(x):
        v = s@fun(x) + x;
        return v;
    def Jgfun(x):
        M = s@Jfun(x)+np.eye(2);
        return M;

    plt.rcParams['figure.figsize'] = [5, 5];
    th=np.linspace(0,2*np.pi,100);
    plt.plot(np.cos(th)+1,np.sin(th));
    plt.plot(np.cos(th)+2,np.sin(th)+1,'r');
    plt.title("Intersection of two circles example");
    plt.show();

    x0 = np.array([0,0]);
    (r,rn,nf)=fixed_point_method_nd(gfun,Jgfun,x0,1e-15,500,verb=True);
    th=np.linspace(0,2*np.pi,100);
    plt.plot(np.cos(th)+1,np.sin(th));
    plt.plot(np.cos(th)+2,np.sin(th)+1,'r')
    plt.plot(rn[:,0],rn[:,1],'-o');
    plt.title("Fixed Point Iteration results");
    plt.show();

    numN = rn.shape[0];
    err = np.log10(np.max(np.abs(rn[0:(numN-1)]-r),1));
    plt.plot(np.arange(numN-1),err,label="Log Error FPI");
    plt.title('Log Error vs n Fixed Point Iteration: Circle Example');
    plt.legend();
    plt.show();


def fixed_point_method_nd(G,JG,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Gn = G(xn); #function value vector
    n=0;
    nf=1;  #function evals

    if verb:
        print("|--n--|----xn----|---|G(xn)-xn|---|");

    while np.linalg.norm(Gn-xn)>tol and n<=nmax:

        if verb:
            rhoGn = np.max(np.abs(np.linalg.eigvals(JG(xn))));
            print("|--%d--|%1.7f|%1.15f|%1.2f|" %(n,np.linalg.norm(xn),np.linalg.norm(Gn-xn),rhoGn));

        # Fixed Point iteration step
        xn = Gn;

        n+=1;
        rn = np.vstack((rn,xn));
        Gn = G(xn);
        nf+=1;

        if np.linalg.norm(xn)>1e15:
            n=nmax+1;
            nf=nmax+1;
            break;

    r=xn;

    if verb:
        if n>=nmax:
            print("Fixed point iteration failed to converge, n=%d, |G(xn)-xn|=%1.1e\n" % (nmax,np.linalg.norm(Gn-r)));
        else:
            print("Fixed point iteration converged, n=%d, |G(xn)-xn|=%1.1e\n" % (n,np.linalg.norm(Gn-r)));

    return (r,rn,n);

# Execute driver
driver()
