# import libraries
import numpy as np
    
def driver():

# test functions 
     f1 = lambda x: 1+0.5*np.sin(x)
# fixed point is alpha1 = 1.4987....

     f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09... 

     Nmax = 100
     tol = 1e-6

# test f1 '''
     x0 = 0.0
     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
    
#test f2 '''
     x0 = 0.0
     [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar)
     print('f2(xstar):',f2(xstar))
     print('Error message reads:',ier)



# define routines
def fixedpt(f,x0,tol,Nmax,iterative):

     ''' x0 = initial guess''' 
     ''' Nmax = max number of iterations'''
     ''' tol = stopping tolerance'''
     xlist = [x0]
     print(Nmax)
     count = 0
     while (count <Nmax):
          count = count +1
          x1 = f(x0)
          print(xlist)
          xlist.append(x1)
          if (abs(x1-x0) <tol):
               if iterative == True:
                    return [xlist,count]
               xstar = x1
               ier = 0
               return [xstar,ier]
          x0 = x1
     if iterative == True:
          return xlist,count
     print(xstar)
     xstar = x1
     ier = 1
     return [xstar, ier]
    
def fixedptbetter(g,p0,tol=1e-10,max_iter=100):
     approx = [p0]

     for i in range(max_iter):
          p1=g(p0)
          approx.append(p1)

          if abs(p1-p0)<tol:
               break
          p0=p1

     return approx

def convergencetest(p,pList):
     p2 = pList[0][len(pList)-1]
     p1 = pList[0][len(pList)-2]
     print(p2)
     print(p1)
     test1=abs(p2-p)/((abs(p1-p)))
     test2=abs(p2-p)/((abs(p1-p))**2)
     if test1< 1:
         return 1
     elif test2<1:
         return 2
     else:
         return 0