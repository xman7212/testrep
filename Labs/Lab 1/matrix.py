import numpy as np
import numpy.linalg as la
import math
def driver():
	n = 100
	x = np.linspace(0,np.pi,n)
# this is a function handle. You can use it to define
# functions instead of using a subroutine like you
# have to in a true low level language.
	f = lambda x: x**2 + 4*x + 2*np.exp(x)
	g = lambda x: 6*x**3 + 2*np.sin(x)
	y = (0,1)#f(x)
	w = (1,0)#g(x)
# evaluate the dot product of y and w
	dp = dotProduct(y,w,n)
# print the output
	print(’the dot product is : ’, dp)
	return
def dotProduct(x,y,n):
# Computes the dot product of the n x 1 vectors x and y
	dp = 0.
	for j in range(n):
	dp = dp + x[j]*y[j]
	return dp

#this is the beginning of the matrix product function
#This function is supposed to take in two matrices and use the dot product function to create a matrix product
#This function was not finished due to time constraints
def matrixProduct(A,B)
	
	C = np.zeroes(arr.shape)
	
	for a in len(
driver()
#this is a comment for testrep2
#another
#another
#another
#another

