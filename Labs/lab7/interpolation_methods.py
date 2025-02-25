#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
파일 이름: interpolation_methods.py
작성자: 글로이정 (Chloe Chung)
작성일: 2025년 2월 25일
설명:
    이 스크립트는 다양한 보간(interpolation) 방법을 구현하고 비교합니다.
    포함된 보간 방법:
    - Vandermonde 행렬을 사용한 단항 보간(Monomial Interpolation)
    - 나눗셈 차분(Divided Differences)을 이용한 뉴턴 보간(Newton Interpolation)
    - BarycentricInterpolator를 이용한 라그랑주 보간(Lagrange Interpolation)
    
    스크립트는 각 방법의 정확도를 평가하고,
    서로 다른 노드 유형(일정 노드 및 Chebyshev 노드)을 사용한 결과와 오류 분석을 시각화합니다.

사용 방법:
    - 스크립트를 실행하여 다양한 N 값에 대한 보간 결과를 확인합니다.
    - 시각화에는 다음이 포함됩니다:
        - 보간 방법 비교
        - 각 방법의 절대 오차 그래프

의존성:
    - Python 3.x
    - NumPy
    - SciPy
    - Matplotlib

함수 설명:
    - f(x): 보간에 사용할 함수 1 / (1 + (10 * x) ** 2)를 정의합니다.
    - uni_nodes(N): [-1, 1] 구간에서 N개의 일정하게 분포된 노드를 생성합니다.
    - mon_interp(xnodes, ynodes, xev): Vandermonde 행렬을 사용하여 단항 보간을 수행합니다.
    - new_eval(xnodes, coef, xev): 주어진 x 값에서 뉴턴 다항식을 평가합니다.
    - cheshe_nodes(N): Chebyshev 노드를 생성합니다.
    - new_div_diff(xnodes, ynodes): 나눗셈 차분을 사용하여 뉴턴 보간 계수를 계산합니다.
    - lag_interp(xnodes, ynodes, xev): BarycentricInterpolator를 사용하여 라그랑주 보간을 수행합니다.
    - long_ass_function(Nvals, nt='uniform'): 보간 방법을 비교하고 결과를 시각화합니다.


버전:
    1.0
"""



#SEE ATTACHED TEXT FILE FOR HINTS/NOTES/INSTRUCTIONS AND PSEUDO CODE FILE FOR MORE INFO 

import numpy as np 
import matplotlib.pyplot as mpl
from scipy.interpolate import BarycentricInterpolator as bci

######## READ ME ############
#see the bottom for an explanation of each of my functions and the more niche functions 
#such as BarycentricInterpolator (bci)
#CHEBYSHEV NOTE -- the formula referenced in 3.2 (where it talks about Runge's) is actually 
#the formula for Chebyshev nodes which is used a lot in stats (error and oscillation reduction) 
#hence why the function is called cheshe


    
###### SHORT HAND #######
#I use mathematical shorthand bc I'm lazy
#∧= and (logical and)
#∨= or (logical or)


#construct helper functions
#f(x) 
def f(x):
    return 1 / (1 + (10 * x) ** 2) 

#uniform nodes
def uni_nodes(N):
    return np.linspace(-1, 1, N)


###### PRELAB ##########
#construct V matrix ∧ solve for monomial coeffs
#vars = xnodes, ynodes, xevals
def mon_interp(xnodes, ynodes, xev):
    V = np.vander(xnodes, increasing=True)
    coeffs = np.linalg.solve(V, ynodes)
    return np.polyval(coeffs[::-1], xev)

#evaluate newton poly @xev
def new_eval(xnodes, coef, xev):
    n = len(coef)
    result = np.zeros_like(xev, dtype=float)  
    for i in range(n - 1, -1, -1):
        result = result * (xev - xnodes[i]) + coef[i]  
    return result 
    
    
######## TECHNICAL BEGINNING OF LAB #########

#I wrote these instead of downloading his code 
#but feel free to use them

#newton divided differences (so coeffs are inc. computed)
def new_div_diff(xnodes, ynodes):
    N = len(xnodes)  
    coef = np.copy(ynodes)
    for i in range(1, N):
        for j in range(N-1, i - 1, -1):
            coef[j] = (coef[j] - coef[j - 1]) / (xnodes[j] - xnodes[j - i])  
    return coef

#lagrange interpolation
def lag_interp(xnodes, ynodes, xev): 
    interpolator = bci(xnodes, ynodes)
    return interpolator(xev)




###### EXERCISE  3.2 "....Runge's formula..." #########
#chebyshev nodes
def cheshe_nodes(N):
    return np.cos((2 * np.arange(1, N + 1) - 1) * np.pi / (2 * N))





####### BULK OF THE LAB ###########

#LOOOOOOOOOOOOOOOOOOONG >:(
def long_ass_function(Nvals, nt='uniform'): #nt=node type 
    xev = np.linspace(-1, 1, 1000)
    tvals = f(xev) #tvals=true values
    
    for N in Nvals:
        xnodes = uni_nodes(N) if nt == 'uniform' else cheshe_nodes(N)
        ynodes = f(xnodes)
        
        mon_res = mon_interp(xnodes, ynodes, xev) #monomial results
        lag_res = lag_interp(xnodes, ynodes, xev) #lagrange results
        
        new_coefs = new_div_diff(xnodes, ynodes)  
        new_res = new_eval(xnodes, new_coefs, xev) #newton results
        
        mpl.plot(xev, tvals, label='True Function', linewidth=2, linestyle='dashed')  
        mpl.plot(xev, lag_res, label=f'Lagrange (N={N})')
        mpl.plot(xev, new_res, label=f'Newton (N={N})')
        mpl.scatter(xnodes, ynodes, color='pink', zorder=5, label='Interpolation Nodes')
        mpl.legend()
        mpl.title(f'Interpolation Methods Comparison (N={N})')
        mpl.xlabel('x')
        mpl.ylabel('f(x)')
        mpl.show()
        
        #plotting error
        mpl.figure(figsize=(10, 6))
        mpl.semilogy(xev, np.abs(mon_res - tvals), label='Monomial Error')
        mpl.semilogy(xev, np.abs(lag_res - tvals), label='Lagrange Error')
        mpl.semilogy(xev, np.abs(new_res - tvals), label='Newton Error')
        mpl.legend()
        mpl.title(f'Absolute Error of Interpolation Methods (N={N})')
        mpl.xlabel('x')
        mpl.ylabel('Error')
        mpl.show()

        
        
######### ACTUALLY RUNNING CODE ########        
#uniform nodes N=2 to 10 
long_ass_function(range(2, 11), nt='uniform')

#uniform nodes for N=11 to 20 
long_ass_function(range(11, 21), nt='uniform')

#Chebyshev nodes for improved accuracy
long_ass_function(range(2, 21), nt='chebyshev')




####### FUNCTION GLOSSARY ###########
#BarycentricInterpolator(x-coords, y-coords, axis, weights) 
#USAGE: given set of points -> constructs polynomial that fits them 
#can also eval polynomials + derivs, interp. y vals, ∧ update x ∧ y vals 
#PARAMETERS: x-coords(array), y-coords(array), axis(OPTIONAL int), weights(OPTIONAL array)
#RETURNS: callable object representing the constructed polynomial interpolant

#zeros_like(x, dtype=)
#USAGE: returns array of same shape and dtype as parameters
#PARAMETERS: x: array w/ same shape ^ data type of desired return array
#RETURNS: zero matrix with the same shape and dtype as the input array

#vander(vec, N, increasing=)
#USAGE: constructs Vandermonde matrix. Cols of output = power of input vec. Bool determines order
#PARAMETERS: vec: 1d input vec, N: number of cols in output -> default square, increasing: increasing power -> default false
#RETURNS: Vandermonde matrix based on the input vector and parameters


#### MY FUNCTION GLOSSARY #########
###### My Functions Glossary ######
# f(x)
# USAGE: defines the function 1 / (1 + (10 * x) ** 2) for use in interpolation
# PARAMETERS: x (float or array of floats) 
# RETURNS: function value at x

# uni_nodes(N)
# USAGE: returns N uniformly spaced nodes in the interval [-1, 1]
# PARAMETERS: N (int) 
# RETURNS: array of N uniformly spaced nodes

# mon_interp(xnodes, ynodes, xev)
# USAGE: performs monomial interpolation using a Vandermonde matrix
# PARAMETERS: xnodes (array), ynodes (array), xev (array)
# RETURNS: interpolated values at xev

# new_eval(xnodes, coef, xev)
# USAGE: evaluates a Newton polynomial at given x values
# PARAMETERS: xnodes (array), coef (array), xev (array)
# RETURNS: polynomial evaluated at xev

# cheshe_nodes(N)
# USAGE: returns N Chebyshev nodes for interpolation
# PARAMETERS: N (int)
# RETURNS: array of N Chebyshev nodes

# new_div_diff(xnodes, ynodes)
# USAGE: calculates Newton's divided differences to generate coefficients
# PARAMETERS: xnodes (array), ynodes (array)
# RETURNS: array of divided difference coefficients

# lag_interp(xnodes, ynodes, xev)
# USAGE: performs Lagrange interpolation at given x values
# PARAMETERS: xnodes (array), ynodes (array), xev (array)
# RETURNS: interpolated values at xev

# long_ass_function(Nvals, nt='uniform')
# USAGE: performs interpolation using monomial, Newton, and Lagrange methods and plots results and errors
# PARAMETERS: Nvals (range or array of ints), nt (str: 'uniform' or 'chebyshev')
# RETURNS: None (displays plots)


# In[6]:





# In[ ]:




