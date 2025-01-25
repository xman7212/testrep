# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# <h2>Problem 1

# %% [markdown]
# <h7>
# Plot
# 
# 
# $P(x)=(x-2)^9$  
# 
# Also represented by:  
# 
# $P(x)=x^9-18x^8+144x^7-672x^6+2016x^5-4032x^4+5376x^3-4608x^2+2304x-512$

# %%
#setting up x for problem 1
x=np.arange(1.920, 2.081, 0.001)

# %%
#inputting the two versions of P(x)

p_coef = x**9-18*x**8+144*x**7-672*x**6+2016*x**5-4032*x**4+5376*x**3-4608*x**2+2304*x-512
p_9 = (x-2)**9

# %%
plt.plot(x,p_coef, label = 'Coefficient Method')
plt.plot(x,p_9, label = "Original (x-2)^9 Method")
plt.legend()

# %% [markdown]
# This discrepancy is due to the number of subtraction operations that are done. In the coefficient method, we do 5 subtractions, each losing some information. For the original method, we only do one, losing much less information. Therefore the original method is better and more "correct" (although it also still has some error, just much less than the coefficient method).

# %% [markdown]
# <h2> Problem 2

# %% [markdown]
# <h5>Part i:

# %%

#Establishing a "small" number as a
a = 0.00000001

#calculating the output with the original function 
# and a newer optimized version
original1 = np.sqrt(a+1)-1
new1 = (a)/(np.sqrt(a+1)+1)

# %%
print(original1)
print(new1)

# %% [markdown]
# Here the original method led to floating point errors but the optimized version did not. This was done by multiplying the top and bottom of the original equation by the conjugate of the binomial to remove any subtraction.

# %%
#Establishing a "small" number as b
b = 0.00000002

#calculating the output with the original function 
# and a newer optimized version
original2= np.sin(a)-np.sin(b)
new2 = 2*np.cos((a+b)/2)*np.sin((a-b)/2)


# %%
print(original2)
print(new2)

# %% [markdown]
# Here the original method once again led to errors, as the "correct" answer here is the second, which was found by a modified function taking advantage of trigonometric functions.

# %%
#calculating the output with the original function 
# and a newer optimized version

original3 = (1-np.cos(a))/np.sin(a)
new3 = np.sin(x)


# %%
print(original3)
print(new3)

# %% [markdown]
# Here the original method actually gave zero, which is not correct for the input, but the modified version, which took advantage of conjugates and trigonometric identities worked and gave the correct output.

# %% [markdown]
# <h2>Problem 3

# %% [markdown]
# <h5>This question asks us to use the taylow poynomial approximation of a function to find and investigate error and the bounds on it.

# %% [markdown]
# a)
# 
# Here the taylor polynomial 
# 
# $P_2(x)=1+x-(x^2)/2$
# 
# was used to approximate $f(0.5)$ to compare the values and find their error.

# %%
#This is where the different functions
#are defined for use later

c=0.5
P2=1+c-(c**2)/2
f=(1+c+c**3)*np.cos(c)
#here the max error is calculated using
max_error = (0.5*(-1.625)*np.sin(0.5)+0.5*1.75*np.cos(0.5))/(1.625*np.cos(0.5))

# %%
print("P2(0.5)=",P2)
print("f(0.5)=",f)
print("The error is:",f-P2)
print("The max error is:", max_error)


# %% [markdown]
# <h5>These errors work, as the found error is within the bound of the maximum calculated.

# %% [markdown]
# <h5>b)Find a bound for the error in general as a function of x.
#   
# 
#     
# 
#     
# In this case, the bound for the error is:
# 
# 
# $|f(x)-P_2(x)|\leq-(x^3-x-1)sin(x)+(3x^2+1)cos(x)$
# 
# Which was found from:
# 
# $|Error|\leq f'(x)(x/f(x))$

# %% [markdown]
# <h5>c)
# 
# Approximate the integral of $f(x)$ from 0 to 1 using $P_2$.

# %% [markdown]
# The integral $\int_{0}^{1}P_2(x)dx=1+0.5-1/6=4/3$

# %% [markdown]
# <h5>d)
# 
# The approximate error should be roughly equal to the magnitude of the integral of the next term of the taylor series. So:
# 
# $\int_{0}^{1}0.5x^3dx=0.125=Error$

# %% [markdown]
# <h2>Problem 4

# %%
a = 1
b = -56
c = 1

#r1 is the negative root, r2 is the positive root


#from calculator
r1 = 55.98213716
r2 = 0.017862841


#With 3 decimal point assumption
r1rounded = (-b+round(np.sqrt(b**2-4*a*c),3))/2*a
r2rounded = (-b-round(np.sqrt(b**2-4*a*c),3))/2*a

relError1 = abs(r1-r1rounded)/abs(r1)
relError2 = abs(r2-r2rounded)/abs(r2)


print("The calculated first root is:",r1rounded)
print("The calculated second root is:",r2rounded)
print("The relative error for the first root is:", relError1)
print("The relative error for the second root is:", relError2)

# %% [markdown]
# <h5>Here you can see that the errors have vastly different orders of magnitude.

# %% [markdown]
# <h5>b)
# 
# Find and apply a "better" approximation using $(x-r_1)(x-r_2)$

# %% [markdown]
# The better relations are:
# 
# $r_1+r_2=-b/a$
# 
# &
# 
# $r_1r_2=c/a$
# 
# These are obtained by dividing the entire polynomial by a, then noting that when foiling out $(x-r_1)(x-r_2)$ these relations occur.

# %%
#Applying the better approximations:

r2new = c/(a*r1rounded)

r2newerror = (r2-r2new)/r2

print("The new second root is:", r2new)
print("The new relative error for the second root is:", r2newerror)

# %% [markdown]
# <h2>Problem 5:
# 
# 

# %% [markdown]
# 
# Consider two inputs of the form $\tilde{x}=x+ \Delta x$
# 
# When added or subtracted, their respecive errors $\Delta x$ also add or subtract.

# %% [markdown]
# <h5>a) Find the upper bounds on the absolute error:
# 
# $|\Delta y|$
# 
# And relative error:
# 
# $|\Delta y|/|y|$

# %% [markdown]
# The upper bound on the absolute error would be:
# 
# $|\Delta y| \leq |\Delta x_1|+|\Delta x_2|$

# %% [markdown]
# And the upper bound on the relative error is:
# 
# $|\Delta y|/|y| \leq (|\Delta x_1|+|\Delta x_2|)/|y|$

# %% [markdown]
# This shows that the relative error will be large when $y<<|\Delta x_1|+|\Delta x_2|$

# %% [markdown]
# <h5>b)
# 
# We find through trig identities that 
# 
# $cos(x+\delta)-cos(x)$ 
# 
# turns into 
# 
# $-2sin((2x+\delta)/2)sin(\delta/2)$
# 
# 

# %%

#establishing the two values for x
# (I found that 10^20 had a more interesting graph than 10^6)
x1 = np.pi
x2 = 1e20

#Establishing delta values
delta = np.array([1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])

#the difference for x1
newx1 = (-2*np.sin((2*x1+delta)/2)*np.sin(delta/2))
cosOnlyx1 = np.cos(x1+delta)-np.cos(x1)
newx2 = (-2*np.sin((2*x2+delta)/2)*np.sin(delta/2))
cosOnlyx2 = np.cos(x2+delta)-np.cos(x2)
differencePi = newx1-cosOnlyx1
difference6 = newx2-cosOnlyx2
#the difference for x2



print(differencePi)
print(difference6)
plt.scatter(delta,differencePi, label = "x=pi")
plt.scatter(delta,difference6, label = "x=1e20")
plt.legend()
plt.xlabel("Delta")
plt.ylabel("Difference")
plt.xscale('log')

# %% [markdown]
# <h5>c)
# Create a taylor expansion based algorithm to approximate the previous cosine function.

# %% [markdown]
# I decided to approximate it with:
# 
# $f(x+\delta)-f(x)=\delta f'(x)+(\delta^2f''(x))/2$
# 
# 
# As it keeps the delta squared values but not any past. Past this the amount of data lost should be around the same scale as the next taylor series component, which will have a $\delta^3$ out front, which is extremely small.

# %%

#These are the two new calculations using the new algorithm

newAlgo1 = -delta*np.sin(x1)-(delta**2/2)*np.cos(x1)
newAlgo2 = -delta*np.sin(x2)-(delta**2/2)*np.cos(x2)

# %%
#here the new algorithm is tested against the old one to see how well they agree

plt.plot(delta,newAlgo1-newx1, label = 'x=pi')
plt.plot(delta,newAlgo2-newx2, label = "x=1e20")
plt.legend()
plt.xscale('log')

# %%
#Here I just plotted the functions themselves as
#  I wanted to see what shapes they were


plt.plot(delta, newAlgo1, label = 'a')
plt.plot(delta,newx1, label = 'n')
plt.legend()
plt.xscale("log")

# %%



