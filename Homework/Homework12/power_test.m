function power_test

A = randn(2000,2000);

dd = ones(2000,1);
dd(1) = 40;

A = A * diag(dd) / A;

tic
 [X, lambda] = eig(A);
toc

Nmax = 100;

tic
[lam, v]= powerIt(A,Nmax);
toc

err = abs(lam-max(dd));

N = 1:Nmax;

semilogy(N,err,'o-', N,(1/dd(1)).^N,'r')



keyboard

return


function [lam, v]= powerIt(A,Nmax)

n = size(A,1);

q = rand(n,1);
q = q/norm(q);
lam = zeros(Nmax,1);

for j = 1:Nmax
    
   z = A*q;
   q = z/norm(z);
   lam(j) = q'*A*q;
end

v= q;

return