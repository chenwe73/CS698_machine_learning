clear; clc

ktype = 'laplace';
p = 5;
ell = 1;
sigma = 1;

T = rand(1)*2 - 1;

tt = 1500;
T = [-1:2/tt:1]';

tt = length(T);
n = 15;


avg = @(x) zeros(size(x));
avg = @(x) x.^2;


switch ktype
    case 'gaussian'
        d = repmat(T .^ 2, 1, tt);
        D = d + d' - 2*(T*T');
        kappa = exp(-D/sigma) / ell;
    case 'laplace'
        d = repmat(T .^ 2, 1, tt);
        D = d + d' - 2*(T*T');
        kappa = exp(-sqrt(D)/sigma) / ell;
    case 'polynomial'
        kappa = (T*T').^p;
    case 'h_polynomial'
        kappa = (1+T*T').^p;
    otherwise
        kappa = @(s, t) exp(-norm(s-t)^p/2/sigma^2) / ell;
end


gp(T, n, avg, kappa)