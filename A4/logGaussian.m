% for a single instance of k
function [r] = logGaussian(X, p, mu, S)
    exponent = -1/2 * (X - mu).^2 * (1./S)';
    a = log(p) - 1/2 * sum(log(S));
    r = a + exponent - size(X, 2) / 2 * log(2*pi);
end