% gaussian mixture model epectation maximization
% X  [n*d]  = data 
% K  [1]    = number of classes 
% p  [k]  = probability of each class 
% mu [k*d]  = gaussian mean
% S  [k*d]  = gaussian covariance (diagonal, k * 1D-vector)
% l  [iter] = negative log-likelihood
% r [n*k] = posterior probability that Xi belongs to class k (given the model)
function [p, mu, S, l] = GMM_EM(X, K, p_init, mu_init, S_init)
    MAXITER = 500;
    TOL = 10e-5;
    EPSILON = 1E-5;
    
    p = p_init;
    mu = mu_init;
    S = S_init;
    
    [n, d] = size(X);
    
    for iter = 1:MAXITER
        if (d == 1)
            draw(X, mu, S);
        end
        
        % fix the parameters (mu, S) and solve for the posterior distribution 
        % for the hidden variables (r)
        for k = 1:K
            log_r(:, k) = logGaussian(X, p(k), mu(k,:), S(k,:));
        end
        
        % offset to avoid overflow
        offset = 1 * max(log_r, [], 2);
        log_ri = log(sum(exp(log_r - offset), 2));
        r = exp(log_r - offset - log_ri);
        
        % compute negative log-likelihood
        l(iter) = -sum( log_ri + 1*offset );
        disp([iter, l(iter)]);
        
        if(iter > 1 && abs(l(iter) - l(iter-1)) <= TOL * abs(l(iter)))
            break;
        end
        
        % fix the posterior distribution for the hidden variables (r) and
        % optimize the parameters (mu, S)
        rk = sum(r, 1);
        p = rk ./ n;
        for k = 1:K
            temp = (r(:,k) ./ rk(k))';
            mu(k,:) = temp * X;
            S(k,:) = temp * (X.^2) - mu(k,:).^2;
        end
        
        % add epsilon to prevent division by 0;
        S = S + EPSILON;
    end
end

function [] = draw(data, mu, S)
    clf;
    hold on;
    scatter(data, zeros(1, size(data,1)));
    x = [0:0.001:1];
    for i = 1:size(mu,1)
        norm = normpdf(x,mu(i),sqrt(S(i)));
        plot(x,norm);
    end
end
