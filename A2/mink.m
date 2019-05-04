% minimum k values
function [B,I] = mink(A,k)

    temp = A;
    B = zeros(k,1);
    I = zeros(k,1);
    for i = 1:k
        [B(i), I(i)] = min(temp);
        temp(I(i)) = inf;
    end
    
end