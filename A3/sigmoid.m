function [P] = sigmoid(Z)
    P = 1 ./ (1 + exp(-Z));
end