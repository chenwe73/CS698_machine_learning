% use my own image
function [guess] = testImage(B, Xtrain, ytrain)

    B = double(255 - B(:,:,1)) / 255;
    %image(B,'CDataMapping','scaled');
    B = reshape(B, 1, 28*28);
    [yHatB, ~] = kNN(Xtrain, ytrain, B, 1);
    guess = yHatB;

end