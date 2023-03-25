function [loss,gradients] = modelLoss(net, X)

    [Z, mu, logSigmaSq] = forward(net, X);

% Forward through encoder.
[Z,mu,logSigmaSq] = forward(netE,X);

% Forward through decoder.
Y = forward(netD,Z);

% Calculate loss and gradients.
loss = elboLoss(Y,X,mu,logSigmaSq);
[gradientsE,gradientsD] = dlgradient(loss,netE.Learnables,netD.Learnables);

end