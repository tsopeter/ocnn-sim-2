function Y = modelPredictions(netE,mbq)

Y = [];

% Loop over mini-batches.
while hasdata(mbq)
    X = next(mbq);

    % Forward through encoder.
    Z = predict(netE,X);

    % Extract and concatenate predictions.
    Y = cat(4,Y,extractdata(Z));
end

end