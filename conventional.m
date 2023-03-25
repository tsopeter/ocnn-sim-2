% conventional
clear;
parameters;

Rx = Nx/ratio;
Ry = Ny/ratio;
rx = nx/ratio;
ry = ny/ratio;

learnRate=1e-6;
miniBatchSize=128;

w1 = fftshift(get_propagation_distance(Rx, Ry, rx, ry, d1, wv));

InputLayer     = imageInputLayer([dimx, dimy, 1], 'Name', 'input_layer', 'Normalization', 'none');

Prop           = convolution2dLayer([Rx, Ry], 3);
relu           = reluLayer("Name", 'relu');
SoftMax        = softmaxLayer('Name', 'softmax');
Classification = classificationLayer('Name', 'classification');

layers = [
    imageInputLayer([dimx, dimy, 1], 'Normalization', 'rescale-zero-one')
    convolution2dLayer([Rx Ry], 10, 'WeightsInitializer', 'glorot')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([Rx/2, Ry/2], 10, 'WeightsInitializer', 'glorot')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([Rx/4, Ry/4], 10, 'WeightsInitializer', 'glorot')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2, "WeightsInitializer","glorot")
    SoftMax
    Classification
];

lgraph = layerGraph();
for layer=layers
    lgraph = addLayers(lgraph, layer);
end
plot(lgraph);

options = trainingOptions('adam', ...
    InitialLearnRate=learnRate,...
    MaxEpochs=numEpochs,...
    Shuffle='every-epoch',...
    L2Regularization=1e-7, ...
    ValidationData=dataTest,...
    ValidationFrequency=512,...
    Verbose=true,...
    Plots='training-progress',...
    ExecutionEnvironment='auto',...
    DispatchInBackground=true,...
    MiniBatchSize=miniBatchSize);

net = trainNetwork(dataTrain,lgraph,options);

YPred = classify(net,dataTest);
YValidation = dataTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

