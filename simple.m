% optical artifical neural network

clear;
parameters;

InputLayer  = imageInputLayer([dimx, dimy, 1], 'Name', 'input_layer');
ResizeLayer = CustomResizeLayer(1, 'resize_layer', Nx, Ny, k, lvalue, P);
KernelLayer = CustomKernelLayer(1, 'kernel_layer', kernel, rate);

Prop1       = CustomFFTPropagationLayer('prop1_layer', Nx, Ny, d1, w1);
Prop2       = CustomFFTPropagationLayer('prop2_layer', Nx, Ny, d2, w2);

AbsLayer   = CustomAbsoluteLayer(2, 'absolute_layer', lvalue);

Nonlinear1Layer  = CustomReLULayer(2, 'nonlinear1_layer', lvalue, sx, sy, sc, sz);
Nonlinear2Layer  = CustomReLULayer(2, 'nonlinear2_layer', lvalue, sx, sy, sc, sz);

Flatten         = CustomFlattenLayer(1, 'flatten', Nx, Ny, nx, ny, r1, r2, lvalue);

SoftMax         = softmaxLayer('Name', 'softmax_layer');

Classification  = classificationLayer('Name', 'classification_layer');

layers = [
    InputLayer
    ResizeLayer
    KernelLayer

    Prop1
    Prop2

    AbsLayer
    Nonlinear1Layer
    Nonlinear2Layer

    Flatten
    SoftMax
    Classification];

lgraph = layerGraph();

for i=1:length(layers)
    lgraph = addLayers(lgraph, layers(i));
end

lgraph = connectLayers(lgraph, 'input_layer', 'resize_layer');
lgraph = connectLayers(lgraph, 'resize_layer', 'kernel_layer');
lgraph = connectLayers(lgraph, 'kernel_layer/out1', 'prop1_layer/in1');
lgraph = connectLayers(lgraph, 'kernel_layer/out2', 'prop1_layer/in2');

lgraph = connectLayers(lgraph, 'prop1_layer/out1', 'nonlinear1_layer/in1');
lgraph = connectLayers(lgraph, 'prop1_layer/out2', 'nonlinear1_layer/in2');

lgraph = connectLayers(lgraph, 'nonlinear1_layer/out1', 'prop2_layer/in1');
lgraph = connectLayers(lgraph, 'nonlinear1_layer/out2', 'prop2_layer/in2');

lgraph = connectLayers(lgraph, 'prop2_layer/out1', 'nonlinear2_layer/in1');
lgraph = connectLayers(lgraph, 'prop2_layer/out2', 'nonlinear2_layer/in2');

lgraph = connectLayers(lgraph, 'nonlinear2_layer/out1', 'absolute_layer/in1');
lgraph = connectLayers(lgraph, 'nonlinear2_layer/out2', 'absolute_layer/in2');
lgraph = connectLayers(lgraph, 'absolute_layer', 'flatten');

% softmax
lgraph = connectLayers(lgraph, 'flatten', 'softmax_layer');

% Classify
lgraph = connectLayers(lgraph, 'softmax_layer', 'classification_layer');

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 750, 'randomize');

options = trainingOptions('adam', ...
    InitialLearnRate=learnRate,...
    MaxEpochs=numEpochs,...
    Shuffle='every-epoch',...
    ValidationData=imdsValidation,...
    ValidationFrequency=50,...
    Verbose=true,...
    Plots='training-progress',...
    MiniBatchSize=miniBatchSize);

plot(lgraph);
