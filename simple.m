% optical artifical neural network

clear;

disp("Getting data...");
Nx     = 128;
Ny     = 128;
ratio  = 2;
ix     = round(Nx/ratio);
iy     = round(Ny/ratio);
nx     = 21e-3;
ny     = 21e-3;
d1     = 50e-2;
d2     = 15e-2;
wv     = 1550e-9;
a0     = 20;
r1     = nx/6;
r2     = nx/35;
rate   = 1;
learningRate = 1e-8;
numTrainFiles = 750;
lvalue = 0;
sx     = 2;
sy     = 1;
sc     = 0.1;
sz     = 0.2;

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% get size of images
img = readimage(imds, 1);
[dimx, dimy] = size(img);

[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');

% get the interpolation value k
kx = log2(double(ix - dimy)/double(dimy - 1))+1;
ky = log2(double(iy - dimx)/double(dimx - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);

w1 = get_propagation_distance(Nx, Ny, nx, ny, d1, wv);
w2 = get_propagation_distance(Nx, Ny, nx, ny, d2, wv);

kernel = internal_random_amp(Nx, Ny);

InputLayer  = imageInputLayer([dimx, dimy, 1], 'Name', 'input_layer');
ResizeLayer = CustomResizeLayer(1, 'resize_layer', Nx, Ny, k, lvalue);
CResize     = resize2dLayer("Name", 'resize_layer', 'OutputSize', [Nx, Ny]);
KernelLayer = CustomKernelLayer(1, 'kernel_layer', kernel, rate);

Prop1       = CustomPropagationLayer('prop_layer', Nx, Ny, d1, w1);

AbsLayer    = CustomAbsoluteLayer(2, 'absolute_layer', lvalue);

Nonlinear1Layer  = reluLayer('Name', 'nonlinear1_layer');
Nonlinear2Layer  = reluLayer('Name', 'nonlinear2_layer');

BtcNorm1 = batchNormalizationLayer('Name', 'norm1_layer');
BtcNorm2 = batchNormalizationLayer('Name', 'norm2_layer');

Flatten         = CustomFlattenLayer(1, 'flatten', Nx, Ny, nx, ny, r1, r2, lvalue);

Fully           = fullyConnectedLayer(10, 'Name', 'fully');
SoftMax         = softmaxLayer('Name', 'softmax_layer');

Classification2 = CustomClassificationLayer('classification_layer');
Classification  = classificationLayer('Name', 'classification_layer', 'ClassWeights','none', 'Classes', 'auto');

layers = [
    InputLayer
    %ResizeLayer
    CResize
    KernelLayer

    Prop1

    %Conv1Layer
    %Conv2Layer

    AbsLayer
    Nonlinear1Layer
    %Nonlinear2Layer

    %BtcNorm1
    %BtcNorm2

    Flatten
    %Fully
    SoftMax
    Classification];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',learningRate, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

lgraph = layerGraph();

for i=1:length(layers)
    lgraph = addLayers(lgraph, layers(i));
end

lgraph = connectLayers(lgraph, 'input_layer', 'resize_layer');
lgraph = connectLayers(lgraph, 'resize_layer', 'kernel_layer');
lgraph = connectLayers(lgraph, 'kernel_layer/out1', 'prop_layer/in1');
lgraph = connectLayers(lgraph, 'kernel_layer/out2', 'prop_layer/in2');

lgraph = connectLayers(lgraph, 'prop_layer/out1', 'absolute_layer/in1');
lgraph = connectLayers(lgraph, 'prop_layer/out2', 'absolute_layer/in2');

lgraph = connectLayers(lgraph, 'absolute_layer', 'nonlinear1_layer');
lgraph = connectLayers(lgraph, 'nonlinear1_layer', 'flatten');

% softmax
lgraph = connectLayers(lgraph, 'flatten', 'softmax_layer');

% Classify
lgraph = connectLayers(lgraph, 'softmax_layer', 'classification_layer');

plot(lgraph);

net = trainNetwork(imdsTrain,lgraph,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)