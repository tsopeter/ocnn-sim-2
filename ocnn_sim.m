% optical artifical neural network

disp("Getting data...");
Nx     = 256;
Ny     = 256;
ratio  = 2;
ix     = round(Nx/ratio);
iy     = round(Ny/ratio);
nx     = 21e-3;
ny     = 21e-3;
d1     = 15e-2;
d2     = 15e-2;
wv     = 1550e-9;
a0     = 20;
r1     = nx/4;
r2     = nx/20;
rate   = 1;
learningRate = 1e-5;
numTrainFiles = 950;
lvalue = 1e-9;
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

w1 = fftshift(get_propagation_distance(ix, iy, nx/ratio, ny/ratio, d1, wv));
w2 = fftshift(get_propagation_distance(ix, iy, nx/ratio, ny/ratio, d2, wv));

kernel = internal_random_amp(Nx, Ny);

InputLayer  = imageInputLayer([dimx, dimy, 1], 'Name', 'input_layer', 'Normalization', 'rescale-zero-one');
ResizeLayer = CustomResizeLayer(1, 'resize_layer', Nx, Ny, k, lvalue);
KernelLayer = CustomKernelLayer(1, 'kernel_layer', kernel, rate);

RealProp1LayerA = convolution2dLayer([ix, iy], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', real(w1), 'Name', 'real_prop_1_conv2d_layer_A');
ImagProp1LayerB = convolution2dLayer([ix, iy], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', imag(w1), 'Name', 'imag_prop_1_conv2d_layer_B');
RealProp1LayerC = convolution2dLayer([ix, iy], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', real(w1), 'Name', 'real_prop_1_conv2d_layer_C');
ImagProp1LayerD = convolution2dLayer([ix, iy], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', imag(w1), 'Name', 'imag_prop_1_conv2d_layer_D');

Negation1Layer  = CustomNegationLayer(1, 'negation_layer_1');
Addition1LayerA = additionLayer(2, 'Name', 'real_addition_layer_1');
Addition1LayerB = additionLayer(2, 'Name', 'imag_addition_layer_1');

NormalizationLayer1A = batchNormalizationLayer('Name', 'batch_norm_1A');
NormalizationLayer1B = batchNormalizationLayer('Name', 'batch_norm_1B');

NonLinearLayer  = CustomNonlinearLayer(2, 'nonlinear_layer', lvalue);

AbsoluteLayer   = CustomAbsoluteLayer(2, 'absolute_layer', lvalue);
NonLinearLayer2 = CustomReLULayer(2, 'nonlinear_layer', lvalue, sx, sy, sc, sz);

RealProp2LayerA = convolution2dLayer([ix, iy], 1, 'Padding', 'same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer', 'zeros', 'Weights', real(w2), 'Name', 'real_prop_2_conv2d_layer_A');
ImagProp2LayerB = convolution2dLayer([ix, iy], 1, 'Padding', 'same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer', 'zeros', 'Weights', imag(w2), 'Name', 'imag_prop_2_conv2d_layer_B');
RealProp2LayerC = convolution2dLayer([ix, iy], 1, 'Padding', 'same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer', 'zeros', 'Weights', real(w2), 'Name', 'real_prop_2_conv2d_layer_C');
ImagProp2LayerD = convolution2dLayer([ix, iy], 1, 'Padding', 'same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer', 'zeros', 'Weights', imag(w2), 'Name', 'imag_prop_2_conv2d_layer_D');

Negation2Layer  = CustomNegationLayer(1, 'negation_layer_2');
Addition2LayerA = additionLayer(2, 'Name', 'real_addition_layer_2');
Addition2LayerB = additionLayer(2, 'Name', 'imag_addition_layer_2');

NormalizationLayer2A = batchNormalizationLayer('Name', 'batch_norm_2A');
NormalizationLayer2B = batchNormalizationLayer('Name', 'batch_norm_2B');

MaskLayer       = CustomMaskLayer(2, 'mask_layer', Nx, Ny, nx, ny, r1, r2, lvalue);
Flatten         = CustomFlattenLayer(1, 'flatten', Nx, Ny, nx, ny, r1, r2, lvalue);
SoftMax         = softmaxLayer('Name', 'softmax_layer');

Classification2 = CustomClassificationLayer('classification_layer');
Classification  = classificationLayer('Name', 'classification_layer', 'ClassWeights','none', 'Classes', 'auto');

layers = [
    InputLayer
    ResizeLayer

    KernelLayer
    
    RealProp1LayerA
    ImagProp1LayerB
    RealProp1LayerC
    ImagProp1LayerD

    Negation1Layer
    Addition1LayerA
    Addition1LayerB

    %NormalizationLayer1A
    %NormalizationLayer1B
    %NonLinearLayer
    %AbsoluteLayer
    NonLinearLayer2

    RealProp2LayerA
    ImagProp2LayerB
    RealProp2LayerC
    ImagProp2LayerD

    Negation2Layer
    Addition2LayerA
    Addition2LayerB

    %NormalizationLayer2A
    %NormalizationLayer2B
    %MaskLayer
    AbsoluteLayer
    Flatten
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
lgraph = connectLayers(lgraph, 'kernel_layer/out1', 'real_prop_1_conv2d_layer_A');
lgraph = connectLayers(lgraph, 'kernel_layer/out1', 'imag_prop_1_conv2d_layer_B');
lgraph = connectLayers(lgraph, 'kernel_layer/out2', 'real_prop_1_conv2d_layer_C');
lgraph = connectLayers(lgraph, 'kernel_layer/out2', 'imag_prop_1_conv2d_layer_D');

% Real
lgraph = connectLayers(lgraph, 'imag_prop_1_conv2d_layer_D', 'negation_layer_1');
lgraph = connectLayers(lgraph, 'real_prop_1_conv2d_layer_A', 'real_addition_layer_1/in1');
lgraph = connectLayers(lgraph, 'negation_layer_1', 'real_addition_layer_1/in2');

% Imag
lgraph = connectLayers(lgraph, 'imag_prop_1_conv2d_layer_B', 'imag_addition_layer_1/in1');
lgraph = connectLayers(lgraph, 'real_prop_1_conv2d_layer_C', 'imag_addition_layer_1/in2');

%lgraph = connectLayers(lgraph, 'real_addition_layer_1', 'batch_norm_1A');
%lgraph = connectLayers(lgraph, 'imag_addition_layer_1', 'batch_norm_1B');

%lgraph = connectLayers(lgraph, 'real_addition_layer_1', 'absolute_layer/in1');
%lgraph = connectLayers(lgraph, 'imag_addition_layer_1', 'absolute_layer/in2');
%lgraph = connectLayers(lgraph, 'absolute_layer', 'nonlinear_layer');

lgraph = connectLayers(lgraph, 'real_addition_layer_1', 'nonlinear_layer/in1');
lgraph = connectLayers(lgraph, 'imag_addition_layer_1', 'nonlinear_layer/in2');

% Nonlinear layer
%lgraph = connectLayers(lgraph, 'batch_norm_1A', 'nonlinear_layer/in1');
%lgraph = connectLayers(lgraph, 'batch_norm_1B', 'nonlinear_layer/in2');

lgraph = connectLayers(lgraph, 'nonlinear_layer/out1', 'real_prop_2_conv2d_layer_A');
lgraph = connectLayers(lgraph, 'nonlinear_layer/out1', 'imag_prop_2_conv2d_layer_B');
lgraph = connectLayers(lgraph, 'nonlinear_layer/out2', 'real_prop_2_conv2d_layer_C');
lgraph = connectLayers(lgraph, 'nonlinear_layer/out2', 'imag_prop_2_conv2d_layer_D');
lgraph = connectLayers(lgraph, 'imag_prop_2_conv2d_layer_D', 'negation_layer_2');

% Real
lgraph = connectLayers(lgraph, 'real_prop_2_conv2d_layer_A', 'real_addition_layer_2/in1');
lgraph = connectLayers(lgraph, 'negation_layer_2', 'real_addition_layer_2/in2');

% Imaginary
lgraph = connectLayers(lgraph, 'imag_prop_2_conv2d_layer_B', 'imag_addition_layer_2/in1');
lgraph = connectLayers(lgraph, 'real_prop_2_conv2d_layer_C', 'imag_addition_layer_2/in2');

%lgraph = connectLayers(lgraph, 'real_addition_layer_2', 'batch_norm_2A');
%lgraph = connectLayers(lgraph, 'imag_addition_layer_2', 'batch_norm_2B');

% Mask
%lgraph = connectLayers(lgraph, 'batch_norm_2A', 'mask_layer/in1');
%lgraph = connectLayers(lgraph, 'batch_norm_2B', 'mask_layer/in2');

%lgraph = connectLayers(lgraph, 'real_addition_layer_2', 'mask_layer/in1');
%lgraph = connectLayers(lgraph, 'imag_addition_layer_2', 'mask_layer/in2');

lgraph = connectLayers(lgraph, 'real_addition_layer_2', 'absolute_layer/in1');
lgraph = connectLayers(lgraph, 'imag_addition_layer_2', 'absolute_layer/in2');

% flatten
lgraph = connectLayers(lgraph, 'absolute_layer', 'flatten');

% softmax
lgraph = connectLayers(lgraph, 'flatten', 'softmax_layer');

% Classify
lgraph = connectLayers(lgraph, 'softmax_layer', 'classification_layer');

plot(lgraph);

net = trainNetwork(imdsTrain,lgraph,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

