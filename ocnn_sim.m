% optical artifical neural network

disp("Getting data...");
Nx    = 256;
Ny    = 256;
ratio = 2;
ix    = round(Nx/ratio);
iy    = round(Ny/ratio);
nx    = 21e-3;
ny    = 21e-3;
d1    = 50e-2;
d2    = 15e-2;
wv    = 1550e-9;
a0    = 20;
r1    = nx/6;
r2    = nx/25;


digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% get size of images
img = readimage(imds, 1);
[dimx, dimy] = size(img);


numTrainFiles = 750;
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
ResizeLayer = functionLayer(@(X)resize_normalize_extend(X, Nx, Ny, k), 'Name', 'resize_layer');
RealKernelLayer = convolution2dLayer([Nx, Ny], 1, 'Padding', 'same', 'Bias', 0, 'Weights', real(kernel), 'WeightLearnRateFactor', 1, 'Name', 'real_kernel_layer');
ImagKernelLayer = convolution2dLayer([Nx, Ny], 1, 'Padding', 'same', 'Bias', 0, 'Weights', imag(kernel), 'WeightLearnRateFactor', 1, 'Name', 'imag_kernel_layer');

RealProp1LayerA = convolution2dLayer([Nx, Ny], 1, 'Padding','same', 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', real(w1), 'Name', 'real_prop_1_conv2d_layer_A');
ImagProp1LayerB = convolution2dLayer([Nx, Ny], 1, 'Padding','same', 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', imag(w1), 'Name', 'imag_prop_1_conv2d_layer_B');
RealProp1LayerC = convolution2dLayer([Nx, Ny], 1, 'Padding','same', 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', real(w1), 'Name', 'real_prop_1_conv2d_layer_C');
ImagProp1LayerD = convolution2dLayer([Nx, Ny], 1, 'Padding','same', 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', imag(w1), 'Name', 'imag_prop_1_conv2d_layer_D');

Negation1Layer  = functionLayer(@(X)(-1 * X), 'Name', 'negation_layer_1');
Addition1LayerA = additionLayer(2, 'Name', 'real_addition_layer_1');
Addition1LayerB = additionLayer(2, 'Name', 'imag_addition_layer_1');

Abs1Layer       = functionLayer(@(X, Y)abs(X + 1i * Y), 'Name', 'absolute_layer_1', 'NumInputs', 2, 'NumOutputs', 1);
Angle1Layer     = functionLayer(@(X, Y)angle(X + 1i * Y), 'Name', 'angle_layer_1', 'NumInputs', 2, 'NumOutputs', 1);

NonLinearLayer  = functionLayer(@(X)nonlinear_forward(X,a0), 'Name', 'nonlinear_layer');

RealLayer       = functionLayer(@(X, Y)real(X .* exp(1i * Y)), 'Name', 'real_layer', 'NumInputs', 2);
ImagLayer       = functionLayer(@(X, Y)imag(X .* exp(1i * Y)), 'Name', 'imag_layer', 'NumInputs', 2);

RealProp2LayerA = convolution2dLayer([Nx, Ny], 1, 'Padding', 'same', 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer', 'zeros', 'Weights', real(w2), 'Name', 'real_prop_2_conv2d_layer_A');
ImagProp2LayerB = convolution2dLayer([Nx, Ny], 1, 'Padding', 'same', 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer', 'zeros', 'Weights', imag(w2), 'Name', 'imag_prop_2_conv2d_layer_B');
RealProp2LayerC = convolution2dLayer([Nx, Ny], 1, 'Padding', 'same', 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer', 'zeros', 'Weights', real(w2), 'Name', 'real_prop_2_conv2d_layer_C');
ImagProp2LayerD = convolution2dLayer([Nx, Ny], 1, 'Padding', 'same', 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer', 'zeros', 'Weights', imag(w2), 'Name', 'imag_prop_2_conv2d_layer_D');

Negation2Layer  = functionLayer(@(X)(-1 * X), 'Name', 'negation_layer_2');
Addition2LayerA = additionLayer(2, 'Name', 'real_addition_layer_2');
Addition2LayerB = additionLayer(2, 'Name', 'imag_addition_layer_2');

ReductionLayer  = functionLayer(@(X, Y)reduction(X, Y, nx, ny, r1, r2), 'Name', 'reduction_layer', 'NumInputs', 2);
SoftMaxLayer    = softmaxLayer('Name', 'softmax_layer');
Classification  = classificationLayer('Name', 'classification_layer');

layers = [
    InputLayer
    ResizeLayer

    RealKernelLayer
    ImagKernelLayer
    
    RealProp1LayerA
    ImagProp1LayerB
    RealProp1LayerC
    ImagProp1LayerD

    Negation1Layer
    Addition1LayerA
    Addition1LayerB

    Abs1Layer
    Angle1Layer
    NonLinearLayer

    RealLayer
    ImagLayer
    
    RealProp2LayerA
    ImagProp2LayerB
    RealProp2LayerC
    ImagProp2LayerD

    Negation2Layer
    Addition2LayerA
    Addition2LayerB

    ReductionLayer
    SoftMaxLayer
    Classification];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
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
lgraph = connectLayers(lgraph, 'resize_layer', 'real_kernel_layer');
lgraph = connectLayers(lgraph, 'resize_layer', 'imag_kernel_layer');
lgraph = connectLayers(lgraph, 'real_kernel_layer', 'real_prop_1_conv2d_layer_A');
lgraph = connectLayers(lgraph, 'real_kernel_layer', 'imag_prop_1_conv2d_layer_B');
lgraph = connectLayers(lgraph, 'imag_kernel_layer', 'real_prop_1_conv2d_layer_C');
lgraph = connectLayers(lgraph, 'imag_kernel_layer', 'imag_prop_1_conv2d_layer_D');

% Real
lgraph = connectLayers(lgraph, 'imag_prop_1_conv2d_layer_D', 'negation_layer_1');
lgraph = connectLayers(lgraph, 'real_prop_1_conv2d_layer_A', 'real_addition_layer_1/in1');
lgraph = connectLayers(lgraph, 'negation_layer_1', 'real_addition_layer_1/in2');

% Imag
lgraph = connectLayers(lgraph, 'imag_prop_1_conv2d_layer_B', 'imag_addition_layer_1/in1');
lgraph = connectLayers(lgraph, 'real_prop_1_conv2d_layer_C', 'imag_addition_layer_1/in2');

% Magnitude
lgraph = connectLayers(lgraph, 'real_addition_layer_1', 'absolute_layer_1/in1');
lgraph = connectLayers(lgraph, 'imag_addition_layer_1', 'absolute_layer_1/in2');

% Phase
lgraph = connectLayers(lgraph, 'real_addition_layer_1', 'angle_layer_1/in1');
lgraph = connectLayers(lgraph, 'imag_addition_layer_1', 'angle_layer_1/in2');

% Nonlinear layer
lgraph = connectLayers(lgraph, 'absolute_layer_1', 'nonlinear_layer');

% decompose to real and imaginary
lgraph = connectLayers(lgraph, 'nonlinear_layer', 'real_layer/in1');
lgraph = connectLayers(lgraph, 'angle_layer_1', 'real_layer/in2');

lgraph = connectLayers(lgraph, 'nonlinear_layer', 'imag_layer/in1');
lgraph = connectLayers(lgraph, 'angle_layer_1', 'imag_layer/in2');

lgraph = connectLayers(lgraph, 'real_layer', 'real_prop_2_conv2d_layer_A');
lgraph = connectLayers(lgraph, 'real_layer', 'imag_prop_2_conv2d_layer_B');
lgraph = connectLayers(lgraph, 'imag_layer', 'real_prop_2_conv2d_layer_C');
lgraph = connectLayers(lgraph, 'imag_layer', 'imag_prop_2_conv2d_layer_D');
lgraph = connectLayers(lgraph, 'imag_prop_2_conv2d_layer_D', 'negation_layer_2');

% Real
lgraph = connectLayers(lgraph, 'real_prop_2_conv2d_layer_A', 'real_addition_layer_2/in1');
lgraph = connectLayers(lgraph, 'negation_layer_2', 'real_addition_layer_2/in2');

% Imaginary
lgraph = connectLayers(lgraph, 'imag_prop_2_conv2d_layer_B', 'imag_addition_layer_2/in1');
lgraph = connectLayers(lgraph, 'real_prop_2_conv2d_layer_C', 'imag_addition_layer_2/in2');

% Reduce
lgraph = connectLayers(lgraph, 'real_addition_layer_2', 'reduction_layer/in1');
lgraph = connectLayers(lgraph, 'imag_addition_layer_2', 'reduction_layer/in2');

lgraph = connectLayers(lgraph, 'reduction_layer', 'softmax_layer');
lgraph = connectLayers(lgraph, 'softmax_layer', 'classification_layer');

%plot(lgraph);

net = trainNetwork(imdsTrain,lgraph,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

