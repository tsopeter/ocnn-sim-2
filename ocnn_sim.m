% optical artifical neural network

disp("Getting data...");
parameters;

Rx1 = Nx/ratio;
Ry1 = Ny/ratio;
rx1 = nx/ratio;
ry1 = ny/ratio;

w1 = fftshift(get_propagation_distance(Rx1, Ry1, rx1, ry1, d1, wv));
w2 = fftshift(get_propagation_distance(Rx1, Ry1, rx1, ry1, d2, wv));

InputLayer  = imageInputLayer([dimx, dimy, 1], 'Name', 'input_layer', 'Normalization', 'none');
ResizeLayer = CustomResizeLayer(1, 'resize_layer', Nx, Ny, k, lvalue, P);
KernelLayer = CustomKernelLayer(1, 'kernel_layer', kernel, rate);
MidLayer    = CustomKernelLayer(1, 'mid_layer', kernel, rate);

RealProp1LayerA = convolution2dLayer([Rx1, Ry1], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', real(w1), 'Name', 'real_prop_1_conv2d_layer_A');
ImagProp1LayerB = convolution2dLayer([Rx1, Ry1], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', imag(w1), 'Name', 'imag_prop_1_conv2d_layer_B');
RealProp1LayerC = convolution2dLayer([Rx1, Ry1], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', real(w1), 'Name', 'real_prop_1_conv2d_layer_C');
ImagProp1LayerD = convolution2dLayer([Rx1, Ry1], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', imag(w1), 'Name', 'imag_prop_1_conv2d_layer_D');

RealProp2LayerA = convolution2dLayer([Rx1, Ry1], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', real(w2), 'Name', 'real_prop_2_conv2d_layer_A');
ImagProp2LayerB = convolution2dLayer([Rx1, Ry1], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', imag(w2), 'Name', 'imag_prop_2_conv2d_layer_B');
RealProp2LayerC = convolution2dLayer([Rx1, Ry1], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', real(w2), 'Name', 'real_prop_2_conv2d_layer_C');
ImagProp2LayerD = convolution2dLayer([Rx1, Ry1], 1, 'Padding','same', 'PaddingValue', lvalue, 'Bias', 0, 'WeightLearnRateFactor', 0, 'BiasInitializer','zeros', 'Weights', imag(w2), 'Name', 'imag_prop_2_conv2d_layer_D');

Negation2Layer  = CustomNegationLayer(1, 'negation_layer_2');
Addition2LayerA = additionLayer(2, 'Name', 'real_addition_layer_2');
Addition2LayerB = additionLayer(2, 'Name', 'imag_addition_layer_2');

Negation1Layer  = CustomNegationLayer(1, 'negation_layer_1');
Addition1LayerA = additionLayer(2, 'Name', 'real_addition_layer_1');
Addition1LayerB = additionLayer(2, 'Name', 'imag_addition_layer_1');

Nonlinear1       = CustomReLULayer(2, 'nonlinear1_layer', lvalue, sx, sy, sc, sz);
Nonlinear2       = CustomReLULayer(2, 'nonlinear2_layer', lvalue, sx, sy, sc, sz);

QLayer          = CustomAbsoluteLayer(2, 'q_layer', lvalue);
AbsLayer        = CustomAbsoluteLayer(2, 'absolute_layer', lvalue);

Flatten         = CustomFlattenLayer(1, 'flatten', Nx, Ny, nx, ny, r1, r2, lvalue);
SoftMax         = softmaxLayer('Name', 'softmax_layer');

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

    Nonlinear1
    QLayer
    MidLayer

    RealProp2LayerA
    ImagProp2LayerB
    RealProp2LayerC
    ImagProp2LayerD

    Negation2Layer
    Addition2LayerA
    Addition2LayerB

    Nonlinear2

    AbsLayer
    Flatten
    SoftMax
    Classification];

options = trainingOptions('adam', ...
    InitialLearnRate=learnRate,...
    CheckpointFrequency=10, ...
    CheckpointFrequencyUnit='epoch',...
    CheckpointPath='CheckPoints/',...
    Epsilon=1e-7,...
    MaxEpochs=numEpochs,...
    Shuffle='every-epoch',...
    ValidationData=dataTest,...
    ValidationFrequency=512,...
    Verbose=true,...
    Plots='training-progress',...
    ExecutionEnvironment='gpu',...
    DispatchInBackground=true,...
    MiniBatchSize=miniBatchSize);

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

% nonlinear
lgraph = connectLayers(lgraph, 'real_addition_layer_1', 'nonlinear1_layer/in1');
lgraph = connectLayers(lgraph, 'imag_addition_layer_1', 'nonlinear1_layer/in2');

lgraph = connectLayers(lgraph, 'nonlinear1_layer/out1', 'q_layer/in1');
lgraph = connectLayers(lgraph, 'nonlinear1_layer/out2', 'q_layer/in2');
lgraph = connectLayers(lgraph, 'q_layer', 'mid_layer');

lgraph = connectLayers(lgraph, 'mid_layer/out1', 'real_prop_2_conv2d_layer_A');
lgraph = connectLayers(lgraph, 'mid_layer/out1', 'imag_prop_2_conv2d_layer_B');
lgraph = connectLayers(lgraph, 'mid_layer/out2', 'real_prop_2_conv2d_layer_C');
lgraph = connectLayers(lgraph, 'mid_layer/out2', 'imag_prop_2_conv2d_layer_D');

% Real
lgraph = connectLayers(lgraph, 'imag_prop_2_conv2d_layer_D', 'negation_layer_2');
lgraph = connectLayers(lgraph, 'real_prop_2_conv2d_layer_A', 'real_addition_layer_2/in1');
lgraph = connectLayers(lgraph, 'negation_layer_2', 'real_addition_layer_2/in2');

% Imag
lgraph = connectLayers(lgraph, 'imag_prop_2_conv2d_layer_B', 'imag_addition_layer_2/in1');
lgraph = connectLayers(lgraph, 'real_prop_2_conv2d_layer_C', 'imag_addition_layer_2/in2');

% nonlinear
lgraph = connectLayers(lgraph, 'real_addition_layer_2', 'nonlinear2_layer/in1');
lgraph = connectLayers(lgraph, 'imag_addition_layer_2', 'nonlinear2_layer/in2');

lgraph = connectLayers(lgraph, 'nonlinear2_layer/out1', 'absolute_layer/in1');
lgraph = connectLayers(lgraph, 'nonlinear2_layer/out2', 'absolute_layer/in2');

% flatten
lgraph = connectLayers(lgraph, 'absolute_layer', 'flatten');

% softmax
lgraph = connectLayers(lgraph, 'flatten', 'softmax_layer');

% Classify
lgraph = connectLayers(lgraph, 'softmax_layer', 'classification_layer');

plot(lgraph);

net = trainNetwork(dataTrain,lgraph,options);

YPred = classify(net,dataTest);
YValidation = dataTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

