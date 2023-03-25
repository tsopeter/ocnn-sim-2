% CNN

clear;
parameters;

Rx = Nx/ratio;
Ry = Ny/ratio;
rx = nx/ratio;
ry = ny/ratio;

%
 w1 = fftshift(get_propagation_distance(Rx, Ry, rx, ry, d1, wv));
%  w1 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d1, wv));

InputLayer     = imageInputLayer([dimx, dimy, 1], 'Name', 'input_layer', 'Normalization', 'none');
ResizeLayer    = CustomResizeLayer(1, 'resize_layer', Nx, Ny, k, lvalue, P);
KernelLayer    = CustomKernelLayer(1, 'kernel_layer', kernel, rate);

Prop           = CustomPropagationLayer('prop_layer', Nx, Ny, d1, w1);

Nonlinear      = CustomReLULayer(2, 'nonlinear_layer', lvalue, sx, sy, sc, sz);

AbsLayer       = CustomAbsoluteLayer(2, 'absolute', lvalue);

res_size       = Prop.res_size;
Flatten        = CustomFlattenLayer(1, 'flatten', res_size(1), res_size(2), rx, ry, r1, r2, lvalue);
SoftMax        = softmaxLayer('Name', 'softmax');
Sigmoid        = sigmoidLayer("Name", 'sigmoid');
Classification = classificationLayer('Name', 'classification');
%Classification = CustomClassificationLayer('classification', 0.5);

%
pixClass = pixelClassificationLayer('Name', 'classification');

layers = [
    InputLayer
    ResizeLayer
    KernelLayer
    Prop
    Nonlinear
    AbsLayer
    Flatten
    SoftMax
    %Sigmoid
    Classification
    %pixClass
];

lgraph = layerGraph();
for i=1:length(layers)
    lgraph = addLayers(lgraph, layers(i));
end

lgraph = connectLayers(lgraph, 'input_layer', 'resize_layer');
lgraph = connectLayers(lgraph, 'resize_layer', 'kernel_layer');
%lgraph = connectLayers(lgraph, 'input_layer', 'kernel_layer');
lgraph = connectLayers(lgraph, 'kernel_layer/out1', 'prop_layer/in1');
lgraph = connectLayers(lgraph, 'kernel_layer/out2', 'prop_layer/in2');

lgraph = connectLayers(lgraph, 'prop_layer/out1', 'nonlinear_layer/in1');
lgraph = connectLayers(lgraph, 'prop_layer/out2', 'nonlinear_layer/in2');

lgraph = connectLayers(lgraph, 'nonlinear_layer/out1', 'absolute/in1');
lgraph = connectLayers(lgraph, 'nonlinear_layer/out2', 'absolute/in2');
lgraph = connectLayers(lgraph, 'absolute', 'flatten');

%softmax
lgraph = connectLayers(lgraph, 'flatten', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classification');

%lgraph = connectLayers(lgraph, 'absolute', 'classification');

plot(lgraph);

options = trainingOptions('adam', ...
    InitialLearnRate=learnRate,...
    MaxEpochs=numEpochs,...
    Shuffle='every-epoch',...
    ValidationData=dataTest,...
    ValidationFrequency=512,...
    Verbose=true,...
    Plots='training-progress',...
    ExecutionEnvironment='auto',...
    MiniBatchSize=miniBatchSize);

net = trainNetwork(dataTrain,lgraph,options);

YPred = classify(net,dataTest);
YValidation = dataTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
