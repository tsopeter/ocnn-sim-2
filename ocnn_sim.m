% optical artifical neural network
% 
%

% just as a reminder
% colormap('hot');
% imagesc(abs(kernel));
% colorbar('hot');

% define the parameters of the network

        Nx = 1024;      % number of columns
        Ny = 1024;      % number of rows

        inputSize  = [Ny, Nx, 1];
        numClasses = 10;    % one for each digit 0-9
        
        % this defines the size of the display
        nx = 40e-3;
        ny = 40e-3;

        % filter_1 ratio
        ratio=4;
        
        % interpolation value
        ix = Nx/ratio;
        iy = Ny/ratio;
        
        a0 = 20;
        
        wavelength = 1000e-9;    % wavelength
        
        epoch = 120;              % we want 100 epochs
        images_per_epoch = 250; % we want 500 images per training session (epoch)
        
        distance_1 = 30e-2;      % propagation distance
        distance_2 = 15e-2;
        
        eta = 12.0;              % learning rate

        testing_ratio = 0.1;     % 10% of testing data (10k images)

        P = 1;

        r1 = nx/7;
        r2 = nx/50;

% create a plate to detect digits
plate = detector_plate(Nx, Ny, nx, ny, r1, r2);

disp("Getting data...");

oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
dataimagefile = '~/Documents/Github/ocnn-sim-2/Images/train-images-idx3-ubyte.gz';
datalabelfile = '~/Documents/Github/ocnn-sim-2/Images/train-labels-idx1-ubyte.gz';
testimagefile = '~/Documents/Github/ocnn-sim-2/Images/t10k-images-idx3-ubyte.gz';
testlabelfile = '~/Documents/Github/ocnn-sim-2/Images/t10k-labels-idx1-ubyte.gz';

XTrain = processImagesMNIST(dataimagefile);
YTrain = processLabelsMNIST(datalabelfile);
XTest = processImagesMNIST(testimagefile);
YTest = processLabelsMNIST(testlabelfile);

% restore the path
addpath(oldpath);

cols = length(XTrain(:,:,1));
rows = length(XTrain(:,:,1).');

% get the interpolation value k
kx = log2(double(ix - cols)/double(cols - 1))+1;
ky = log2(double(iy - rows)/double(rows - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);

d1   = get_propagation_distance(round(ix), round(iy), nx/ratio, ny/ratio, distance_1, wavelength);

disp("Building network...");

I1D1Layer = imageInputLayer(inputSize, 'Name', 'input_layer'); 

I1D2Layer = imageInputLayer(inputSize, 'Name', 'kernel_layer_real');
I1D3Layer = imageInputLayer(inputSize, 'Name', 'kernel_layer_imag');

M1D1Layer = multiplicationLayer(2, 'Name', 'multiplication_layer_real');
M1D2Layer = multiplicationLayer(2, 'Name', 'multiplication_layer_imag');

C1D1Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_1_real_real');
C1D2Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_1_imag_imag');
C1D3Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_1_real_imag');
C1D4Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_1_imag_real');

M2D1Layer = multiplicationLayer(2, 'Name', 'multiplication_layer_1_negation');
M2D2Layer = multiplicationLayer(2, 'Name', 'multiplication_layer_1_imag');

A1D1Layer = additionLayer(2, 'Name', 'addition_layer_1_real');
A1D2Layer = additionLayer(2, 'Name', 'addition_layer_1_imag');
A1D3Layer = additionLayer(2, 'Name', 'addition_layer_1_comb');

N1D1Layer = functionLayer(@(x)sa_forward(x), 'Name', 'nonlinear_layer');

C2D1Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_2_real_real');
C2D2Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_2_imag_imag');
C2D3Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_2_real_imag');
C2D4Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_2_imag_real');

M3D1Layer = multiplicationLayer(2, 'Name', 'multiplication_layer_2_negation');
M3D2Layer = multiplicationLayer(2, 'Name', 'multiplication_layer_2_imag');

A2D1Layer = additionLayer(2, 'Name', 'addition_layer_2_real');
A2D2Layer = additionLayer(2, 'Name', 'addition_layer_2_imag');
A2D3Layer = additionLayer(2, 'Name', 'addition_layer_2_comb');

R1D1Layer = multiplicationLayer(2, 'Name', 'plate_layer');
R1D2Layer = imageInputLayer(inputSize, 'Name', 'plate_input');

lgraph = layerGraph();
layers = [
   I1D1Layer
   I1D2Layer
   I1D3Layer

   M1D1Layer
   M1D2Layer

   C1D1Layer
   C1D2Layer
   C1D3Layer
   C1D4Layer

   M2D1Layer
   M2D2Layer

   A1D1Layer
   A1D2Layer
   A1D3Layer

   N1D1Layer

   C2D1Layer
   C2D2Layer
   C2D3Layer
   C2D4Layer

   M3D1Layer
   M3D2Layer

   A2D1Layer
   A2D2Layer
   A2D3Layer

   R1D1Layer
   R1D2Layer
];

lgraph = addLayers(lgraph, layers);

plot(lgraph);


