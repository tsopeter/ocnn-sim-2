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
 
d1 = get_propagation_distance(Nx, Ny, nx, ny, distance_1, wavelength);
d2 = get_propagation_distance(Nx, Ny, nx, ny, distance_2, wavelength);

% restore the path
addpath(oldpath);