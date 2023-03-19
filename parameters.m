%
% parameters

Nx     = 128;
Ny     = 128;
ratio  = 2;
ix     = round(Nx/ratio);
iy     = round(Ny/ratio);
nx     = 21e-3;
ny     = 21e-3;
d1     = 100e-2;
d2     = 25e-2;
wv     = 1550e-9;
a0     = 20;
r1     = nx/7;
r2     = nx/40;
rate   = 1;
learningRate = 0.009;
numTrainFiles = 250;
lvalue = 0;
sx     = 2;
sy     = 1;
sc     = 0.1;
sz     = 0.2;
P      = 1;

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

w1 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d1, wv));
w2 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d2, wv));
