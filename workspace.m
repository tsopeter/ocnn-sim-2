% workspace
clear;
parameters;

Rx = Nx/ratio;
Ry = Ny/ratio;
rx = nx/ratio;
ry = ny/ratio;

w1 = fftshift(get_propagation_distance(Rx, Ry, rx, ry, d1, wv));

% create resize layer
resize = CustomResizeLayer(1, 'no_name', Nx, Ny, k, lvalue, P);

% create the propgation layer
prop1 = CustomLearnablePropagationLayer('no_name', Nx, Ny, Rx, Ry, nx, ny, d1, wv);

% create the nonlinear layer
nonlinear = CustomReLULayer(2, 'no_name', lvalue, sx, sy, sc, sz);

% create the absolute layer
abslayer = CustomAbsoluteLayer(2, 'no_name', lvalue);

% flatten
res_size = prop1.res_size;
Flatten  = CustomXORFlattenLayer(1, 'no_name', res_size(1), res_size(2), rx, ry, r1, r2, lvalue);

% softmax
SoftMax  = softmaxLayer("Name", 'no_name');

img = imread(cell2mat(dataTrain.Files(1)));
x = single(img ./ 255);
x = resize.predict(single(img));
[pry, piy] = prop1.predict(x, zeros(size(x)));
[prz, piz] = nonlinear.predict(pry, piy);
prp        = abslayer.predict(prz, piz);
qrp        = Flatten.predict(prp);
srp        = softmax(qrp, DataFormat='SSCB')

y = pry + 1i * piy;
z = prz + 1i * piz;

figure;
imagesc(Flatten.plate);

figure;
imagesc(x);
colorbar();
figure;
imagesc(extractdata(abs(y)));
colorbar();
figure;
imagesc(extractdata(abs(z)));
colorbar();
figure;
imagesc(extractdata(prp));
colorbar();
figure;
v1 = extractdata(abs(prp .* imrotate(Flatten.plates(:,:,1), 90, 'crop')));
imagesc(v1);
colorbar();
v1s = sum(v1, 'all');

figure;
v2 = extractdata(abs(prp .* imrotate(Flatten.plates(:,:,1), 270, 'crop')));
imagesc(v2);
colorbar();
v2s = sum(v2, 'all');

% % network test properties
% Nx = 64;
% Ny = 64;
% ss = [Nx Ny];
% rx = 32;
% ry = 32;
% d  = 1;
% wx = randn(rx,ry)+1i*randn(rx,ry);
% wy = randn(ss)+1i*randn(ss);
% 
% %
% checkLayer(CustomPropagationLayer('no_name', 32, 32, 1, wx), {ss, ss});
% 
% %
% % checkLayer(CustomFFTPropagationLayer('no_name', 32, 32, 1, wy), {ss, ss});