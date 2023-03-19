% workspace
clear;
parameters;
% create resize layer
resize = CustomResizeLayer(1, 'no_name', Nx, Ny, k, lvalue);

% create the propgation layer
prop1 = CustomFFTPropagationLayer('no_name', Nx, Ny, d1, w1);
prop2 = CustomFFTPropagationLayer('no_name', Nx, Ny, d2, w2);

% create the nonlinear layer
nonlinear = CustomReLULayer(2, 'no_name', lvalue, sx, sy, sc, sz);

% create the absolute layer
abslayer = CustomAbsoluteLayer(2, 'no_name', lvalue);

img = readimage(imds, 3001);
x = resize.predict(single(img));
[pry, piy] = prop1.predict(x, zeros(size(x)));
[prz, piz] = nonlinear.predict(pry, piy);
[prw, piw] = prop2.predict(prz, piz);
[pru, piu] = nonlinear.predict(prw, piw);
prp        = abslayer.predict(pru, piu);

y = pry + 1i * piy;
z = prz + 1i * piz;
w = prw + 1i * piw;
u = pru + 1i * piu;

figure;
imagesc(x);
colorbar();
figure;
imagesc(abs(y));
colorbar();
figure;
imagesc(abs(z));
colorbar();
figure;
imagesc(abs(w));
colorbar();
figure;
imagesc(abs(u));
colorbar();
figure;
imagesc(prp);
colorbar();

% network test properties
%Nx = 64;
%Ny = 64;
%ss = [Nx Ny];
%rx = 32;
%ry = 32;
%d  = 1;
%wx = randn(rx,ry)+1i*randn(rx,ry);
%wy = randn(ss)+1i*randn(ss);

%
% checkLayer(CustomPropagationLayer('no_name', 32, 32, 1, wx), {ss, ss});
%
% checkLayer(CustomFFTPropagationLayer('no_name', 32, 32, 1, wy), {ss, ss});