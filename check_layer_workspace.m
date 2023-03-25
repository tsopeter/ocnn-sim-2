% check layers
parameters;

Nx = 64;
Ny = 64;

Rx = Nx/ratio;
Ry = Ny/ratio;
rx = nx/ratio;
ry = ny/ratio;

w1 = fftshift(get_propagation_distance(Rx, Ry, rx, ry, d1, wv));

checkLayer(CustomPropagationLayer('no_name', Nx, Ny, d1, w1), {[Nx, Ny], [Nx, Ny]});