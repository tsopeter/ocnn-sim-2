function output = internal_random_amp(Nx, Ny)
    output = arrayfun(@(t)exp(1i * randn()), zeros(Nx, Ny));
end