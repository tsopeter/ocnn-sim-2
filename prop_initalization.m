function weights = prop_initalization(sz, nx, ny, distance, wavelength)
    weights = get_propagation_distance(sz(1), sz(2), nx, ny, distance, wavelength);
end