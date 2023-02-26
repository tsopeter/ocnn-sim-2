function z = resize_normalize_extend(X, Nx, Ny, k)
    W = size(X);
    Q = extractdata(X);
    if length(size(X)) == 2
        M = mask_resize(interp2(Q, k, 'linear'), Nx, Ny);
    else
        M = zeros(Nx, Ny, W(3), W(4), 'single');
        for i=1:W(4)
            M(:,:,1,i) = mask_resize(interp2(Q(:,:,1,i), k, 'linear'), Nx, Ny);
        end
    end
    z = dlarray(M);
end