function z = resize_normalize_extend(X, Nx, Ny, k)
    W = size(X);
    if length(size(X)) == 2
        M = mask_resize(interp2(X, k, 'linear'), Nx, Ny);
    else
        M = gpuArray(zeros(Nx, Ny, W(3), W(4), 'single'));
        for i=1:W(4)
            M(:,:,1,i) = mask_resize(interp2(X(:,:,1,i), k, 'linear'), Nx, Ny);
        end
    end
    z = M;
end