function z = resize_normalize_extend(X, Nx, Ny, k, lvalue)
    W = size(X);
    if length(W)<=2
        W(3)=1;
        W(4)=1;
    end
    M = single(zeros(Nx, Ny, W(3), W(4), 'single'));
    for i=1:W(4)
        Q = mask_resize(interp2(X(:,:,1,i), k), Nx, Ny);
        Q(Q==0)=lvalue;
        M(:,:,1,i) = Q;
    end
    z = M;
end