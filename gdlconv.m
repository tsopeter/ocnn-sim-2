function dLdX = gdlconv(f, dLdZ, r)
    dQ = cast(dLdZ, 'like',1);
    df = cast(f, 'like',1);
    w = size(dLdZ, 4);
    dLdX = zeros([r, 1, w], 'like', dLdZ);
    for i=1:w
        dLdX(:,:,1,i)=conv2(df, dQ(:,:,1,i), 'full');
    end
end