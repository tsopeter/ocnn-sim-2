function z = fft_conv2(x, y)
    z = ifft2(fft2(x) .* y);
end