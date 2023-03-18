function z = dlfft2(x, y)
    z = fft(fft(x).').' .* y;
    z = real(ifft(ifft(z).').');
end