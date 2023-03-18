function z = dlfft2(x, y)
    z = fft(fft(x).').' .* y;
    z = abs(ifft(ifft(z).').');
end