function Z = reduction(X, Y, nx, ny, r1, r2)   % reduction layer reduces to 10 circles and presents values
    dim = size(X);

    Q = X + 1i * Y;

    Z = dlarray(zeros(1, 1, 10, 1, 'single'));
    for r=0:1:9
        plate = circle_at(dim(1), dim(2), nx, ny, r1, 0, r2);
        plate = imrotate(plate, 36*r, 'crop');

        Z(1,1,r+1,1)=single(sum(sum(abs(plate .* Q))));
    end
end