function Z = reduction(X, Y, nx, ny, r1, r2)   % reduction layer reduces to 10 circles and presents values
    W = size(X);
    if length(W) == 2
        dim = size(X);
    
        Q = X + 1i * Y;
    
        Z = dlarray(zeros(1, 1, 10, 1, 'single'));
        for r=0:1:9
            plate = circle_at(dim(1), dim(2), nx, ny, r1, 0, r2);
            plate = imrotate(plate, 36*r, 'crop');
    
            Z(1,1,r+1,1)=single(sum(sum(abs(plate .* Q))));
        end
    else
        dim = size(X);
        Z = dlarray(zeros(1, 1, 10, W(4), 'single'));

        Q = X + 1i * Y;

        for i=1:1:W(4)
            for r=0:1:9
                plate = circle_at(dim(1), dim(2), nx, ny, r1, 0, r2);
                plate = imrotate(plate, 36*r, 'crop');

                Z(1, 1, r+1, i)=single(sum(sum(abs(plate .* Q(:, :, 1, i)))));
            end
        end
    end
end