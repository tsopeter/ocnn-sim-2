% create xor images
function lbls = create_xor_images(location, number, Nx, Ny, nx, ny, r1, r2)
    extn='.png';
    lbls = zeros(number, 1, 'single');

    cent = circle_at(Nx, Ny, nx, ny, 0, 0, r2, 0);
    for i=1:number
        %
        % set a circle in the center
        vis = cent;
        
        %
        % elsewhere place circles randomly
        q = randi([1 4], 1);
        L   = randi([120 240], 1, 1);
        R   = randi([-60 60], 1, 1);
        if q==1
            % zero elements --> 1
            lbls(i) = 0;
        elseif q==2
            % one element, left side
            c   = circle_at(Nx, Ny, nx, ny, r1, 0, r2, 0);
            c   = imrotate(c, L, 'crop');
            vis = vis + c;
            lbls(i) = 1;
        elseif q==3
            % one element, right side
            c   = circle_at(Nx, Ny, nx, ny, r1, 0, r2, 0);
            c   = imrotate(c, R, 'crop');
            vis = vis + c;
            lbls(i) = 1;
        else
            % two elements
            c   = circle_at(Nx, Ny, nx, ny, r1, 0, r2, 0);
            c   = imrotate(c, L, 'crop');
            vis = vis + c;
            c   = circle_at(Nx, Ny, nx, ny, r1, 0, r2, 0);
            c   = imrotate(c, R, 'crop');
            vis = vis + c;
            lbls(i) = 0;
        end
        name = location+"/"+i+extn;
        imwrite(vis, name);
    end
    lbls = categorical(lbls);
end