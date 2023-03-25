classdef CustomLearnablePropagationLayer < nnet.layer.Layer 
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        Nx
        Ny
        Rx
        Ry
        nx
        ny
        w
        b
        wv
        wq
        wf
        wfq
        dist
        Rw
        Iw
        Rwq
        Iwq
        kz
        res_size
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
        d   % distance
    end

    properties (State)
        % (Optional) Layer state parameters.

        % Declare state parameters here.
    end

    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state parameters.

        % Declare nested networks with learnable and state parameters here.
    end

    methods
        function layer = CustomLearnablePropagationLayer(Name, Nx, Ny, Rx, Ry, nx, ny, dist, wv)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name       = Name;
            layer.NumInputs  = 2;
            layer.NumOutputs = 2;
            layer.Nx         = Nx;
            layer.Ny         = Ny;
            layer.nx         = nx;
            layer.ny         = ny;
            layer.Rx         = Rx;
            layer.Ry         = Ry;
            layer.dist       = dist;
            layer.wv         = wv;
            layer.d          = dlarray(dist);
            layer = layer.internal_initialize();
            layer.res_size   = size(dlconv(ones(layer.Nx, layer.Ny, 'single'), layer.Rw, layer.b, 'DataFormat', 'SS'));
        end

        function layer = internal_initialize(layer)
            
            dx = layer.nx/layer.Rx;
            dy = layer.ny/layer.Ry;
        
            rangex = 1/dx;  % number of frequencies available
            rangey = 1/dy;
        
            posx = dlarray(linspace(-rangex/2, rangex/2, layer.Rx));
            posy = dlarray(linspace(-rangey/2, rangey/2, layer.Ry));

            [fxx, fyy] = meshgrid(posy, posx);
        
            kz = 2 * pi * sqrt((1/layer.wv)^2 -(fxx.^2)-(fyy.^2));
        
            W = exp(1i * kz * layer.d);
            W = fftshift(ifft(ifft(ifftshift(W)).').');

            layer.kz = kz;

            layer.w          = W;
            layer.b          = 0;
            layer.wf         = rot90(W, 2);
            layer.Rw         = real(layer.w);
            layer.Iw         = imag(layer.w);
            layer.Rwq        = real(layer.wf);
            layer.Iwq        = imag(layer.wf);
        end
        
        function [Z1, Z2] = predict(layer, R, I)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.
            %
            % Z1 = R * Rw - I * Iw
            % Z2 = R * Iw + I * Rw
        
            W = exp(1i * layer.kz * layer.d);

            W = fftshift(ifft(ifft(ifftshift(W)).').');

            layer.w          = W;
            layer.Rw         = real(layer.w);
            layer.Iw         = imag(layer.w);

            Z1 = dlconv(R, layer.Rw, layer.b, 'DataFormat', 'SSCB') - dlconv(I, layer.Iw, layer.b, 'DataFormat', 'SSCB');
            Z2 = dlconv(R, layer.Iw, layer.b, 'DataFormat', 'SSCB') + dlconv(I, layer.Rw, layer.b, 'DataFormat', 'SSCB');
        end

    %     function [dLdR, dLdI, dLdW] = backward(layer, R, I, Z1, Z2, dLdZ1, dLdZ2, dLdSout)
    %         % (Optional) Backward propagate the derivative of the loss
    %         % function through the layer.
    %         %
    %         % Inputs:
    %         %         layer   - Layer to backward propagate through 
    %         %         X       - Layer input data 
    %         %         Z       - Layer output data 
    %         %         dLdZ    - Derivative of loss with respect to layer 
    %         %                   output
    %         %         dLdSout - (Optional) Derivative of loss with respect 
    %         %                   to state output
    %         %         memory  - Memory value from forward function
    %         % Outputs:
    %         %         dLdX   - Derivative of loss with respect to layer input
    %         %         dLdW   - (Optional) Derivative of loss with respect to
    %         %                  learnable parameter 
    %         %         dLdSin - (Optional) Derivative of loss with respect to 
    %         %                  state input
    %         %
    %         %  - For layers with state parameters, the backward syntax must
    %         %    include both dLdSout and dLdSin, or neither.
    %         %  - For layers with multiple inputs, replace X and dLdX with
    %         %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
    %         %    the number of inputs.
    %         %  - For layers with multiple outputs, replace Z and dlZ with
    %         %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
    %         %    number of outputs.
    %         %  - For layers with multiple learnable parameters, replace 
    %         %    dLdW with dLdW1,...,dLdWP, where P is the number of 
    %         %    learnable parameters.
    %         %  - For layers with multiple state parameters, replace dLdSin
    %         %    and dLdSout with dLdSin1,...,dLdSinK and 
    %         %    dLdSout1,...,dldSoutK, respectively, where K is the number
    %         %    of state parameters.
    % 
    %         % Define layer backward function here.
    %         V  = size(R);
    % 
    %         if length(V) <= 2
    %             V(3) = 1;
    %             V(4) = 1;
    %         end
    % 
    %         % Z1 = conv(R, Rw) - conv(I, Iw)
    %         % Z2 = conv(R, Iw) + conv(I, Rw)
    %         %
    % 
    %         % dLdR = dLd
    %         %
    % 
    %         dLdR = zeros(V, 'like', dLdZ1);
    %         dLdI = zeros(V, 'like', dLdZ2);
    % 
    %        % dLdRw = zeros(size(layer.Rw), 'like', layer.Rw);
    %        % dLdIw = zeros(size(layer.Iw), 'like', layer.Iw);
    % 
    %         % dLdRw =  conv2(R, dLdZ1) + conv2(I, dLdZ2)
    %         % dLdIw = -conv2(I, dLdZ1) + conv2(R, dLdZ2)
    % 
    %         for i=1:V(4)
    %             Re = R(:,:,1,i);
    %             Ie = I(:,:,1,i);
    %             dLdR(:,:,1,i) =      conv2(layer.Rwq, dLdZ1(:,:,1,i), 'full')  + conv2(layer.Iwq, dLdZ2(:,:,1,i), 'full');
    %             dLdI(:,:,1,i) = -1 * conv2(layer.Iwq, dLdZ1(:,:,1,i), 'full')  + conv2(layer.Rwq, dLdZ2(:,:,1,i), 'full');
    %            % dLdRw = dLdRw + conv2(Re, dLdZ1(:,:,1,i), 'valid') + conv2(Ie, dLdZ2(:,:,1,i), 'valid');
    %            % dLdIw = dLdIw - conv2(Ie, dLdZ1(:,:,1,i), 'valid') + conv2(Re, dLdZ2(:,:,1,i), 'valid');
    %         end
    % 
    %     end
    end

    
end