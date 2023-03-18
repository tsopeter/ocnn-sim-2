classdef CustomPropagationLayer < nnet.layer.Layer 
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        Nx
        Ny
        w
        wq
        wf
        wfq
        dist
        Rw
        Iw
        Rwq
        Iwq
        nIwq;
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
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
        function layer = CustomPropagationLayer(Name, Nx, Ny, dist, W)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name       = Name;
            layer.NumInputs  = 2;
            layer.NumOutputs = 2;
            layer.Nx         = Nx;
            layer.Ny         = Ny;
            layer.dist       = dist;
            layer.w          = W;
            layer.wq         = fft2(layer.w);
            layer.wf         = W';
            layer.wfq        = fft2(layer.wf);

            layer.Rw         = fft2(real(layer.w));
            layer.Iw         = fft2(imag(layer.w));
            layer.Rwq        = fft2(real(layer.w.'));
            layer.Iwq        = fft2(imag(layer.w.'));
            layer.nIwq       = fft2(imag((-1 * layer.w).'));
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
            W  = size(R);

            if length(W) <= 2
                W(3) = 1;
                W(4) = 1;
            end

            Z1 = zeros(W, 'like', R);
            Z2 = zeros(W, 'like', I);

            %
            % Z1 = R * Rw - I * Iw
            % Z2 = R * Iw + I * Rw

            for i=1:W(4)
                Z1(:,:,1,i) = dlfft2(R(:,:,1,i), layer.Rw) - dlfft2(I(:,:,1,i), layer.Iw);
                Z2(:,:,1,i) = dlfft2(R(:,:,1,i), layer.Iw) + dlfft2(I(:,:,1,i), layer.Rw);
            end
        end

        function [dLdR, dLdI] = backward(layer, R, I, Z1, Z2, dLdZ1, dLdZ2, dLdSout)
            % (Optional) Backward propagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %         layer   - Layer to backward propagate through 
            %         X       - Layer input data 
            %         Z       - Layer output data 
            %         dLdZ    - Derivative of loss with respect to layer 
            %                   output
            %         dLdSout - (Optional) Derivative of loss with respect 
            %                   to state output
            %         memory  - Memory value from forward function
            % Outputs:
            %         dLdX   - Derivative of loss with respect to layer input
            %         dLdW   - (Optional) Derivative of loss with respect to
            %                  learnable parameter 
            %         dLdSin - (Optional) Derivative of loss with respect to 
            %                  state input
            %
            %  - For layers with state parameters, the backward syntax must
            %    include both dLdSout and dLdSin, or neither.
            %  - For layers with multiple inputs, replace X and dLdX with
            %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
            %    the number of inputs.
            %  - For layers with multiple outputs, replace Z and dlZ with
            %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
            %    number of outputs.
            %  - For layers with multiple learnable parameters, replace 
            %    dLdW with dLdW1,...,dLdWP, where P is the number of 
            %    learnable parameters.
            %  - For layers with multiple state parameters, replace dLdSin
            %    and dLdSout with dLdSin1,...,dLdSinK and 
            %    dLdSout1,...,dldSoutK, respectively, where K is the number
            %    of state parameters.

            % Define layer backward function here.
            W  = size(R);

            if length(W) <= 2
                W(3) = 1;
                W(4) = 1;
            end

            dLdR = zeros(W, 'like', dLdZ1);
            dLdI = zeros(W, 'like', dLdZ2);

            for i=1:W(4)
                dLdR(:,:,1,i) = dlfft2(dLdZ1(:,:,1,i), layer.Rwq)  + dlfft2(dLdZ2(:,:,1,i), layer.Iwq);
                dLdI(:,:,1,i) = dlfft2(dLdZ1(:,:,1,i), layer.nIwq) + dlfft2(dLdZ2(:,:,1,i), layer.Rwq);
            end

        end

    end
end