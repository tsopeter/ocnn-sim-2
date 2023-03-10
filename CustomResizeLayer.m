classdef CustomResizeLayer < nnet.layer.Layer % ...
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        Nx
        Ny
        k
        lvalue
        P
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
        function layer = CustomResizeLayer(NumInputs, Name, Nx, Ny, k, lvalue, P)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = NumInputs;
            layer.NumOutputs = 1;
            layer.Nx = Nx;
            layer.Ny = Ny;
            layer.k  = k;
            layer.lvalue = lvalue;
            layer.P = P;
        end
        
        function Z = predict(layer,X)
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
            W = size(X);
            Z = (resize_normalize_extend(X, layer.Nx, layer.Ny, layer.k, layer.lvalue)) * layer.P;
            if length(W) > 2
                Z = gpuArray(Z);
            end
        end

        function [Z, state] = forward(layer,X)
            % (Optional) Forward input data through the layer at training
            % time and output the result, the updated state, and a memory
            % value.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Layer input data
            % Outputs:
            %         Z      - Output of layer forward function 
            %         state  - (Optional) Updated layer state 
            %         memory - (Optional) Memory value for custom backward
            %                  function
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer forward function here.
            Z = layer.predict(X);
            state = 0;
        end

        function dLdX = backward(layer,X,Z,dLdZ,dLdSout)
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
            T = size(interp2(X(:,:,1,1), layer.k));
            W = size(dLdZ);
            if length(W) <= 2
                W(3)=1;
                W(4)=1;
            end
            stx  = layer.Nx/2 - round(T(1)/2);
            endx = layer.Nx/2 + round(T(1)/2);
            sty  = layer.Ny/2 - round(T(2)/2);
            endy = layer.Ny/2 + round(T(2)/2);
            AZ   = dLdZ(stx:endx,sty:endy,:,:);
            dLdX = zeros(size(X), 'single');
            xq = (0:(T(1)/W(1)):T(1))';
            yq = (0:(T(2)/W(2)):T(2))';
            for i=1:W(4)
                F = griddedInterpolant({T(1), T(2)}, AZ(:,:,1,i));
                dLdX(:,:,1,i)=F({xq, yq});
            end
        end
    end
end