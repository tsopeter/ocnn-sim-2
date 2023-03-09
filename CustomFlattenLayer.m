classdef CustomFlattenLayer < nnet.layer.Layer 
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        r1;
        r2;
        nx;
        ny;
        Nx;
        Ny;
        plate;
        plates;
        lvalue;
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
        function layer = CustomFlattenLayer(NumInputs, Name, Nx, Ny, nx, ny, r1, r2, lvalue)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = NumInputs;
            layer.NumOutputs = 1;
            layer.Nx = Nx;
            layer.Ny = Ny;
            layer.nx = nx;
            layer.ny = ny;
            layer.r1 = r1;
            layer.r2 = r2;
            layer.lvalue = lvalue;
            layer.plate = detector_plate(Nx, Ny, nx, ny, r1, r2, lvalue);
           
            for r=0:9
                layer.plates(:,:,r+1)=imrotate(circle_at(Nx, Ny, nx, ny, r1, 0, r2, lvalue), 36*r, 'crop');
            end
            layer.plates(layer.plates==0)=layer.lvalue;
        end
        
        function Z = predict(layer,X1)
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
            W = size(X1);
            if length(W)<=2
                W(3)=1;
                W(4)=1;
                Z = zeros(1,1,10,W(4), 'single');
            else
                Z = gpuArray(zeros(1,1,10,W(4), 'single'));
            end

            for i=1:W(4)
                for r=0:9
                    Z(1,1,r+1,i)=sum(sum(X1(:,:,1,i).*layer.plates(:,:,r+1)));
                end
            end
        end

        function dLdX1 = backward(layer,X1,Z,dLdZ,dLdSout)
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

            % Define layer backward function here
            % 
            % The input data X
            % comes in as
            % [X][Y][1][N]
            %
            %
            % The input dLdZ data comes as
            % [1][1][10][N]
            % 
            % We want the output dLdX1
            % to be in
            % [X][Y][10][N]%
            %
            % Each output Z is defined as 
            %    Z(i) = [zi1, zi2, zi3, ..., zi10]T
            %
            %
            %    where Zij = sum(sum(X.*plate(j)));
            %
            %
            %
            %
            % .
            W = size(X1);
            if length(W) <= 2
                W(3)=1;
                W(4)=1;
                dLdX1 = zeros(W, 'single');
            else
                dLdX1 = gpuArray(zeros(W, 'single'));
            end

            for i=1:W(4)
                for j=1:10
                    dLdX1(:,:,1,i) = dLdX1(:,:,1,i) + (layer.plates(:,:,j) * dLdZ(1,1,j,i));
                end
            end
        end
    end
end