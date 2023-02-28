classdef CustomFlattenLayer < nnet.layer.Layer & nnet.layer.Acceleratable 
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
        function layer = CustomFlattenLayer(NumInputs, Name, Nx, Ny, nx, ny, r1, r2)
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
            layer.plate = detector_plate(Nx, Ny, nx, ny, r1, r2);
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
            end

            Z = dlarray(zeros(1,1,10,W(4), 'single'));
            for i=1:W(4)
                Z(1,1,:,i)=detector_values(X1(:,:,1,i), layer.Nx, layer.Ny, layer.nx, layer.ny, layer.r1, layer.r2);
            end
        end
    end
end