classdef CustomNonlinearLayer < nnet.layer.Layer % ...
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        a0 = 20;
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
        function layer = CustomNonlinearLayer(NumInputs, Name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = NumInputs;
            layer.NumOutputs = 2;
        end
        
        function [Z1, Z2] = predict(layer,X1, X2)
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
            if (length(W)==2)
                M = sqrt(X1.^2 + X2.^2);
                G = single(nonlinear_forward(M, layer.a0))./M;
                Z1 = G .* X1;
                Z2 = G .* X2;
            else
                AZ1 = zeros(W,'single');
                AZ2 = zeros(W,'single');
                for i=1:W(4)
                    QX = X1(:,:,1,i);
                    QY = X2(:,:,1,i);
                    M = sqrt(QX.^2+QY.^2);
                    G = single(nonlinear_forward(M, layer.a0)./M);
                    AZ1(:,:,1,i)=single(QX .* G);
                    AZ2(:,:,1,i)=single(QY .* G);
                end
                Z1 = gpuArray(AZ1);
                Z2 = gpuArray(AZ2);
            end
            
        end
    end
end