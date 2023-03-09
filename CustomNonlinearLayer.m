classdef CustomNonlinearLayer < nnet.layer.Layer % & nnet.layer.Acceleratable  
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        a0 = 20;
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
        function layer = CustomNonlinearLayer(NumInputs, Name, lvalue)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = NumInputs;
            layer.NumOutputs = 2;
            layer.lvalue = lvalue;
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
            if length(W)<=2
                W(3)=1;
                W(4)=1;
            end
            AZ1 = zeros(W,'single');
            AZ2 = zeros(W,'single');
            for i=1:W(4)
                QX = X1(:,:,1,i);
                QY = X2(:,:,1,i);
                M = sqrt(QX.^2+QY.^2);
                M(M==0) = layer.lvalue;
                G = single(nonlinear_forward(M, layer.a0)./M);
                AZ1(:,:,1,i)=single(QX .* G);
                AZ2(:,:,1,i)=single(QY .* G);
            end
            if W(4) == 1
                Z1 = single(AZ1);
                Z2 = single(AZ2);
            else
                Z1 = gpuArray(AZ1);
                Z2 = gpuArray(AZ2);
            end
        end

        function [dLdX1, dLdX2] = backward(layer,X1, X2, Z1, Z2, dLdZ1, dLdZ2, dLdSout)
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
            %
            %
            % Input dLdZ comes in as [X][Y][10][N]
            % We want each channel [1-10] to be
            % dLidX = f'(X,Y)./abs(X+Yi)...
            %
            %

            % Define layer backward function here.
            W = size(dLdZ1);
            AdLdX1 = gpuArray(zeros(W,'single'));
            AdLdX2 = gpuArray(zeros(W,'single'));

            if length(W)<=2
                W(3)=1;
                W(4)=1;
            end

            for i=1:W(4)
                QX   = X1(:,:,1,i);
                QY   = X2(:,:,1,i);

                %
                %
                % common terms
                C       = sqrt(QX.^2+QY.^2);
                C(C==0) = layer.lvalue;
                Cinv    = 1 ./ C;

                Common1 = nonlinear_forward(C, layer.a0);
                Common2 = Common1 .* Cinv;
                Common3 = nonlinear_backward(C, layer.a0) .* Cinv;

                dZ1dX1 = Common2 + QX .* (Common3 .* QX .* Cinv + QX .* Cinv .* Common1);
                dZ1dX2 = QX .* (Common3 .* QY .* Cinv + QY .* Common2);

                dZ2dX1 = QY .* (Common3 .* QX .* Cinv + QX .* Common2);
                dZ2dX2 = Common2 + QY .* (Common3 .* QY .* Cinv + QY .* Cinv .* Common1);


                AdLdX1(:,:,1,i)=dLdZ1(:,:,1,i) .* dZ1dX1 + dLdZ2(:,:,1,i) .* dZ2dX1;
                AdLdX2(:,:,1,i)=dLdZ1(:,:,1,i) .* dZ1dX2 + dLdZ2(:,:,1,i) .* dZ2dX2;
            end
            dLdX1 = single(AdLdX1);
            dLdX2 = single(AdLdX2);
        end
    end
end