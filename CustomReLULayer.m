classdef CustomReLULayer < nnet.layer.Layer % ...
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        sx
        sy
        sc
        sz
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
        function layer = CustomReLULayer(NumInputs, Name, lvalue, sx, sy, sc, sz)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = Name;
            layer.NumInputs = NumInputs;
            layer.NumOutputs = 2;
            layer.lvalue = lvalue;
            layer.sx = sx;
            layer.sy = sy;
            layer.sc = sc;
            layer.sz = sz;
        end
        
        function [Z1, Z2] = predict(layer,X1,X2)
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
            Q = sqrt(X1.^2+X2.^2);
            C = 1 ./ Q;
            Z = layer.sc * (tanh((Q/layer.sz)-layer.sx)+layer.sy);
            Z(Z==0)=layer.lvalue;
            Z1 = Z .* (X1 .* C);
            Z2 = Z .* (X2 .* C);
        end
% 
%         function [dLdX1, dLdX2] = backward(layer,X1, X2, Z1, Z2, dLdZ1, dLdZ2, dLdSout)
%             % (Optional) Backward propagate the derivative of the loss
%             % function through the layer.
%             %
%             % Inputs:
%             %         layer   - Layer to backward propagate through 
%             %         X       - Layer input data 
%             %         Z       - Layer output data 
%             %         dLdZ    - Derivative of loss with respect to layer 
%             %                   output
%             %         dLdSout - (Optional) Derivative of loss with respect 
%             %                   to state output
%             %         memory  - Memory value from forward function
%             % Outputs:
%             %         dLdX   - Derivative of loss with respect to layer input
%             %         dLdW   - (Optional) Derivative of loss with respect to
%             %                  learnable parameter 
%             %         dLdSin - (Optional) Derivative of loss with respect to 
%             %                  state input
%             %
%             %  - For layers with state parameters, the backward syntax must
%             %    include both dLdSout and dLdSin, or neither.
%             %  - For layers with multiple inputs, replace X and dLdX with
%             %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
%             %    the number of inputs.
%             %  - For layers with multiple outputs, replace Z and dlZ with
%             %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
%             %    number of outputs.
%             %  - For layers with multiple learnable parameters, replace 
%             %    dLdW with dLdW1,...,dLdWP, where P is the number of 
%             %    learnable parameters.
%             %  - For layers with multiple state parameters, replace dLdSin
%             %    and dLdSout with dLdSin1,...,dLdSinK and 
%             %    dLdSout1,...,dldSoutK, respectively, where K is the number
%             %    of state parameters.
% 
%             % Define layer backward function here.
%             Q = sqrt(X1.^2+X2.^2);
%             Q(Q==0)=layer.lvalue;
%             C = 1 ./ Q;
%             ZQ   =layer.sc * (tanh((Q/layer.sz)-layer.sx)+layer.sy);
%             dZdQ = layer.sc * (sech((Q/layer.sz)-layer.sx).^2) ./ layer.sz;
%             ZQ(ZQ==0)=layer.lvalue;
%             dZdQ(dZdQ==0)=layer.lvalue;
%             dZ1dX1 = C .* (ZQ + X1 .* (dZdQ .* C .* X1 + ZQ .* X1));
%             dZ1dX2 = X1 .* (dZdQ .* C .* X2 + ZQ .* C .* X2);
%             dZ2dX1 = X2 .* (dZdQ .* C .* X1 + ZQ .* C .* X1);
%             dZ2dX2 = C .* (ZQ + X2 .* (dZdQ .* C .* X2 + ZQ .* X2));
%             dLdX1 = (dLdZ1 .* dZ1dX1) + (dLdZ2 .* dZ2dX1);
%             dLdX2 = (dLdZ2 .* dZ2dX2) + (dLdZ1 .* dZ1dX2);
%         end
    end
end