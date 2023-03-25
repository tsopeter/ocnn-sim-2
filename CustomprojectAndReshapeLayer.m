classdef CustomprojectAndReshapeLayer < nnet.layer.Layer ...
       % & nnet.layer.Formattable ...
       % & nnet.layer.Acceleratable

    properties
        % Layer properties.
        OutputSize
    end

    properties (Learnable)
        % Layer learnable parameters.
    end

    methods
        function layer = CustomprojectAndReshapeLayer(outputSize, Name)
            % layer = projectAndReshapeLayer(outputSize)
            % creates a projectAndReshapeLayer object that projects and
            % reshapes the input to the specified output size.
            %
            % layer = projectAndReshapeLayer(outputSize,Name=name)
            % also specifies the layer name.

            % Parse input arguments.
            

            % Set layer name.
            layer.Name = Name;

            % Set output size.
            layer.OutputSize = outputSize;
        end

        function layer = initialize(layer,layout)
            % layer = initialize(layer,layout) initializes the layer
            % learnable parameters.
            %
            % Inputs:
            %         layer  - Layer to initialize
            %         layout - Data layout, specified as a 
            %                  networkDataLayout object
            %
            % Outputs:
            %         layer - Initialized layer

            % Layer output size.
        end

        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data, specified as a formatted dlarray
            %                 with a "C" and optionally a "B" dimension.
            % Outputs:
            %         Z     - Output of layer forward function returned as
            %                 a formatted dlarray with format "SSCB".

            % Fully connect.

            % Reshape.
            outputSize = layer.OutputSize;
            Z = reshape(X,outputSize(1),outputSize(2),outputSize(3),[]);
        end
    end
end