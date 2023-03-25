classdef CustomClassificationLayer < nnet.layer.ClassificationLayer % ...
        % & nnet.layer.Acceleratable % (Optional)
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
        modifier
    end
 
    methods
        function layer = CustomClassificationLayer(Name, modifier)           
            % (Optional) Create a myClassificationLayer.

            % Layer constructor function goes here.
            layer.Name = Name;
            layer.modifier = modifier;
        end

        function loss = forwardLoss(layer,Y,T)
            % Return the loss between the predictions Y and the training 
            % targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            loss = 0.5 * sum((T-Y).^2, 'all')/length(size(Y,4));
        end

         function dYdL = backwardLoss(layer, Y, T)
             dYdL = Y - T;
         end
    end
end