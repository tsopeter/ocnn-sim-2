classdef CustomClassificationLayer < nnet.layer.ClassificationLayer % ...
        % & nnet.layer.Acceleratable % (Optional)
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
        
    end
 
    methods
        function layer = CustomClassificationLayer(Name)           
            % (Optional) Create a myClassificationLayer.

            % Layer constructor function goes here.
            layer.Name = Name;
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
        end
        
        function dLdY = backwardLoss(layer,Y,T)
            % (Optional) Backward propagate the derivative of the loss 
            % function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the 
            %                 predictions Y

            % Layer backward loss function goes here.
        end
    end
end