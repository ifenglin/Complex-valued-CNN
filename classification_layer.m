classdef classification_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        classifier    % classification type
        num           % number of classes
        delta         % hyperparameter - thershold that one class surpasses another
        lambda        % hyperparameter - regulaization
        known_labels  % known labels
        forward_input_data % copy forward_input_blob for backpropagation
        weight_vector % all weights in the net for loss calculation
        est_label     % index of the estimated label
    end 
    
    methods (Access = public)
        function self = classification_layer(classifier, num, delta, lambda)
            self@Layer('classification');
            self.classifier = classifier;
            self.num = num;
            self.delta = delta;
            self.lambda = lambda;
            self.est_label = 0;
        end
        function self = set_labels(self, labels)
            self.known_labels = labels;
        end
        function est_label = get_estimation(self)
            est_label = self.est_label;
        end
        function self = set_weight_vector(self, weight_vector)
            self.weight_vector = weight_vector;
        end
        function [self, output_data, label] = forward(self, input_blob)
            labels = self.known_labels;
            input_data = input_blob.get_data();
            self.forward_input_data = input_data;
            assert(length(input_blob.get_data()) == self.num)
            assert(length(labels) == self.num)
            loss = 0;
            [~, true_label] = max(labels);
            mag_input_data = arrayfun(@norm, input_data);
            for i = 1:length(labels)
                if i ~= true_label
                    % we implement maginitude for calculating loss
                    loss = loss + max([0  (mag_input_data(i) - mag_input_data(true_label) + self.delta)]);
                end
            end
            % add regularization loss
            loss = loss + self.lambda*sum(arrayfun(@norm, self.weight_vector).^2);
            [~, label] = max(mag_input_data);
            output_data = [input_data; loss];
        end
        function [self, output_diff] = backward(self, ~)
            labels = self.known_labels;
            input_data = self.forward_input_data;
            assert(length(input_data) == self.num)
            assert(length(labels) == self.num)
            output_diff = zeros(self.num, 1);
            [~, true_label] = max(labels);
            mag_input_data = arrayfun(@norm, input_data);
            false_positive_count = 0;
            % calculate the gradient for the unit corresponding to false label
            for i = 1:length(labels)
                if i ~= true_label
                    % we implement maginitude for calculating loss
                    if (mag_input_data(i) - mag_input_data(true_label) + self.delta) > 0
                        false_positive_count = false_positive_count + 1;
                        output_diff(i) = input_data(i); 
                        % otherwise diff is zero
                    end
                end
            end
            % calculate the gradient for the unit corresponding to true label
            output_diff(true_label) = -false_positive_count*input_data(true_label);
            
            % add gradient of regularization loss
            output_diff = output_diff + 2*self.lambda*sum(self.weight_vector);
        end
    end
end

