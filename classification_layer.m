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
        function self = set_delta(self, delta)
            self.delta = delta;
        end
        function self = set_lambda(self, lambda)
            self.lambda = lambda;
        end
        function self = get_delta(self, delta)
            self.delta = delta;
        end
        function [delta, lambda] = get_parameters(self)
            delta = self.delta;
            lambda = self.lambda;
        end
        function est_label = get_estimation(self)
            est_label = self.est_label;
        end
        function self = set_weight_vector(self, weight_vector)
            self.weight_vector = weight_vector;
        end
        function weight_vector = get_weight_vector(self)
            weight_vector = self.weight_vector;
        end
        function [self, output_data, est_label, loss] = forward(self, input_blob)
            input_data = input_blob.get_data();
            self.forward_input_data = input_data;
            assert(length(input_blob.get_data()) == self.num)
            loss = zeros(self.num, 1);
            for i = 1:self.num % calculate the loss when the true label is i
                for j = 1:self.num % calculate the contribution from input j
                    if i ~= j 
                        % complex quadratic error function
                        mysigma = complex(0,0) - input_data(j);
                        loss(i) = loss(i) + max([0, 1/2 * mysigma* conj(mysigma) - self.delta]);
                    else
                        mysigma = complex(1,1) - input_data(j);
                        loss(i) = loss(i) + max([0, 1/2 * mysigma* conj(mysigma) - self.delta]);
                    end
                end
            end
%             for i = 1:self.num % calculate the loss for label i
%                 mysigma = complex(1, 1) - input_data(j);
%                 loss(i) = max([0, 1/2 * mysigma* conj(mysigma) - self.delta]);
%             end
            % add regularization loss
            loss = loss + self.lambda * sum(arrayfun(@norm, self.weight_vector).^2);
            % find estimated label
            [~, est_label] = min(loss);
            output_data = input_data;
        end
        function [self, output_diff] = backward(self, ~)
            labels = self.known_labels;
            input_data = self.forward_input_data;
            assert(length(input_data) == self.num)
            assert(length(labels) == self.num)
            true_label = find(labels == 1);
            labels = complex(labels, labels);
            output_diff = zeros(self.num, 1);
            mysigma = zeros(length(labels), 1);
            false_positive_count = 0;
            % calculate the gradient for the unit corresponding to false label
            for i = 1:length(labels)
                if i ~= true_label
                    mysigma(i) = labels(i) - input_data(i);
                    if (1/2 * mysigma(i)* conj(mysigma(i)) - self.delta) > 0 % otherwise diff is zero
                        false_positive_count = false_positive_count + 1;
                        output_diff(i) = 1/2 * conj(input_data(i)); 
                    end
                end
            end
            % calculate the gradient for the unit corresponding to true label
            output_diff(true_label) = -1/2 * false_positive_count * conj(input_data(true_label));
            
            % add gradient of regularization loss
            output_diff = output_diff + 2*self.lambda*sum(self.weight_vector);
        end
    end
end

