classdef classification_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        classifier    % classification type
        num           % number of classes
        delta         % hyperparameter - thershold that one class surpasses another
        lambda        % hyperparameter - regulaization
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
        function [self, output_data, est_label, errors, loss] = forward(self, input_blob)
            input_data = input_blob.get_data();
            assert(length(input_blob.get_data()) == self.num)
            errors = zeros(self.num, 1);
            labels = [ 1+0i 0.309+0.9511i -0.809+0.5878i -0.809-0.5878i 0.3090-0.9511i ];
            my_delta = self.delta;
            parfor i = 1:self.num % calculate the loss when the true label is i
%                 for j = 1:self.num % calculate the contribution from input j
%                     if i ~= j 
%                         % complex quadratic error function
%                         mysigma = complex(0,0) - input_data(j);
%                         loss(i) = loss(i) + max([0, 1/2 * (mysigma* conj(mysigma)) - self.delta]);
%                     else
%                         mysigma = complex(0,0) - input_data(j);
%                         loss(i) = loss(i) + max([0, 1/2 * (mysigma* conj(mysigma)) - self.delta]);
%                     end
%                 end
                  mysigma = labels(i) - input_data(i);
                  errors(i) = max([0, 1/2 * (mysigma* conj(mysigma)) - my_delta]);
            end
            % find estimated label
            [loss, est_label] = min(errors);
            % calculate the total loss
             parfor i = 1:self.num % calculate the loss for label i
                 if i ~= est_label
                     mysigma = complex(0, 0) - input_data(i);
                     loss =  loss + max([0, 1/2 * mysigma* conj(mysigma) - my_delta]);
                 end
             end
            % weight regularization: add to loss
%             loss = loss + self.lambda * sum(arrayfun(@norm, self.weight_vector).^2);
            output_data = input_data;
        end
        
        function [self, output_diff] = backward(self, input_blob, known_labels)
            %labels = self.known_labels;
            input_data = input_blob.get_data();
            % check dimensions
            assert(length(input_data) == self.num)
            assert(length(known_labels) == self.num)
            % set true label
            true_label = find(known_labels == 1);
            % mask false labels to 0
            known_labels = [ 1+0i 0.309+0.9511i -0.809+0.5878i -0.809-0.5878i 0.3090-0.9511i ] .* known_labels;
            % initialize variable
            output_diff = zeros(self.num, 1);
            false_positive_count = 0;
            my_delta = self.delta;
            % calculate the gradient for the unit corresponding to false label
            parfor i = 1:length(known_labels)
                if i ~= true_label
                    mysigma = known_labels(i) - input_data(i);
                    if (1/2 * (mysigma* conj(mysigma)) - my_delta) > 0 % otherwise diff is zero
                        false_positive_count = false_positive_count + 1;
                        % left-right opposite
                        %output_diff(i) = -2 * mysigma(i) * conj(mysigma(i))^2 ; 
                        % upside-down
                        %output_diff(i) = -2 * mysigma(i) * conj(mysigma(i))^2 * -1;
                        %output_diff(i) = 2 * conj( mysigma(i) * conj(mysigma(i))^2);
                        output_diff(i) = 1/2 * mysigma ;
                    end
                end
            end
            %calculate the gradient for the unit corresponding to true label
            mysigma = known_labels(true_label) - input_data(true_label);
            output_diff(true_label) = 1/2 * false_positive_count * mysigma;
            
            % weight regularization: add gradient of regularization loss
%             output_diff = output_diff + 2*self.lambda*sum(self.weight_vector);
        end
    end
end

