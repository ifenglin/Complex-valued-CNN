function plot_outputs(test_size_batch, test_output_data, test_names_labels, label_plot)
figure;
hold on
j = label_plot:5:test_size_batch;
color = ['r', 'g', 'b', 'c', 'm'];
for i = 1:5  
    plot(real(squeeze(mean(test_output_data(:,j,i),2))), imag(squeeze(mean(test_output_data(:,j,i),2))),sprintf('%c',color(i)));
end
for i = 1:5  
    plot(real(squeeze(mean(test_output_data(1,j,i),2))), imag(squeeze(mean(test_output_data(1,j,i),2))), sprintf('%co',color(i)));
    plot(real(squeeze(mean(test_output_data(end,j,i),2))), imag(squeeze(mean(test_output_data(end,j,i),2))), sprintf('%c*',color(i)));
end
a = 1:0.1:2*pi;
x = cos(a);
y = sin(a);
plot(x,y)
for i=1:5
    text(cos((i-1)*72*pi/180),sin((i-1)*72*pi/180),num2str(i),'Color', color(i));
end
legend(test_names_labels{1}, test_names_labels{2}, test_names_labels{3},...
        test_names_labels{4}, test_names_labels{5});
xlabel('real part of average outputs')
ylabel('imagery part of average outputs')
title(sprintf('Outputs for each class over epochs in tests where true class = %s', test_names_labels{label_plot}))
