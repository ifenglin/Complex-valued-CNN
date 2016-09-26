figure;
hold on
j = label_plot:5:test_num_reps*num_all_labels;
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
legend('label 1', 'label 2', 'label 3', 'label 4', 'label 5')
xlabel('average real')
ylabel('average imag')
title(sprintf('vote over epochs in tests where true label = %d', j(1)))
