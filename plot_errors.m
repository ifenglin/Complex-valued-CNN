figure;
hold on
s = 5;
color = ['r', 'g', 'b', 'c', 'm'];
plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,1),2)),color(1));
plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,2),2)),color(2));
plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,3),2)),color(3));
plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,4),2)),color(4));
plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,5),2)),color(5));
legend('label 1', 'label 2', 'label 3', 'label 4', 'label 5')
xlabel('epoch')
ylabel(sprintf('average loss in tests where true label = %d', label_plot));
xlim([0 num_epochs])
title(sprintf('errors over epochs in tests where true label = %d', label_plot));
hold off;

