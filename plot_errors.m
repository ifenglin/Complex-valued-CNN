function plot_errors(num_epochs, test_errors, test_names_labels, label_plot)
    figure;
    hold on
    s = 5;
    color = ['r', 'g', 'b', 'c', 'm'];
    plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,1),2)),color(1));
    plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,2),2)),color(2));
    plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,3),2)),color(3));
    plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,4),2)),color(4));
    plot(0:num_epochs, squeeze(mean(test_errors(:,label_plot:s:end,5),2)),color(5));
    legend(test_names_labels{1}, test_names_labels{2}, test_names_labels{3},...
        test_names_labels{4}, test_names_labels{5});
    xlabel('epoch')
    ylabel(sprintf('average loss'));
    xlim([0 num_epochs])
    title(sprintf('errors over epochs in tests where true label = %s', test_names_labels{label_plot}));
    hold off;
end
