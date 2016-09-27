function plot_loss(num_epochs, test_loss, test_correctness_rate)
    figure
    hold on
    plot(0:num_epochs, squeeze(mean(test_loss(:,:),2)));
    xlim([0 num_epochs])
    
    plot(test_correctness_rate)
    xlabel('epoch')
    ylabel('average loss / correctness rate [%]')
    legend('loss', 'correctness')
    title('loss and correctness rate over epochs')
    hold off
end