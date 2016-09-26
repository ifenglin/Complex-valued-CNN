figure;
hold on;
plot(0:num_epochs, squeeze(mean(test_loss(:,:),2)));
xlabel('epoch');
ylabel('average loss in tests')
xlim([0 num_epochs])
title('loss over num_epochs')

plot(test_correctness_rate)
title('correctness rate')
xlabel('epoch')
ylabel('correctness rate')
legend('loss', 'correctness')