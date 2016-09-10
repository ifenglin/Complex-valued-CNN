figure;
hold on
j = 5:5:25;
s = 5;
color = ['r', 'g', 'b', 'c', 'm'];
plot(1:epochs+1, squeeze(mean(test_errors(:,j:s:end,1),2)),color(1));
plot(1:epochs+1, squeeze(mean(test_errors(:,j:s:end,2),2)),color(2));
plot(1:epochs+1, squeeze(mean(test_errors(:,j:s:end,3),2)),color(3));
plot(1:epochs+1, squeeze(mean(test_errors(:,j:s:end,4),2)),color(4));
plot(1:epochs+1, squeeze(mean(test_errors(:,j:s:end,5),2)),color(5));
legend('label 1', 'label 2', 'label 3', 'label 4', 'label 5')
xlabel('epoch')
ylabel(sprintf('average loss in tests in which true label = %d', j))
xlim([1 epochs+1])
title(sprintf('erros over epochs where true label = %d', j));
hold off;

figure;
plot(1:epochs+1, squeeze(mean(test_loss(:,:),2)));
xlabel('epoch');
ylabel('average loss in tests')
xlim([1 epochs+1])
title('loss over epochs')

figure;
plot(test_correctness_rate)
title('correctness rate')
xlabel('epoch')
ylabel('correctness rate')


