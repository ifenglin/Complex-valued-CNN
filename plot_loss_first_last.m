figure;
hold on
epoch = 1;
%difference = squeeze(test_errors(epoch+1,:,:)) - squeeze(test_errors(epoch,:,:)); 
difference = test_last_errors - test_first_errors;
bar(difference);
legend('1', '2' , '3', '4', '5');
ylabel('error change')
xlabel('tests')
xlim([1 test_num*num_all_labels]);
title('error for each label before and after training.');