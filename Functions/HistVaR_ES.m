function [VaR,ES] = HistVaR_ES(returns,confidence_level)

% Sort returns from smallest to largest
sorted_returns = sort(returns);

% Store the number of returns
T = length(returns);

% Calculate the index of the sorted return that will be VaR
VaR_index = ceil((1-confidence_level)*T);
%VaR_index = floor(confidence_level*T);

% Use the index to extract VaR from sorted returns
VaR = sorted_returns(VaR_index);
ES = mean(sorted_returns(1:VaR_index,1));

% Plot results
% Histogram data
[count,bins] = hist(returns, 30);
% Create 2nd data set that is zero above Var point
count_cutoff = count.*(bins < VaR);
% Scale bins
scale = (bins(2)-bins(1))*T;
% Plot full data set
bar(bins,count/scale,'b');
hold on;
% Plot cutoff data set
bar(bins,count_cutoff/scale,'r');
grid on;
hold off;
title(['Histogram of Returns. Red Indicates Returns Below VaR: ',num2str(VaR)],'FontWeight','bold');
