function [total_return] = Total_Return(portfolio_profits)

[J,N] = size(portfolio_profits);
total_return=[1,N];

for i = 1:N
    pe(i) = portfolio_profits(end,i);
    p1(i) = portfolio_profits(1,i);
    total_return(i)=((pe(i) / p1(i))-1)*100;
end