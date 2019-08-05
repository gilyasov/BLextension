function [sortino,inforatio] = Ratios(portfolio_returns,BT_SP)

[J,N] = size(portfolio_returns);
avg_exs_ret=[1,N];
avg_exs_negret=[1,N];
sqrtnegret=[1,N];
sortino=[1,N];

avg_exs_ret_SP=[1,N];
vol_exs=[1,N];
returns_SP=BT_SP;
inforatio=[1,N];


for i=1:N
    avg_exs_ret(i) = mean(portfolio_returns(:,i))*52;            % assumption cash = 0
    portfolio_returns(portfolio_returns(:,i)>0)=0;
    avg_exs_negret(i) = mean(portfolio_returns(:,i).^2)*52;
    sqrtnegret(i)=sqrt(avg_exs_negret(i));
    sortino(i)=avg_exs_ret(i)/sqrtnegret(i);
end

for j=1:N
    exs_ret_SP(:,j)=portfolio_returns(:,j)-returns_SP;           % excess returns over S&P500
    avg_exs_ret_SP(j)=mean(exs_ret_SP(:,j))*52;
    vol_exs(j)=std(exs_ret_SP(:,j))*sqrt(52);
    inforatio(j)=(avg_exs_ret_SP(j))/(vol_exs(j));
end