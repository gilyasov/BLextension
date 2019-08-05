function [positive_trades] = Positive_Trades(portfolio_returns)

[J,N] = size(portfolio_returns);
positive_trades=[1,N];

for i=1:N
    s(:,i)=sign(portfolio_returns(:,i));
    ifpositif(i)=sum(s(:,i)==1);
    ifnegatif(i)=sum(s(:,i)==-1);
    positive_trades(i)=ifpositif(i)/(ifpositif(i)+ifnegatif(i));
end