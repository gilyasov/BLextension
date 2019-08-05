%---------Global-Mean-Variance Model------------

%1. You have your returns in Excel file
%2. Let them be incorporated into matlab through xlsread
%3. Only thing you need to change is the name of the excel file, which
%sheet and where it is located. 
%4. For the 1/N model, define the number of total assets. 

returns=xlsread('Name of Excel File.xls','Which Sheet?','A1:A44');

%The Sample Covariance Matrix
S=cov(returns);

%The Sample Mean Vector
m=mean(returns)';

%Calculation of GMV-weights
iota=ones(size(S,1),1);
w_GMV=(inv(S)*iota)/(iota'*inv(S)*iota)

%----------The 1/N allocation model-------------
%Define the total assets first;
total_assets=8;

w_1N=ones(1,size(returns,2))*(1/total_assets);