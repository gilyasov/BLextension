%% Cleaning of the session
clear variables; close all; clc;

API_KEY = '9UntWxT7BNxose7552jo';

c = quandl(API_KEY);

format bank;

s = 'FINRA/FNRA_AGG';
s = 'FINRA/FNRA_IWM';

% Time period
startdate = datetime('01-01-2017','InputFormat','MM-dd-yyyy');
enddate = datetime('12-31-2018','InputFormat','MM-dd-yyyy');
periodicity = 'monthly';

% s = 'SP500';%'AGG','IWM','SP500','TNX','VIX','VNQ','BTC_USD'};
d = history(c,s);
% d = history(c,s,startdate,enddate,periodicity,'transform','rdiff','order','asc');
disp(d);
