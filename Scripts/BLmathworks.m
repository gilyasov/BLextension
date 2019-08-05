%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% David Wang, Engineering Science 1T8+PEY
% MIE377 - Financial Optimization Models
% blacklitterman.m 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Cleaning of the session
clear variables; close all; clc;

% Disable the warning: "Table variable names were modified to make them valid
warning('off','MATLAB:table:ModifiedAndSavedVarnames'); % MATLAB identifiers"
%#ok<*UNRCH>
%#ok<*SAGROW>
 
%% Import all raw data into the workspace

% Constants
DATA_FOLDER = '..\Data\';
CSV_EXT     = '.csv';
% FILES_EXT   = '.xlsx';

% Raw asset data from CSV files
tableAGG_Asset   = readtable([DATA_FOLDER    'AGG'    CSV_EXT]);
tableVBTIX_Asset = readtable([DATA_FOLDER   'VBTIX'   CSV_EXT]);
tableNKE_Asset   = readtable([DATA_FOLDER    'NKE'    CSV_EXT]);
tableMSFT_Asset  = readtable([DATA_FOLDER    'MSFT'   CSV_EXT]);
tableHCP_Asset   = readtable([DATA_FOLDER    'HCP'    CSV_EXT]);
tableBTC_Asset   = readtable([DATA_FOLDER  'BTC_USD'  CSV_EXT]);
tableDLR_Asset   = readtable([DATA_FOLDER    'DLR'    CSV_EXT]);
tableDBC_Asset   = readtable([DATA_FOLDER    'DBC'    CSV_EXT]);
tableGSG_Asset   = readtable([DATA_FOLDER    'GSG'    CSV_EXT]);
tableBX_Asset    = readtable([DATA_FOLDER    'BX'     CSV_EXT]);
tableLAZ_Asset   = readtable([DATA_FOLDER    'LAZ'    CSV_EXT]);
tableKKR_Asset   = readtable([DATA_FOLDER    'KKR'    CSV_EXT]);

% Raw market capitalization data from XLSX files
tableMarketCap   = readtable([DATA_FOLDER 'MarketCap.xlsx']);
tableMktCap      = tableMarketCap(1,2:end);
% % Raw data from file
% tablePrices = readtable([DATA_FOLDER 'BlackLitterman' FILES_EXT]);
% tableMktCap = readtable([DATA_FOLDER 'BlackLitterman' FILES_EXT],...
%                         'Sheet','capitalization','ReadVariableNames',false);

%% Create the Prices table with all stock exchange close values

% Constants
VAR_NAMES  = {  'AGG' ,  'VBTIX'  , 'NKE',  'MSFT' ,  'HCP' ,  'DLR'  ,  'DBC' ,  'GSG' ,  'BX'   ,  'LAZ' ,  'KKR' ,'BTC_USD'};
VAR_TYPES  = {'double', 'double','double','double' ,'double', 'double','double','double', 'double','double','double', 'double'};
NB_COLUMNS = numel(VAR_NAMES);
NB_ENTRIES = height(tableAGG_Asset);

% Empty table with headers only
tablePrice = table('Size',[NB_ENTRIES NB_COLUMNS],...
                   'VariableTypes',VAR_TYPES,...
                   'VariableNames',VAR_NAMES);

% Fulfill the table with the needed columns of all data tables
tablePrice.(VAR_NAMES{1})  = tableAGG_Asset.Close;
tablePrice.(VAR_NAMES{2})  = tableVBTIX_Asset.Close;
tablePrice.(VAR_NAMES{3})  = tableNKE_Asset.Close;
tablePrice.(VAR_NAMES{4})  = tableMSFT_Asset.Close;
tablePrice.(VAR_NAMES{5})  = tableHCP_Asset.Close;
tablePrice.(VAR_NAMES{6})  = tableDLR_Asset.Close;
tablePrice.(VAR_NAMES{7}) = tableDBC_Asset.Close;
tablePrice.(VAR_NAMES{8}) = tableGSG_Asset.Close;
tablePrice.(VAR_NAMES{9}) = tableBX_Asset.Close;
tablePrice.(VAR_NAMES{10}) = tableLAZ_Asset.Close;
tablePrice.(VAR_NAMES{11}) = tableKKR_Asset.Close;
% For BTC_USD it takes only the rows with the same dates
rows2extractInBTC_USD = ismember(tableBTC_Asset.Date,tableAGG_Asset.Date);
tableBTC_Asset = tableBTC_Asset(rows2extractInBTC_USD,:);
tablePrice.(VAR_NAMES{12}) = tableBTC_Asset.Close;

%% 0. PORTFOLIO BUILD, PHASE 0 - INPUT ASSETS

% Constants
NB_ENTRIES    = height(tablePrice);
NB_STOCKS     = width(tablePrice);
NB_MARKET_CAP = width(tableMktCap);

% Initialization of data sets to use
marketCap = tableMktCap.Variables;
prices = tablePrice.Variables;

% Normailzation of weights
capWeights = marketCap/sum(marketCap);
% Find returns for the different asset class
% rets = diff(prices(:,1:NB_MARKET_CAP))./prices(1:end-1,1:NB_MARKET_CAP);
rets = tick2ret(prices);
% Risk free rate
rf = 0.0045;
mu = mean(rets);

% Performance measures of assets
annual_return  = mu * 252;
monthly_return = mu * 22 ;
annual_volatility=std(rets)*sqrt(252);

%% Part 1 Market Implied Equilibrium Return

% Covariance matrix sigma
sigma = cov(rets-rf);
% Find risk aversion lambda 
expectedRet = mu*capWeights';
lambda = (expectedRet-rf)/(capWeights*sigma*capWeights');
% Find CAPM  portofolio return (return should be same as pi)
capmRet = (expectedRet-rf)*(sigma*capWeights')/(capWeights*sigma*capWeights');
% Implied equilibrium vector
pi = lambda*sigma*capWeights';
% Recommended portofolio weight based on pi should be same as market capweights
w_pi = (lambda*sigma)\pi;

%% Part 2 incorporateing views

% Build view vector Q and vector P that matches view to assests
Q = annual_return';
p = [1,0,0,0,0,0,0,0,0,0,0,0;...
     0,1,0,0,0,0,0,0,0,0,0,0;...
     0,0,1,0,0,0,0,0,0,0,0,0;...
     0,0,0,1,0,0,0,0,0,0,0,0;...
     0,0,0,0,1,0,0,0,0,0,0,0;...
     0,0,0,0,0,1,0,0,0,0,0,0;...
     0,0,0,0,0,0,1,0,0,0,0,0;...
     0,0,0,0,0,0,0,1,0,0,0,0;...
     0,0,0,0,0,0,0,0,1,0,0,0;...
     0,0,0,0,0,0,0,0,0,1,0,0;...
     0,0,0,0,0,0,0,0,0,0,1,0;...
     0,0,0,0,0,0,0,0,0,0,0,1];
     
% Pick scaling constant tau to be 0.025;
tau = 1/NB_STOCKS;
% Build variance of view - omega matrix
omega = diag(diag(p*sigma*p'*tau));

% viewTable = array2table([p Q diag(omega)], 'VariableNames', [VAR_NAMES "View_Return" "View_Uncertainty"])
C = tau*sigma;

%% Compute the estimated mean return and covariance

mu_bl = (p'*(omega\p) + inv(C)) \ ( C\pi + p'*(omega\Q));
cov_mu = inv(p'*(omega\p) + inv(C));

table(VAR_NAMES', pi*252, mu_bl*252, 'VariableNames', ["Asset_Name", ...
    "Prior_Belief_of_Expected_Return", "Black_Litterman_Blended_Expected_Return"])

%% Portfolio Optimization Sharpe Ratio

port = Portfolio('NumAssets', NB_STOCKS, 'lb', 0, 'budget', 1, 'Name', 'Mean Variance');
port = setAssetMoments(port, mean(rets), sigma);
wts = estimateMaxSharpeRatio(port);

portBL = Portfolio('NumAssets', NB_STOCKS, 'lb', 0, 'budget', 1, 'Name', 'Mean Variance with Black-Litterman');
portBL = setAssetMoments(portBL, mu_bl, sigma + cov_mu);  
wtsBL = estimateMaxSharpeRatio(portBL);

ax1 = subplot(1,2,1);
idx = wts>0.001;
pie(ax1, wts(idx), VAR_NAMES(idx));
title(ax1, port.Name ,'Position', [-0.05, 1.6, 0]);

ax2 = subplot(1,2,2);
idx_BL = wtsBL>0.001;
pie(ax2, wtsBL(idx_BL), VAR_NAMES(idx_BL));
title(ax2, portBL.Name ,'Position', [-0.05, 1.6, 0]);

table(VAR_NAMES', wts, wtsBL, 'VariableNames', ["AssetName", "Mean_Variance", ...
     "Mean_Variance_with_Black_Litterman"])
 