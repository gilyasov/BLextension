%% Using Fully Flexible Probabilities for Strategic Asset Allocation: 
%  an Empirical Experiment on a Global Minimum Variance Portfolio
%
% All scripts and function written by Gleb ILYASOV based on various sources
% mentionned in the associated thesis.
% -------------------------------------------------------------------------

%% Cleaning of the session
clear variables; close all; clc;

% Disable the warning: "Table variable names were modified to make them valid
warning('off','MATLAB:table:ModifiedAndSavedVarnames'); % MATLAB identifiers"
%#ok<*SAGROW>

%% Import all raw data into the workspace

% Constants
DATA_FOLDER  = '..\Data\';
CSV_EXT    = '.csv';

% Raw asset data from CSV files
tableAGG_Asset   = readtable([DATA_FOLDER    'AGG'    CSV_EXT]);
tableVBTIX_Asset = readtable([DATA_FOLDER   'VBTIX'   CSV_EXT]);
% tableNKE_Asset   = readtable([DATA_FOLDER    'NKE'    CSV_EXT]);
tableMSFT_Asset  = readtable([DATA_FOLDER    'MSFT'   CSV_EXT]);
tableVIX_Asset   = readtable([DATA_FOLDER    'VIX'    CSV_EXT]);
tableHCP_Asset   = readtable([DATA_FOLDER    'HCP'    CSV_EXT]);
% tableBTC_Asset   = readtable([DATA_FOLDER  'BTC_USD'  CSV_EXT]);
tableDLR_Asset   = readtable([DATA_FOLDER    'DLR'    CSV_EXT]);
tableDBC_Asset   = readtable([DATA_FOLDER    'DBC'    CSV_EXT]);
tableGSG_Asset   = readtable([DATA_FOLDER    'GSG'    CSV_EXT]);
tableBX_Asset    = readtable([DATA_FOLDER    'BX'     CSV_EXT]);
tableWMT_Asset   = readtable([DATA_FOLDER    'WMT'    CSV_EXT]);
tableKKR_Asset   = readtable([DATA_FOLDER    'KKR'    CSV_EXT]);
% Raw market capitalization data from XLSX files
tableMarketCap   = readtable([DATA_FOLDER 'MarketCap.xlsx']);
tableMktCap      = tableMarketCap(1,2:end);

%% Create the prices table with all stock exchange close values

% Constants
VAR_NAMES  = {  'ID'  ,  'Date'  ,  'AGG' ,  'BX'  ,  'DBC',  'DLR' ,  'GSG'  ,  'HCP' ,  'KKR' ,  'MSFT'   ,  'VBTIX' ,  'WMT' ,  'VIX'  };
VAR_TYPES  = {'uint32','datetime','double', 'double','double','double','double', 'double','double','double', 'double','double', 'double'};
NB_COLUMNS = numel(VAR_NAMES);
NB_VALUES  = NB_COLUMNS-2;
NB_ENTRIES = height(tableAGG_Asset);

% Empty table with headers only
tablePrice = table('Size',[NB_ENTRIES NB_COLUMNS],...
                   'VariableTypes',VAR_TYPES,...
                   'VariableNames',VAR_NAMES);

% Fulfill the table with the needed columns of all data tables
tablePrice.(VAR_NAMES{1})  = (1:1:NB_ENTRIES)';
tablePrice.(VAR_NAMES{2})  = tableAGG_Asset.Date;
tablePrice.(VAR_NAMES{3})  = tableAGG_Asset.Close;
tablePrice.(VAR_NAMES{4})  = tableVBTIX_Asset.Close;
% tablePrice.(VAR_NAMES{5})  = tableNKE_Asset.Close;
tablePrice.(VAR_NAMES{5})  = tableMSFT_Asset.Close;
tablePrice.(VAR_NAMES{6})  = tableHCP_Asset.Close;
tablePrice.(VAR_NAMES{7})  = tableDLR_Asset.Close;
tablePrice.(VAR_NAMES{8})  = tableDBC_Asset.Close;
tablePrice.(VAR_NAMES{9}) = tableGSG_Asset.Close;
tablePrice.(VAR_NAMES{10}) = tableBX_Asset.Close;
tablePrice.(VAR_NAMES{11}) = tableWMT_Asset.Close;
tablePrice.(VAR_NAMES{12}) = tableKKR_Asset.Close;
% For BTC_USD it takes only the rows with the same dates
% rows2extractInBTC_USD = ismember(tableBTC_Asset.Date,tableAGG_Asset.Date);
% tableBTC_Asset = tableBTC_Asset(rows2extractInBTC_USD,:);
% tablePrice.(VAR_NAMES{14}) = tableBTC_Asset.Close;
tablePrice.(VAR_NAMES{13})  = tableVIX_Asset.Close;

%% Processing of data

% Constants
TAU = 1;         % Daily data if 1 or weekly if 5
NB_STOCKS = 10;  % Number of selected stocks
WINDOW = 252*3;  % size in days of the rolling window (=±1y)

% Date vector
tauDates = tablePrice.Date(1:TAU:end);
% Stocks prices matrices
prices = [tablePrice.AGG tablePrice.BX tablePrice.DBC tablePrice.DLR   ...
          tablePrice.GSG tablePrice.HCP  tablePrice.KKR  tablePrice.MSFT   ...
          tablePrice.VBTIX tablePrice.WMT   tablePrice.VIX];
returns = tick2ret(prices(1:TAU:end,1:NB_STOCKS));
% Risk drivers for the stocks under consideration
riskDrivers = log(prices);
% Invariants (i.i.d. variables) from the timeseries analysis of the risk drivers (log-values)
invariants = diff(riskDrivers);

% Here we switch to weekly data. Rationale vs. monthly or yearly :
% Eliminates some of the noise from the daily prices without oversmoothing
% tau-days empirical observations:
tauPrices = prices(1:TAU:end,1:NB_STOCKS);
tauInvariants = diff(log(tauPrices));
tauVIX = prices(1:TAU:end,NB_STOCKS+1);
invariantsVIX = diff(log(tauVIX));

% Recover the projected scenarios for the risk drivers at the tau-day horizon
projRiskDrivers = log(tauPrices((1:end-1),:)) + tauInvariants;
projPrices = exp(projRiskDrivers);
PnL = projPrices - tauPrices((1:end-1),:); % ex-ante P&L of each stock

% Aggregation of the individual stock P&Ls into projected portfolio P&L for all scenarios
% Assume equally weighted protfolio at beginning of investment period
[J,N] = size(invariants(:,1:NB_STOCKS));
selectedStocks = (1:1:NB_STOCKS)';
% Initial holdings (# of shares)
eqw = zeros(N,1);
eqw(selectedStocks) = 1/length(selectedStocks);
h = eqw./prices(end,1:NB_STOCKS)'; % quantity of each stock held in the portfolio
agg_PnL = PnL*h;                   % portfolio aggregated P&L

%% GLOBAL MINIMUM VARIANCE TWO-STEP OPTIMISATION 
% Step 0: data prep
% Historical prices and invariants for non-rolling optimisation (January 1999 - December 2003)
optInvariants = tauInvariants(1:WINDOW,:);
optPrices = tauPrices(WINDOW,1:NB_STOCKS);
optWindow = tauDates(1:WINDOW);
optVIX = tauVIX(1:WINDOW);
Options.NumPortf = 40;            % number of portfolios forming the efficient frontier
Options.FrontierSpan = [.05 .95]; % range of normalized expected values spanned by efficient frontier

% ==============================
% NO Rolling Window + NO Entropy
% ==============================
% Step 1: MV quadratic optimization to determine one-parameter frontier of quasi-optimal solutions
eq_p = ones([length(optInvariants),1])/length(optInvariants); % equal probabilities
[e1,s1,w1,M1,S1] = EfficientFrontier(optInvariants,eq_p,Options);

% % Step 2: Exact Satisfaction Computation (Case of a "Minimum Variance Portfolio")
% satisfaction1 = -s1;
% % Choose the allocation that maximises satisfaction
% [maxSatValue1,maxSatIndex1] = max(satisfaction1);
% optimalAllocation1 = w1(maxSatIndex1,:);

% ==============================
% NO Rolling Window + Entropy
% ==============================

% 3 step (Prior, Views, Posterior)
% Processing of the views - Conditiooptning framework
t = size(optInvariants,1);

% Time Crisp Conditioning
rw_p = zeros(t,1);
%timeWindow = 2*52;
timeWindow = WINDOW*0.5;
rw_p(end-timeWindow:end)=1; % more weight on the end of the sample than the start
%rw_p(1:tau)=1;             % opposite
rw_p = rw_p/sum(rw_p);      % re-normalising of the probabilities

% Exponential Smoothing Conditioning
%halfLife = 2*52;           % we test for a half-life of 6 months and 2 years
halfLife = 0.5*WINDOW;      % decrease to have a steeper curve
lambda = log(2)/halfLife;
t_p = exp(-lambda*(t-(1:t)'));
t_p = t_p/sum(t_p); 

% State Crisp Conditioning
s_p = zeros(t,1);
levelVIX = 20;              % change here to 15; 25 to replicate results in Appendix B
condVIX = tauVIX(1:WINDOW); % based on expectation of VIX above or below [X] level
cond = condVIX >= levelVIX; % change here to >=20; >=15; >=25 to replicate results in Section 6
s_p(cond)=1;
s_p = s_p/sum(s_p);  

% Kernel Smoothing Conditioning
k_p = zeros(t,1);
cond = condVIX >= 20;    % change here to >=20; >=15; >=25 to replicate results in Section 6
h2 = cov(diff(condVIX));
%absdiff = condVIX-levelVIX; % when VIX lower than [X]
absdiff = condVIX - levelVIX;   % when VIX higher than [X]
k_p = mvnpdf(absdiff,2,h2);
k_p = k_p/sum(k_p);

% % Kernel Smoothing Conditioning
% kernel = diag(cond)*condVIX
% k_p = fitdist(kernel,'Kernel','Kernel','epanechnikov')
% % % f = mvksdensity(kernel,cond,'Bandwidth',0.8)
% % absdiff= condVIX - 10;
% % h2 = cov(tick2ret(condVIX));
% % k_p = mvnpdf(absdiff,levelVIX,h2);
% % k_p = k_p/sum(k_p);

% Joint time and state conditioning through minimum relative entropy

% Time exponentially and state crisp

% 1. specify view on expectation and standard deviation
pPrior = t_p; % set the prior to exponentially decaying conditioned probabilities
expvalue = sum(s_p.*condVIX);
variance = sum(s_p.*condVIX.*condVIX)-expvalue.^2;
% 2. posterior market distribution using minimum relative entropy
pPost = TimeStateConditioning(condVIX,pPrior,expvalue,variance);
% % 2-step optimisation (see above for explanations)
% [e2,s2,w2,M2,S2] = EfficientFrontier(optInvariants,pPost,Options);
% satisfaction2 = -s2;
% [maxSatValue2,maxSatIndex2] = max(satisfaction2);
% optimalAllocation2 = w2(maxSatIndex2,:);

% Minimum variance or mean variance allocation construction
% p = pPost;
% X = optInvariants;
% Exps = X'*p;
% Scnd_Mom = X'*(X.*(p*ones(1,NB_STOCKS))); 
% Scnd_Mom = (Scnd_Mom+Scnd_Mom')/2;
% Covs = Scnd_Mom-Exps*Exps';
% f = zeros (NB_STOCKS, 1);
% Aeq = ones(1,NB_STOCKS); % positive weights
% beq = 1;
% Aleq = [-Exps'   ;  -eye(NB_STOCKS)    ];  % weights sum to one
% bleq = [-0.05 ;  zeros(NB_STOCKS,1)];
% % Minimisation problem
% MinVol_Weights = quadprog(Covs,f,Aleq,bleq,Aeq,beq);
% Time exponentially and kernel state

pPrior2 = t_p; % set the prior to exponentially decaying conditioned probabilities
expvalue2 = sum(k_p.*condVIX);
variance2 = sum(k_p.*condVIX.*condVIX)-expvalue2.^2;
% frontier in terms of number of shares
pPost2 = TimeStateConditioning(condVIX,pPrior2,expvalue2,variance2);
% 2-step optimisation (see above for explanations)
[e22,s22,w22,M22,S22] = EfficientFrontier(optInvariants,pPost2,Options);
satisfaction22 = -s22;
[maxSatValue22,maxSatIndex22] = max(satisfaction22);
optimalAllocation22 = w22(maxSatIndex22,:);

figure (1)
subplot(3,1,1)
plot(optWindow,agg_PnL(1:756));
title('Aggregated P&L', 'fontsize', 14);
ylabel('Returns', 'fontsize', 12);
grid on
set(gca,'xlim',[min(optWindow) max(optWindow)])
datetick('x','mmmyy','keeplimits','keepticks');

subplot(3,1,2)
area(optWindow,rw_p);
ylabel('Probability', 'fontsize', 12);
title('Crisp Time Conditioning', 'fontsize', 14);
grid on
set(gca,'xlim',[min(optWindow) max(optWindow)])
datetick('x','mmmyy','keeplimits','keepticks');

subplot(3,1,3)
area(optWindow,t_p);
title('Exponential Smoothing conditioning', 'fontsize', 14);
ylabel('Probability', 'fontsize', 12);
grid on
set(gca,'xlim',[min(optWindow) max(optWindow)])
datetick('x','mmmyy','keeplimits','keepticks');
%% BLACK LITTERMAN PART
%% 0. PORTFOLIO BUILD, PHASE 0 - INPUT ASSETS

% Initialization of data sets to use
marketCap = tableMktCap.Variables;
pricesBL = tauPrices(:,1:NB_STOCKS);

% Normailzation of weights
capWeights = marketCap/sum(marketCap);
% Find returns for the different asset class
% rets = diff(prices(:,1:NB_MARKET_CAP))./prices(1:end-1,1:NB_MARKET_CAP);
rets = tick2ret(pricesBL);
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
p = [1,0,0,0,0,0,0,0,0,0;...
     0,1,0,0,0,0,0,0,0,0;...
     0,0,1,0,0,0,0,0,0,0;...
     0,0,0,1,0,0,0,0,0,0;...
     0,0,0,0,1,0,0,0,0,0;...
     0,0,0,0,0,1,0,0,0,0;...
     0,0,0,0,0,0,1,0,0,0;...
     0,0,0,0,0,0,0,1,0,0;...
     0,0,0,0,0,0,0,0,1,0;...
     0,0,0,0,0,0,0,0,0,1];
     
% Pick scaling constant tau to be 0.025;
tau = 1/NB_STOCKS;
% Build variance of view - omega matrix
omega = diag(diag(p*sigma*p'*tau));
% Q = Q/252; 
% omega = omega/252;
% viewTable = array2table([p Q diag(omega)], 'VariableNames', [VAR_NAMES "View_Return" "View_Uncertainty"])
C = tau*sigma;

%% Compute the estimated mean return and covariance

mu_bl = (p'*(omega\p) + inv(C)) \ ( C\pi + p'*(omega\Q));
cov_mu = inv(p'*(omega\p) + inv(C));

table(VAR_NAMES(3:NB_STOCKS+2)', pi*252, mu_bl*252, 'VariableNames', ["Asset_Name", ...
    "Prior_Belief_of_Expected_Return", "Black_Litterman_Blended_Expected_Return"])

%% Portfolio Optimization Sharpe Ratio

port = Portfolio('NumAssets', NB_STOCKS, 'lb', 0, 'budget', 1, 'Name', 'Mean Variance');
port = setAssetMoments(port, mean(rets), sigma);
wts = estimateMaxSharpeRatio(port);

portBL = Portfolio('NumAssets', NB_STOCKS, 'lb', 0, 'budget', 1, 'Name', 'Mean Variance with Black-Litterman');
portBL = setAssetMoments(portBL, mu_bl, sigma + cov_mu);  
w_SH = estimateMaxSharpeRatio(portBL);

% % Minimum Variance Portfolio
% iota=ones(size(sigma,1),1);
% w_GMV=(inv(sigma)*iota)/(iota'*inv(sigma)*iota);
% 
% % Minimum Variance with BL expected returns
% teta = sigma + cov_mu;
% iota=ones(size(teta,1),1);
% w_GMVBL=(inv(teta)*iota)/(iota'*inv(teta)*iota);

% GLOBAL Minimum Variance with BL expected returns
Covs = sigma + cov_mu;
f = zeros (NB_STOCKS, 1);
Aeq = ones(1,NB_STOCKS); % positive weights
beq = 1;
Aleq = [eye(NB_STOCKS)    ;  -eye(NB_STOCKS)   ];  % weights sum to one
bleq = [ones(NB_STOCKS,1) ;  zeros(NB_STOCKS,1)];
% Minimisation problem
w_GMVBL = quadprog(Covs,f,Aleq,bleq,Aeq,beq);

% Basic Mean-Variance framework 
H = sigma;
f = zeros (NB_STOCKS, 1);
A = [-mu; -eye(NB_STOCKS)];
b = [0; zeros(NB_STOCKS,1)];
Aeq = ones (1,NB_STOCKS);
beq = 1;
lb = zeros(NB_STOCKS);
ub = ones(NB_STOCKS);
wmv = quadprog(H,f,A,b,Aeq,beq,lb,ub);

% Mean-Variance framework on BL
H_BL = sigma + cov_mu;
f_BL = zeros (NB_STOCKS, 1);
A_BL = [-mu_bl'; -eye(NB_STOCKS)];
b_BL = [-0.05; zeros(NB_STOCKS,1)];
Aeq = ones (1,NB_STOCKS);
beq = 1;
wmvBL = quadprog(H_BL,f_BL,A_BL,b_BL,Aeq,beq,lb,ub);

% VAR_NAMES = VAR_NAMES(3:end-1);
% figure
% ax1 = subplot(1,2,1);
% idx = wts>0.001;
% pie(ax1, wts(idx), VAR_NAMES(idx));
% title(ax1, port.Name ,'Position', [-0.05, 1.6, 0]);

% ax2 = subplot(1,2,2);
% idx_BL = wtsBL>0.001;
% pie(ax2, wtsBL(idx_BL), VAR_NAMES(idx_BL));
% title(ax2, portBL.Name ,'Position', [-0.05, 1.6, 0]);

%% ROLLING WINDOW

% Inputs for rolling window
ss = datefind(tauDates(1),tauDates);
se = datefind(tauDates(end),tauDates);
LastRoll = se-ss+1-WINDOW; % last rolling window

% Inputs for backtest when using a rolling window
b_start = tauDates(WINDOW);
b_end = tauDates(end);
bs = datefind(b_start,tauDates);
be = datefind(b_end,tauDates)-1;
b_window = 252/2 ;

% Initialisation
k = 0; pr3 = []; pr4 = []; pr5 = []; p_post_rr = []; 
% p_post_rr2 = [];
RollingReturns = nan(WINDOW,NB_STOCKS,11);
RollingPrices  = nan(NB_STOCKS,11);
% Rolling Window loop
for j = 1:b_window:LastRoll-b_window
    k = k+1;
    fprintf(1,'Calculating Rolling Bootstrapped Portfolio: Month');
    fprintf(1,'%i',k);
    
    ssRolling = ss+j-1;  % Rolling Sample Start (First Raw)
    seRolling = ssRolling+WINDOW-1; % Sample End (Last Raw) 
    ObsSampleRolling = seRolling-ssRolling+1; % Number of observations within Rolling Sample

    RollingReturns(:,:,k) = tauInvariants((ssRolling:seRolling),:);
    RollingPrices(:,k)    = tauPrices(seRolling,:);
    Options.CurrentPrices = RollingPrices(:,k)';

    % Rolling Window + NO Entropy
    % 2-step optimisation (see above for explanations)
    [e3(:,k),s3(:,:,k),w3(:,:,k),M3(:,k),S3(:,:,k)] = ...
    EfficientFrontier(RollingReturns(:,:,k),eq_p,Options);
    satisfaction3(:,:,k) = -s3(:,:,k);
    [maxSatValue3(:,:,k), maxSatIndex3(:,:,k)] = max(satisfaction3(:,:,k));
    optimalAllocation3(:,:,k) = w3(maxSatIndex3(:,:,k),:,k);
    
    bsRolling = bs+j-1;
    beRolling = bsRolling + b_window - 4;
    ObsOutSampleRolling = beRolling - bsRolling;
    
    % Backtest without Entropy
    rolling_ret(:,:,k) = tauInvariants((bsRolling:beRolling),:);
    pr3(:,:,k)=rolling_ret(:,k)*optimalAllocation3(:,k)';
    
    % ------------------------------
    % Rolling Window + Entropy
    % ==============================
    
    % t_p does not need to be made rolling but s_p does. 
    % "Crisp" Macro-economic conditioning
    s_pr=zeros(t,1);
    cond_VIXr=tauVIX(ssRolling:seRolling);
    condr=cond_VIXr<=20;
    s_pr(condr)=1;
    s_pr=s_pr/sum(s_pr);     
    
    % Kernel Smoothing
    k_pr=zeros(t,1);
    h2r=cov(diff(cond_VIXr));
    %absdiffr=cond_VIXr-levelVIX;        % when VIX higher than X
    absdiffr=levelVIX-cond_VIXr;         % when VIX lower than X
    k_pr=mvnpdf(absdiffr,levelVIX,h2r);
    k_pr=k_pr/sum(k_pr);

    % Time & State conditioning via Entropy Pooling
    p_prior_r=t_p;
    ExpValue_r=sum(s_pr.*cond_VIXr);
    Variance_r=sum(s_pr.*cond_VIXr.*cond_VIXr)-ExpValue_r.^2;
    % posterior market distribution using the Entropy Pooling approach
    p_post_r=TimeStateConditioning(cond_VIXr,p_prior_r,ExpValue_r,Variance_r);
    p_post_rr(:,:,k)=TimeStateConditioning(cond_VIXr,p_prior_r, ExpValue_r, Variance_r);
    
    p = p_post_r;
    X = RollingReturns(:,:,k);
    Exps = X'*p;
    Scnd_Mom = X'*(X.*(p*ones(1,NB_STOCKS))); 
    Scnd_Mom = (Scnd_Mom+Scnd_Mom')/2;
    Covs = Scnd_Mom-Exps*Exps';
    f = zeros (NB_STOCKS, 1);
    Aeq = ones(1,NB_STOCKS); % positive weights
    beq = 1;
    Aleq = [-Exps'   ;  -eye(NB_STOCKS)    ];  % weights sum to one
    bleq = [0 ;  zeros(NB_STOCKS,1)];
    lb = zeros(NB_STOCKS);
    ub = ones(NB_STOCKS);
     %Minimisation problem
    stateMV(:,:,k) = quadprog(Covs,f,Aleq,bleq,Aeq,beq,lb,ub);
    
    port(:,:,k) = Portfolio('NumAssets', NB_STOCKS, 'lb', 0, 'budget', 1, 'Name', 'Sharpe');
    port(:,:,k) = setAssetMoments(port(:,:,k), Exps',Covs);
    stateSH(:,:,k) = estimateMaxSharpeRatio(port(:,:,k));
    
    % 2-step optimisation (see above for explanations)
    [e4(:,k),s4(:,:,k),w4(:,:,k),M4(:,k),S4(:,:,k)]=EfficientFrontier(RollingReturns(:,:,k),p_post_r,Options);
    satisfaction4(:,:,k)=-s4(:,:,k);
    [max_sat_value4(:,:,k), max_sat_index4(:,:,k)]=max(satisfaction4(:,:,k));
    optimalAllocation4(:,:,k)=w4(max_sat_index4(:,:,k),:, k);
    
    % Backtest with Entropy
    pr4(:,:,k)=rolling_ret(:,:,k)*optimalAllocation4(:,:,k)';
    pr6(:,:,k)=rolling_ret(:,:,k)*stateMV(:,:,k);
    pr8(:,:,k)=rolling_ret(:,:,k)*stateSH(:,:,k);
    
    % rw_p & t_p do not change
    p_prior_r2=t_p;
    ExpValue_r2=sum(k_pr.*cond_VIXr);
    Variance_r2=sum(k_pr.*cond_VIXr.*cond_VIXr)-ExpValue_r2.^2;
    % posterior market distribution using the Entropy Pooling approach
    p_post_r2=TimeStateConditioning(cond_VIXr,p_prior_r2,ExpValue_r2,Variance_r2);
    p_post_rr2(:,:,k)=TimeStateConditioning(cond_VIXr,p_prior_r2, ExpValue_r2, Variance_r2);
    
    [e5(:,k),s5(:,:,k),w5(:,:,k),M5(:,k),S5(:,:,k)]=EfficientFrontier(RollingReturns(:,:,k),p_post_r2,Options);
    satisfaction5(:,:,k)=-s5(:,:,k);
    [max_sat_value5(:,:,k), max_sat_index5(:,:,k)]=max(satisfaction5(:,:,k));
    optimalAllocation5(:,:,k)=w5(max_sat_index5(:,:,k),:, k);
    
    p = p_post_r2;
    X = RollingReturns(:,:,k);
    Exps = X'*p;
    Scnd_Mom = X'*(X.*(p*ones(1,NB_STOCKS))); 
    Scnd_Mom = (Scnd_Mom+Scnd_Mom')/2;
    Covs = Scnd_Mom-Exps*Exps';
    f = zeros (NB_STOCKS, 1);
    Aeq = ones(1,NB_STOCKS); % positive weights
    beq = 1;
    Aleq = [-Exps'   ;  -eye(NB_STOCKS)    ];  % weights sum to one
    bleq = [0 ;  zeros(NB_STOCKS,1)];
    lb = zeros(NB_STOCKS);
    ub = ones(NB_STOCKS);
     %Minimisation problem
    kernelMV(:,:,k) = quadprog(Covs,f,Aleq,bleq,Aeq,beq,lb,ub);
    
    port1(:,:,k) = Portfolio('NumAssets', NB_STOCKS, 'lb', 0, 'budget', 1, 'Name', 'Sharpe');
    port1(:,:,k) = setAssetMoments(port1(:,:,k), Exps',Covs);
    kernelSH(:,:,k) = estimateMaxSharpeRatio(port1(:,:,k));
        % Backtest with Entropy
    pr5(:,:,k)=rolling_ret(:,:,k)*optimalAllocation5(:,:,k)';
    pr7(:,:,k)=rolling_ret(:,:,k)*kernelMV(:,:,k);
    pr9(:,:,k)=rolling_ret(:,:,k)*kernelSH(:,:,k);
end
a = tick2ret(prices);
b = cov(a);
a = corrcov(b);
aaa = mean(a(1:10,11));
% BACKTESTING 
% -------------------------------------------------------------------------
portfolio_returns3 = num2cell(pr3, [1,2]);
portfolio_returns3 = vertcat(portfolio_returns3{:});
rbsend = size (portfolio_returns3,1);
bsend = rbsend + WINDOW - 1 ;
% Backtesting of a simple buy and hold strategy

% Historical PnL for Backtest (Jan 2004 - December 2016)
BT_invariants = tauInvariants(WINDOW:bsend,1:NB_STOCKS);
BT_dates = tauDates(WINDOW:bsend+1);

% Backtest with Equal Weights (no flex probs)
portfolio_returns0=BT_invariants*eqw;
portfolio_profits0=10*[1;exp(cumsum(portfolio_returns0))];

% Benchmark Market weights
portfolio_returns1=BT_invariants*capWeights';
portfolio_profits1=10*[1;exp(cumsum(portfolio_returns1))];

% BL with mean variance
portfolio_returnsA=BT_invariants*wmvBL;
portfolio_profitsA=10*[1;exp(cumsum(portfolio_returnsA))];

% BL with minimum Vairance
portfolio_returnsB=BT_invariants*w_GMVBL;
portfolio_profitsB=10*[1;exp(cumsum(portfolio_returnsB))];

% BL with Sharpe Ratio
portfolio_returnsC=BT_invariants*w_SH;
portfolio_profitsC=10*[1;exp(cumsum(portfolio_returnsC))];

% Backtesting of a buy and hold strategy with a Walk Forward Rolling Window
% No entropy profits - Rolling
portfolio_returns3 = num2cell(pr3, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns3 = vertcat(portfolio_returns3{:});
portfolio_returns3 = portfolio_returns3(1:rbsend);
portfolio_profits3 = 10*[1;exp(cumsum(portfolio_returns3))];

% Rolling Exp Time and State
portfolio_returns4 = num2cell(pr4, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns4 = vertcat(portfolio_returns4{:});
portfolio_returns4 = portfolio_returns4(1:rbsend);
portfolio_profits4 = 10*[1;exp(cumsum(portfolio_returns4))];

% Rolling Exp Time and Kernel
portfolio_returns5 = num2cell(pr5, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns5 = vertcat(portfolio_returns5{:});
portfolio_returns5 = portfolio_returns5(1:rbsend);
portfolio_profits5 = 10*[1;exp(cumsum(portfolio_returns5))];

% Rolling Exp State Mean Variance
portfolio_returns6 = num2cell(pr6, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns6 = vertcat(portfolio_returns6{:});
portfolio_returns6 = portfolio_returns6(1:rbsend);
portfolio_profits6 = 10*[1;exp(cumsum(portfolio_returns6))];

% Rolling Exp Kernel Mean Variance
portfolio_returns7 = num2cell(pr7, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns7 = vertcat(portfolio_returns7{:});
portfolio_returns7 = portfolio_returns7(1:rbsend);
portfolio_profits7 = 10*[1;exp(cumsum(portfolio_returns7))];

% Rolling Exp Kernel Mean Variance
portfolio_returns8 = num2cell(pr8, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns8 = vertcat(portfolio_returns8{:});
portfolio_returns8 = portfolio_returns8(1:rbsend);
portfolio_profits8 = 10*[1;exp(cumsum(portfolio_returns8))];

% Rolling Exp Kernel Mean Variance
portfolio_returns9 = num2cell(pr9, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns9 = vertcat(portfolio_returns9{:});
portfolio_returns9 = portfolio_returns9(1:rbsend);
portfolio_profits9 = 10*[1;exp(cumsum(portfolio_returns9))];

% % Views Performance
figure (2)
plot(BT_dates, portfolio_profits0, BT_dates, portfolio_profits1, ...
     BT_dates, portfolio_profitsA, BT_dates, portfolio_profitsB, BT_dates, portfolio_profitsC, ...
     BT_dates, portfolio_profits4, BT_dates, portfolio_profits5,...
     BT_dates, portfolio_profits6, BT_dates, portfolio_profits7,...
     BT_dates, portfolio_profits8, BT_dates, portfolio_profits9, 'LineWidth',1.5);
set(gca,'xlim',[tauDates(WINDOW) tauDates(bsend)]);
ylim([7.5 15]);
% set(gca, 'fontsize', 35);
datetick('x','mmmyy','keeplimits','keepticks');
lgd =legend( '1/N'  , 'MktW', ...
        'BL-MV', 'BL-MinV','BL-ShR', ...
        'Sta-MinV', 'Ker-MinV', ...
        'StateMV', 'KernelMV', 'StateSH','Kernel SH','location', 'northwest');
set(lgd, 'Fontsize', 14);
% title('Performances on a Multi-asset portfolio', 'fontsize', 15);
xlabel('Year', 'fontsize', 15);
ylabel('Portfolio Balance (in Thousands)', 'fontsize', 15);
grid on


%% EX-POST ANALYSIS - Portfolio Performance Measures
% -------------------------------------------------------------------------

portfolio_returns = [portfolio_returns0, portfolio_returns1, ...
                     portfolio_returnsA, portfolio_returnsB, portfolio_returnsC,  ...
                     portfolio_returns4, portfolio_returns5,portfolio_returns6, ...
                     portfolio_returns7,portfolio_returns8, portfolio_returns9];
                 
                 
portfolio_profits = [portfolio_profits0, portfolio_profits1,  ...
                     portfolio_profitsA, portfolio_profitsB, portfolio_profitsC, ...
                     portfolio_profits4, portfolio_profits5, portfolio_profits6, ...
                     portfolio_profits7, portfolio_profits8, portfolio_profits9];


total_return=Total_Return(portfolio_profits);            % Total Return
annual_return=mean(portfolio_returns)*252;               % Average Annual Return
% monthly_return=mean(portfolio_returns)*22;               % Average Monthly Return
annual_volatility=std(portfolio_returns)*sqrt(252);      % Annual Volatility
[csortino,cinforatio] = Ratios(portfolio_returns,0.045);
MAR = 0.0045;
on = lpm(-portfolio_returns*12,-MAR,1);                  %Omega numerator
od = lpm(-portfolio_returns*12, MAR,1);                  %Omega denominator
om = on./od; %Omega ratio
max_dd=maxdrawdown(portfolio_profits);                   % Max Drawdown
% positive_trades=Positive_Trades(portfolio_returns);    % Positive Trades

% Historical Value at Risk & Expected Shortfall
confidence_level = 0.99;
figure(3)
[VaR0, ES0] = HistVaR_ES(portfolio_returns0,confidence_level);
[VaR1, ES1] = HistVaR_ES(portfolio_returns1,confidence_level);
[VaR2, ES2] = HistVaR_ES(portfolio_returnsA,confidence_level);
[VaR3, ES3] = HistVaR_ES(portfolio_returnsB,confidence_level);
[VaR4, ES4] = HistVaR_ES(portfolio_returnsC,confidence_level);
[VaR5, ES5] = HistVaR_ES(portfolio_returns4,confidence_level);
[VaR6, ES6] = HistVaR_ES(portfolio_returns5,confidence_level);
[VaR7, ES7] = HistVaR_ES(portfolio_returns6,confidence_level);
[VaR8, ES8] = HistVaR_ES(portfolio_returns7,confidence_level);
[VaR9, ES9] = HistVaR_ES(portfolio_returns8,confidence_level);
[VaR10, ES10] = HistVaR_ES(portfolio_returns9,confidence_level);
VaR = [-VaR0 , -VaR1 , -VaR2 , -VaR3 , -VaR4 , -VaR5 , -VaR6 , -VaR7 , -VaR8 , -VaR9 , -VaR10 ];
ES  = [-ES0  , -ES1  , -ES2  , -ES3  , -ES4  , -ES5  , -ES6  , -ES7  , -ES8  , -ES9  , -ES10 ];

% Various other figures
% ================================================

% prot=num2cell(p_post_rr, [1,2]);
% prot=vertcat(prot{:});
% prot2=prot(1:5:end);
% figure (4)
% area(prot2)
% 
% figure (5)
% PlotFrontier(e2,s2,w2)

% figure
% plot(tauDates, tauVIX)
% xlim([min(tauDates) max(tauDates)]);
% datetick('x','mmmyy','keeplimits','keepticks');

% figure
% histogram(portfolio_returns);
