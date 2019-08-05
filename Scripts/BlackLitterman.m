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
FILES_EXT   = '.xlsx';

% Raw data from file
tablePrices = readtable([DATA_FOLDER 'BlackLitterman' FILES_EXT]);
tableMktCap = readtable([DATA_FOLDER 'BlackLitterman' FILES_EXT],...
                        'Sheet','capitalization','ReadVariableNames',false);

%% 0. PORTFOLIO BUILD, PHASE 0 - INPUT ASSETS

% Constants
NB_ENTRIES    = height(tablePrices);
NB_STOCKS     = width(tablePrices);
NB_MARKET_CAP = width(tableMktCap);

% Initialization of data sets to use
marketCap = tableMktCap.Variables;
prices = tablePrices.Variables;

% Normailzation of weights
capWeights = marketCap/sum(marketCap);
% Find returns for the different asset class
rets = diff(prices(:,1:NB_MARKET_CAP))./prices(1:end-1,1:NB_MARKET_CAP);

% Risk free rate
rf = 0.0045;
mu = mean(rets);
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
% 1.US Small Value will have an absolute return of 0.3% (25% of confidence);
% 2.international bond will outperform us bound by 0.1%; (tilt away from
% international) (50% of confidence)
% 3.international Dev Equity and int'l Emerg Equity will outperform  US Large
% Growth and US small growth by 0.15%.(65% of confidence) 

% view 3, find weighted averge implied return for two set.
% for us large growth and samll growth (nominally 'underperforming' asset)
totalCap_1   = marketCap(3)+marketCap(5);
weightedPi_1 = (marketCap(3)/totalCap_1)*pi(3)+(marketCap(5)/totalCap_1)*pi(5);
% For int'l dev equity and int'l emerging equity (nominally 'outperforming' asset)
totalCap_2   = marketCap(7)+marketCap(8);
weightedPi_2 = (marketCap(7)/totalCap_2)*pi(7)+(marketCap(8)/totalCap_2)*pi(8);
weightedDifference = weightedPi_2 - weightedPi_1;

% Build view vector Q and vector P that matches view to assests
Q = [0.003;0.001;0.0015];
p = [0,0,0,0,0,1,0,0;...
    -1,1,0,0,0,0,0,0;...
     0,0,-0.9961,0,-0.0039,0,0.2872,0.7128];
% Pick scaling constant tau to be 0.025;
tau = 0.025;
% Build variance of view - omeaga matrix
omega = diag(diag(p*sigma*p'*tau));

% Calculate new(posterior) combined return vector from black-litterman model
first  = inv(tau*sigma) + p'*(omega\p);
second = (tau*sigma)\pi + p'*(omega\Q);
newReturn = first\second;
% New recommend weight of portofolio
newWeight = (lambda*sigma)\newReturn;

%% Part 3 The New Method - An intuitive Approach

% Constants
DISPLAY = false;
MAX_EVALUATIONS = 210;
STARTING_PT = 0.1;

% Step 1
ret_100 = pi + tau*sigma*p'*((p*tau*sigma*p')\(Q-p*pi));
% Step 2
w_100 = (lambda*sigma)\ret_100;
% Step 3
D_100 = w_100 - capWeights';
% Step 4
c_k  = [0.5;0.5;0.65;0;0.65;0.25;0.65;0.65];
tilt = D_100.*c_k;
% Step 5
targetWeight = capWeights' + tilt;
% Step 6
omega_k = zeros(3,3);
% Optimization parameters
if (DISPLAY)
  options = optimset('MaxFunEvals',MAX_EVALUATIONS,'OutputFcn',@plotConvergence);
else
  options = optimset('MaxFunEvals',MAX_EVALUATIONS);
end
% Optimization computation
for k=1:1:length(omega_k)
  w_k  = @(x)inv(lambda*sigma)*inv(inv(tau*sigma)+p(k,:)'*(x\p(k,:)))...
             *((tau*sigma)\pi+p(k,:)'*(x\Q(k)));
  func = @(x)(targetWeight-w_k(x))'*(targetWeight-w_k(x));
  if (DISPLAY) % Graphical representation of the convergence
    figure(k);
    title (['Convergence of parameter \Omega_' num2str(k)]);
    xlabel('Iteration #');
    ylabel('Parameter value');
    hold on;
    [omega_k(k,k),~,~,outputs(k,:)] = fminsearch(func,0.01,options);
    hold off;
  else
    omega_k(k,k) = fminsearch(func,STARTING_PT,options);
  end
end

final_ret    = (inv(tau*sigma)+p'*(omega_k\p))\((tau*sigma)\pi+p'*(omega_k\Q));
final_weight = (lambda*sigma)\final_ret;

%% Function to display the convergence of the optimization algorithm
function stop = plotConvergence(x,optimValues,~)
  stop = false;
  plot(optimValues.iteration,x,'ob');
  drawnow;
end
