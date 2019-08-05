function [e,s,w,Exps,Covs] = EfficientFrontier(X,p, Options)
% This function returns the NumPortf x 1 vector expected returns,
%                       the NumPortf x 1 vector of volatilities and
%                       the NumPortf x N matrix of compositions
% of NumPortf efficient portfolios whos returns are equally spaced along the whole range of the efficient frontier
% -------------------------------------------------------------------------

[~,N]=size(X);

Exps = X'*p;
Scnd_Mom = X'*(X.*(p*ones(1,N))); 
Scnd_Mom = (Scnd_Mom+Scnd_Mom')/2;
Covs = Scnd_Mom-Exps*Exps';

% 1. Positive Weights
Aeq = ones(1,N);
beq = 1;

% 2. Weights sum to 1
Aleq = [eye(N)
    -eye(N)];
bleq = [ones(N,1)
    0*ones(N,1)];


% --------------------------------------------------------- %
% === determine exp value of minimum-variance portfolio === %   
% --------------------------------------------------------- %

FirstDegree = zeros(N,1);
SecondDegree = Covs;

options = optimset('Display','off');
MinVol_Weights = quadprog(SecondDegree,FirstDegree,Aleq,bleq,Aeq,beq,[],[],[],options);
MinSDev_Exp = MinVol_Weights'*Exps;

% --------------------------------------------------------------- %
% === determine exp value of maximum-expected value portfolio === %   
% --------------------------------------------------------------- %

FirstDegree = -Exps;
MaxRet_Weights = linprog(FirstDegree,Aleq,bleq,Aeq,beq);
MaxExp_Exp = MaxRet_Weights'*Exps;

% ------------------------------------------------------------------------------------------------------ %
% === slice efficient frontier in NumPortf equally thick horizontal sectors in the upper branch only === %   
% ------------------------------------------------------------------------------------------------------ %

Grid = [Options.FrontierSpan(1) : (Options.FrontierSpan(end)-Options.FrontierSpan(1))/(Options.NumPortf-1) : Options.FrontierSpan(end)];
Targets = MinSDev_Exp + Grid*(MaxExp_Exp-MinSDev_Exp);

% --------------------------------------------------------------------- %
% === compute the NumPortf compositions and risk-return coordinates === %   
% --------------------------------------------------------------------- %

e=[];
s=[];
w=[];
FirstDegree = zeros(N,1);

for i=1:Options.NumPortf

    % determine least risky portfolio for given expected return
    AEq=[Aeq
        Exps'];
    bEq=[beq
        Targets(i)];
    Weights = quadprog(SecondDegree,FirstDegree,Aleq,bleq,AEq,bEq,[],[],[],options)';
    w=[w
        Weights];
    s=[s
        sqrt(Weights*Covs*Weights')];
    e=[e
        Weights*Exps];
end