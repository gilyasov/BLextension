function p_ = TimeStateConditioning(X,p,ExpValue,Variance)

[J,K]=size(X);

% constrain probabilities to sum to one...
Aeq = ones(1,J);
beq=1;

% ...constrain the expectation...
V= X;
v= ExpValue;

Aeq=[Aeq
    V'];
beq=[beq
    v];

A=(X.^2)';
b=ExpValue.^2 + Variance;

% ...compute posterior probabilities
p_ = EntropyProg(p,A,b,Aeq ,beq);
