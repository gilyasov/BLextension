function p=LeastInfoKernel(Y,y,h2)

[T,N]=size(Y);
Aeq = ones(1,T);  % constrain probabilities to sum to one...
beq=1;
Aeq = [Aeq   % ...constrain the first moments...
    Y'];
beq=[beq
    y];
if ~isnan(h2)
    SecMom=h2+y*y';  % ...constrain the second moments...
    for k=1:N
        for l=k:N
            Aeq = [Aeq
                (Y(:,k).*Y(:,l))'];
            beq=[beq
                SecMom(k,l)];
        end
    end
end
p_0=ones(T,1)/T;
p = EntropyProg(p_0,[],[],Aeq ,beq); % ...compute posterior probabilities


