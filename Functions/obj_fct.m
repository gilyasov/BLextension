function f=obj_fct(VaR,z)
alpha=.01;
h=std(z)*length(z)^(-.2);
N1=length(VaR);
N2=length(z);
f=(mean(normcdf((repmat(z,1,N1)-repmat(VaR,N2,1))/h))-alpha).^2;
