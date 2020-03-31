clc
clear all
warning off
n_repeat=1;
missrate=0.5;% fraction of unknown entries
rng(10)
for pp=1:n_repeat
% make data
ns=5;% number of subspaces
m=30;% row dimension
n=300;% number of data in each subspaces
X=[];
r=3;
rho=0.5;
for k=1:ns
    x=unifrnd(0,1,[r,n]);
    XT=randn(m,r)*x...
        +rho*(randn(m,r)*x.^2+randn(m,1)*[x(1,:).*x(2,:)]++randn(m,1)*[x(1,:).*x(3,:)]+randn(m,1)*[x(2,:).*x(3,:)]...
        +randn(m,r)*x.^3+randn(m,1)*[x(1,:).*x(2,:).*x(3,:)]+randn(m,1)*[x(1,:).^2.*x(2,:)]+randn(m,1)*[x(1,:).^2.*x(3,:)]...
        +randn(m,1)*[x(2,:).^2.*x(1,:)]+randn(m,1)*[x(2,:).^2.*x(3,:)]+randn(m,1)*[x(3,:).^2.*x(1,:)]+randn(m,1)*[x(3,:).^2.*x(2,:)]);
    X=[X XT];
end
% miss data
[nr,nc]=size(X);
M=ones(nr,nc);
for i=1:nc
    temp=randperm(nr,ceil(nr*missrate));% 1
    M(temp,i)=0;
end
X0=X;% original matrix
X=X.*M;% incomplete matrix
%% LRMC
% nuclear norm minimization
[Xr{1}]=LRMC_nnm(X,M); 
% factor nuclear norm minimization
[Xr{2}]=LRMC_fnnm(X,M,15,1);
%% KFMC
d=30*ns;% number of columns of D
%
ker.type='poly';ker.par=[1 2];
alpha=1;beta=1;
[Xr{3}]=KFMC(X,M,d,alpha,beta,ker);
%
ker.type='rbf';ker.par=[];ker.par_c=1;
alpha=0;beta=0.001;
[Xr{4}]=KFMC(X,M,d,alpha,beta,ker);
%%
for i=1:length(Xr)
re_error(pp,i)=norm((X0-Xr{i}).*(1-M),'fro')/norm(X0.*(1-M),'fro');
end

end

