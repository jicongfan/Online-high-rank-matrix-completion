clc
clear all
warning off
n_repeat=10;
missrate=0.5;% fraction of unknown entries
for pp=1:n_repeat
% make data
ns=1;% number of subspaces
m=30;% row dimension
n=200;% number of data in each subspaces
X=[];
r=3;
rho=1;
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
%% KFMC
rr=30;% number of columns of D
%
ker.type='poly';ker.par=[0 2];options.offline_maxiter=1000;options.eta=0.5;
[Xr{1},Dt{1},Z,J{1},ker]=KFMC(X,M,rr*ns,0.1,0.01,ker,options);
%
rr=30;
ker.type='rbf';ker.par=0;ker.c=3;options.offline_maxiter=500;options.eta=0.5;
[Xr{2},Dt{2},Z,J{2},ker]=KFMC(X,M,rr*ns,0,0.0001,ker,options);
%%
for i=1:length(Xr)
re_error(pp,i)=norm((X0-Xr{i}).*(1-M),'fro')/norm(X0.*(1-M),'fro');
end

end

