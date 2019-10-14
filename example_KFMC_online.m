clc
clear all
warning off
n_repeat=1;
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
ker.type='poly';ker.par=[1 2];
options.online_maxiter=20; options.eta=0.5; options.X_true=X0; options.npass=20; 
[Xr{1},D,Z,J,ker,RE{1}]=KFMC_online(X,M,rr*ns,0.01,0.001,ker,options);
%
rr=30;
ker.type='rbf';ker.par=0;ker.c=3;
options.online_maxiter=20; options.eta=0.5; options.X_true=X0; options.npass=20; 
[Xr{2},D,Z,J,ker,RE{2}]=KFMC_online(X,M,rr*ns,0,0.001,ker,options);
%%
for i=1:length(Xr)
re_error(pp,i)=norm((X0-Xr{i}).*(1-M),'fro')/norm(X0.*(1-M),'fro');
end

end
figure
plot(RE{1})
hold on
plot(RE{2})
legend('Polynomial kernel','RBF kernel')
ylabel('Relative recovery error')
xlabel('Pass number')

