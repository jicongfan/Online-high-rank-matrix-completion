function [X,D,Z,ker,options]=KFMC_new(X,M,d,alpha,beta,ker,options)
% This is the codes of offline version of KFMC in the following paper:
% Jicong Fan, Madeleine Udell. Online High-Rank Matrix Completion. CVPR 2019.
% Input 
%       X: the incomplete matrix (mxn)
%       M: binary matrix (mxn), 0 for missing entries, 1 for observed entries
%       d: the dimension of D
%       alpha: regularization parameter of D;
%           for RBF kernel, alpha has no effect on the method
%       beta: regularization parameter of Z
%       ker: kernel type and parameter
%           ker.type='rbf' or 'poly'; 
%           for gaussian rbf kernel,
%            ker.par=sigma2; if ker.par=[], estimate it with
%            ker.par_c; ker.par_c is often set as 1, 2, or 3 (1 default)
%           for poly kernel, 
%            ker.par=[c d], c: the constant; d: the degree of polynomial
%            if ker.par=[], automatically estimate c
%       options.gamma: 1.1 default, no need to tune in practice
%       options.eta: the momentum paprameter, 0.5 default
%       options.maxiter: 500 default
%       options.tolX: relative change of X, stopping criteria
% Written by Jicong Fan, 09/2019 (updated 03/2020), E-mail: jf577@cornell.edu
[m,n]=size(X);
if nargin<7
    options.gamma=1.1;
    options.eta=0.5;
    options.maxiter=500;
    options.tolX=1e-5;
end
if nargin<6
    ker.type='rbf';
    ker.par=[];
end
if nargin<5
    beta=0.001;
end
if nargin<4
    alpha=0;
end
if nargin<3
    d=m;
end
if nargin<2
   disp('Require the mask matrix M to indicate the missing entries!')
   D=[];Z=[];
   return      
end
D=randn(m,d);
Z=zeros(d,n);
X0=X;
% kernel
if strcmp(ker.type,'rbf') && isempty(ker.par)
    if n<8000
        Xs=X;
    else    
        Xs=X(:,randperm(n,8000));
        disp('Too large "n" ! For acceleration, do not compute the objective function!')
    end
    XX=sum(Xs.*Xs,1);
    dist=repmat(XX,size(Xs,2),1) + repmat(XX',1,size(Xs,2)) - 2*Xs'*Xs;
    if ~isfield(ker,'par_c')
        ker.par_c=1;
    end
    ker.par=(mean(real(dist(:).^0.5))*ker.par_c)^2;% sigma^2
    clear XX Xs dist
end
if strcmp(ker.type,'poly') && isempty(ker.par)
    temp=mean(sum(X.^2));
    disp(['The estimated constant for polynomial kernel is ' num2str(temp)])
    ker.par=[temp 2];% [c d]
end
% monentum
if ~isfield(options,'eta')
    options.eta=0.5;
end
eta=options.eta;
if ~isfield(options,'gamma')
    options.gamma=1.1;
end
gamma=options.gamma;
vD=zeros(size(D));
vX=zeros(size(X));
% iter
if ~isfield(options,'maxiter')
    options.maxiter=500;
end
maxiter=options.maxiter;
if ~isfield(options,'tolX')
    options.tolX=1e-5;
end
e=options.tolX;
iter=0;
disp(['kernel type: ' ker.type ' kernel parameter(s):' num2str(ker.par) ' alpha=' num2str(alpha) ' beta=' num2str(beta) ' momentum_eta=' num2str(eta)]) 
while iter<maxiter
    iter=iter+1;
    Kdx=kernel(D,X,ker);
    Kdd=kernel(D,D,ker);
    Z_new=inv(Kdd+beta*eye(d))*Kdx;
    switch ker.type
        case 'rbf'
            g_Kxd=-Z_new';
            g_Kdd=0.5*Z_new*Z_new'+0.5*alpha*eye(d);
            [g_D1,T1,C1]=gXY(g_Kxd,Kdx',X,D,ker,'Y');
            [g_D2,T2,C2]=gXX(g_Kdd,Kdd,D,ker);
            tau=gamma/ker.par*(2*T2-diag(C1(1,:)+2*C2(1,:))); 
            g_D=(g_D1+g_D2)/tau;
        case 'poly'
            XD=X'*D;
            DD=D'*D;
            W1=(XD+ker.par(1)).^(ker.par(2)-1);
            W2=(DD+ker.par(1)).^(ker.par(2)-1);
            g_D=-X*(W1.*Z_new')+D*(Z_new*Z_new'.*W2)+alpha*D*(W2.*eye(size(W2)));
            tau=gamma*(Z_new*Z_new'.*W2+alpha*W2.*eye(size(W2)));
            g_D=g_D/tau;
    end
    vD=g_D+eta*vD;
    D_new=D-vD;
    % dX
    switch ker.type
        case 'rbf'
            g_Kxd=-Z_new';
            Kdx=kernel(D_new,X,ker);
            [g_X,~,C]=gXY(g_Kxd,Kdx',X,D_new,ker,'X');
            tau=gamma*(1/ker.par*(-C(1,:)));
            g_X=g_X.*repmat((tau.^(-1)),m,1);
        case 'poly'
            XD=X'*D_new;
            x2=sum(X.^2);
            W1=(x2+ker.par(1)).^(ker.par(2)-1);
            W2=(XD+ker.par(1)).^(ker.par(2)-1);
            g_X=X.*(ones(m,1)*W1)-D*(W2'.*Z_new);
            v=(gamma*W1).^(-1);
            g_X=g_X.*repmat(v,size(X,1),1);            
    end
    vX=g_X+eta*vX;
    X_new=X-vX;
    X_new=X_new.*(1-M)+X0.*M;
    %
    dZ=norm(Z-Z_new,'fro')/norm(Z,'fro');
    dD=norm(D-D_new,'fro')/norm(D,'fro');
    dX=norm(X-X_new,'fro')/norm(X,'fro');
    cvg=0;
    if dX<e
        cvg=1;
    else
        cvg=0;
    end
    if iter<4||mod(iter,50)==0||cvg==1
        if n<8000
            switch ker.type
                case 'rbf'
                    Kxx=eye(n);
                case 'poly'
                    Kxx=diag((x2+ker.par(1)).^ker.par(2));
            end
            J=0.5*trace(Kxx-2*Kdx'*Z+Z'*Kdd*Z)+0.5*alpha*trace(Kdd)+0.5*beta*sum(Z(:).^2);
        else
            J='NotComputed';
        end
        disp(['Iter ' num2str(iter) ': J=' num2str(J) ', dZ=' num2str(dZ) ', dD=' num2str(dD) ', dX=' num2str(dX)])
    end
    Z=Z_new;
    D=D_new;
    X=X_new;
    if cvg==1
        disp('Converged!')
        break
    end
end
end

%%
function [K,XY]=kernel(X,Y,ker)
nx=size(X,2);
ny=size(Y,2);
XY=X'*Y;
if strcmp(ker.type,'rbf')
    xx=sum(X.*X,1);
    yy=sum(Y.*Y,1);
    D=repmat(xx',1,ny) + repmat(yy,nx,1) - 2*XY;
    K=exp(-D/2/ker.par); 
end
if strcmp(ker.type,'poly')
    K=(XY+ker.par(1)).^ker.par(2);
end
end
%%
function [g,T,C]=gXY(g_Kxd,Kxd,X,D,ker,v)
switch v
    case 'Y'
        T=g_Kxd.*Kxd;% n x d
        C=repmat(sum(T),size(X,1),1);
        g=1/ker.par*(X*T-D.*C);  
    case 'X'
        T=g_Kxd'.*Kxd';% d x n;
        C=repmat(sum(T),size(X,1),1);
        g=1/ker.par*(D*T-X.*C);
end
end

%%
function [g,T,C]=gXX(g_Kdd,Kdd,D,ker,I)
if ~exist('I')
    T=g_Kdd.*Kdd;
    C=repmat(sum(T),size(D,1),1);
    g=2/ker.par*(D*T-D.*C);
else
    T=g_Kdd.*Kdd;
    C=repmat(sum(T),size(D,1),1);
    g=2/ker.par*(D.*repmat(diag(T)',size(D,1),1)-D.*C);
end
end 
