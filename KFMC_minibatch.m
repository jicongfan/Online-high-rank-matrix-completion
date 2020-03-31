function [X,D,Z,J,ker,RE] = KFMC_minibatch(X,M,d,alpha,beta,ker,options)
% This is the codes of online version of KFMC in the following paper:
% Jicong Fan, Madeleine Udell. Online High-Rank Matrix Completion. CVPR 2019.
% Input:    X the incomplete matrix (mxn)
%           M binary matrix (mxn), 0 for missing entries, 1 for observed entries
%           d the dimension of D
%           alpha regularization parameter of D; for RBF kernel, alpha has
%                      no effect on the method
%           beta regularization parameter of Z
%           ker kernels; ker.type='rbf' or 'poly'; for rbf kernel,
%                        ker.par=sigma2; if ker.par=0, estimate it with
%                        ker.c; ker.c is often set as 1, 2, or 3.
%                        for poly kernel, ker.par=[c d], d is the degree of
%                        polynomial
%           options.gamma
%           options.eta the momentum paprameter, the default value is 0.5
%           options.online_maxiter (iteration for x and z), the default value is 50
%           options.npass the number of passes of whole data matrix
% Written by Jicong Fan, 09/2019, E-mail: jf577@cornell.edu
[m,n]=size(X);
if isfield(options,'D')
    D=options.D;
else
    D=randn(m,d);
end
Z=zeros(d,n);
X0=X;
e=1e-5;
%
if n>10000
Xt=X(:,randperm(n,10000));
else
Xt=X;
end
nt=size(Xt,2);
if strcmp(ker.type,'rbf') && ker.par==0
    if ker.c>10
        ker.par=var(X(:,1))*ker.c;
    else
        XX=sum(Xt.*Xt,1);
        dist=repmat(XX,nt,1) + repmat(XX',1,nt) - 2*Xt'*Xt;
        ker.par=(mean(mean(real(dist.^0.5)))*ker.c)^2;
    end
    clear dist Xt XX
end
if strcmp(ker.type,'poly') && ker.par(1)==0
    ker.par=[1 2];% [c d]
end
% monentum
if isfield(options,'eta')
    eta=options.eta;
else
	eta=0.5;
end
vD=zeros(size(D));
% iter
if isfield(options,'online_maxiter')
    maxiter=options.online_maxiter;
else
	maxiter=50;
end
iter=0;
disp(['kernel type: ' ker.type ' kernel parameter(s):' num2str(ker.par) ' alpha=' num2str(alpha) ' beta=' num2str(beta) ' momentum_eta=' num2str(eta)]) 
if isfield(options,'npass')
    npass=options.npass;
else
	npass=5;
end
if isfield(options,'batch_size')
    batch_size=options.batch_size;
else
	batch_size=10;
end
if isfield(options,'gamma')
    gamma=options.gamma;
else
	gamma=1.1;
end
tt=0;
eJs=0;
exs=0;
eDs=0;
for k=1:npass
    vD=zeros(size(D));
    i=1;
while i<n
    idx=[i:min(i+batch_size-1,n)];
    x=X(:,idx);
    vx=zeros(size(x));
    z=Z(:,idx);
    Kdd=kernel(D,D,ker);
    invKdd=inv(Kdd+beta*eye(d));
    for j=1:maxiter
        Kxx=kernel(x,x,ker);
        Kdx=kernel(D,x,ker);
        % obj_f
        J(i)=0.5*trace(Kxx-Kdx'*z-z'*Kdx+z'*Kdd*z)+0.5*alpha*trace(Kdd)+0.5*beta*sum(z(:).^2);
        % update z
        z_new=invKdd*Kdx;
        % update x
        if sum(sum(M(:,idx)))>0
        switch ker.type
            case 'rbf'
                g_Kxx=0.5;
                g_Kxd=-z_new';
                Kdx=kernel(D,x,ker);
                [g_X1,T1,C1]=gXY(g_Kxd,Kdx',x,D,ker,'X');     
                tau=gamma/ker.par*mean(abs(C1(:)));
                g_X=g_X1/tau;
            case 'poly'
                W1=(x'*x+ker.par(1)).^(ker.par(2)-1);
                W2=(x'*D+ker.par(1)).^(ker.par(2)-1);
                g_X=ker.par(2)*x*W1-ker.par(2)*D*(W2'.*z_new);
                v=diag(ker.par(2)*W1.*eye(size(W1))).^(-1);
                tau=gamma*(ker.par(2)*W1);
                g_X=g_X/tau;
        end
        vx=eta*vx+g_X;
        x_new=x-vx;
        x_new=x_new.*(1-M(:,idx))+X0(:,idx).*M(:,idx);
        else
            x_new=x;
        end
        %
        ez=max(abs(z-z_new));
        ex=max(abs(x-x_new));
        z=z_new;
        x=x_new;
        if max(ez,ex)<1e-5
            break
        end
    end
%     ex00=norm(options.X_true(:,i)-x)/norm(options.X_true(:,i));
    tt=tt+1;
    eJs=eJs+J(i);
%     exs=exs+ex00;
    Z(:,idx)=z;
    X(:,idx)=x;
    
   %% update D
    switch ker.type
        case 'rbf'
            Kdx=kernel(D,x,ker);
            Kdd=kernel(D,D,ker);
            g_Kxd=-z';
            g_Kdd=0.5*z*z'+0.5*alpha*eye(d);
            [g_D1,T1,C1]=gXY(g_Kxd,Kdx',x,D,ker,'Y');
            [g_D2,T2,C2]=gXX(g_Kdd,Kdd,D,ker);
            tau=gamma*normest(1/ker.par*(2*T2-diag(1*C1(1,:)+2*C2(1,:))));
            g_D=(g_D1+g_D2)/tau;
        case 'poly'
            W1=(x'*D+ker.par(1)).^(ker.par(2)-1);
            W2=(D'*D+ker.par(1)).^(ker.par(2)-1);
            g_D=-ker.par(2)*x*(W1.*z')+ker.par(2)*D*(z*z'.*W2)+alpha*ker.par(2)*D*(W2.*eye(size(W2)));
            tau=gamma*normest(ker.par(2)*z*z'.*W2+alpha*ker.par(2)*W2.*eye(size(W2)));
            g_D=g_D/tau;
    end
    vD=eta*vD+g_D;
    D_new=D-vD;
    eD=max(abs(D-D_new));
    D=D_new;
    eD=norm(vD,'fro')/norm(D,'fro');
    if mod(i-1,batch_size*10)==0
    disp(['pass ' num2str(k) ', iteration(data)' num2str(i) ', J=' num2str(J(i))  ', ez=' num2str(ez) ', ex=' num2str(ex) ', eD=' num2str(eD)])
    end
    i=i+batch_size;
%     eDs=eDs+eD;
%     cD=[cD [ex00;exs/tt;J(i);eJs/tt]];
end
if isfield(options,'X_true')
    RE(k)=norm(X-options.X_true,'fro')/norm(options.X_true,'fro');
else
    RE(k)=1;
end
end
end
%%
function K=kernel(X,Y,ker)
nx=size(X,2);
ny=size(Y,2);
if strcmp(ker.type,'rbf')
    xx=sum(X.*X,1);
    yy=sum(Y.*Y,1);
    D=repmat(xx',1,ny) + repmat(yy,nx,1) - 2*X'*Y;
    K=exp(-D/2/ker.par); 
end
if strcmp(ker.type,'poly')
    K=(X'*Y+ker.par(1)).^ker.par(2);
end
end
%%
function [g,T,C]=gXX(g_Kdd,Kdd,D,ker)
T=g_Kdd.*Kdd;
g=2/ker.par*(D*T-D.*repmat(sum(T),size(D,1),1));
C=abs(repmat(sum(T),size(D,1),1));
end 
%%
function [g,T,C]=gXY(g_Kxd,Kxd,X,D,ker,v)% m x d
switch v
    case 'X'
        T=g_Kxd'.*Kxd';% d x n;
        g=1/ker.par*(D*T-X.*repmat(sum(T),size(X,1),1));
        C=repmat(sum(T),size(X,1),1);
    case 'Y'
        T=g_Kxd.*Kxd;% n x d
        g=1/ker.par*(X*T-D.*repmat(sum(T,1),size(X,1),1));
        C=repmat(sum(T,1),size(X,1),1);
end
end 
