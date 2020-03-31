function [X,D,Z,J,ker] = KFMC_ose(X,M,D,beta,ker,options)
% % out-of-sample extension, for new column
% min |phi(x)-phi(D)z|+beta|Z|
[m,n]=size(X);
d=size(D,2);
Z=zeros(d,n);
X0=X;
e=1e-5;
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
	maxiter=100;
end
iter=0;
disp(['kernel type: ' ker.type ' kernel parameter(s):' num2str(ker.par) ' beta=' num2str(beta) ' momentum_eta=' num2str(eta)]) 
Kdd=kernel(D,D,ker);
invKdd=inv(Kdd+beta*eye(d));
for i=1:n
    x=X(:,i);
%     idx=find(M(:,i)==0);
%     x(idx)=randn(1,length(idx))*100;
    vx=zeros(size(x));
    z=Z(:,i);
    for j=1:maxiter
        Kxx=kernel(x,x,ker);
        Kdx=kernel(D,x,ker);
        % obj_f
        J(j,i)=0.5*trace(Kxx-Kdx'*z-z'*Kdx+z'*Kdd*z)+0.5*beta*sum(z.^2);
        % update z
        z_new=invKdd*Kdx;
        % update x
        switch ker.type
            case 'rbf'
                g_Kxx=0.5;
                g_Kxd=-z_new';
                Kdx=kernel(D,x,ker);
                [g_X1,T1,C1]=gXY(g_Kxd,Kdx',x,D,ker,'X');
        %         [g_X2,T2,C2]=gD2(g_Kxx,Kxx,x,ker);
%                 tau=1*normest(1/ker.par*(T1));
                 tau=1.1/ker.par*mean(abs(C1(:)));
        %         tau=(1/ker.par*(T2-eye(size(T2))*(mean(C1(:)+C2(:)))));
                g_X=1*g_X1/tau;
            case 'poly'
                W1=(x'*x+ker.par(1)).^(ker.par(2)-1);
                W2=(x'*D+ker.par(1)).^(ker.par(2)-1);
                g_X=ker.par(2)*x*W1-ker.par(2)*D*(W2'.*z_new);
%                 v=diag(ker.par(2)*W1.*eye(size(W1))).^(-1);
%                 g_X=g_X.*repmat(v',size(x,1),1);
                g_X=1*g_X/(ker.par(2)*W1);
        end
        vx=eta*vx+g_X;
        x_new=x-vx;
        x_new=x_new.*(1-M(:,i))+X0(:,i).*M(:,i);
        %
        ez=max(abs(z-z_new));
        ex=max(abs(x-x_new));
        z=z_new;
        x=x_new;
        if max(ez,ex)<1e-5
            break
        end
        %         disp(['i=' num2str(i) ' j=' num2str(j) ' f=' num2str(J(j,i))])
    end
    if mod(i,100)==0
    disp(['i=' num2str(i) ' J=' num2str(J(end,i))  ' ez=' num2str(ez) ' ex=' num2str(ex)])
    end
    Z(:,i)=z;
    X(:,i)=x;
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
function [g,T,C]=gD1(g_Kxd,Kxd,X,D,ker)% m x d
if strcmp(ker.type,'rbf')

%     tau=1/ker.par*max(abs(BH(:)));
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
