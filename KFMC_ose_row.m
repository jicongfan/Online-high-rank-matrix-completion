function [Xr,D,Z,J,ker] = KFMC_ose_row(X,M,D,Z,lambda)
% out-of-sample extension, for new row
[d,n]=size(Z);
[mx,nx]=size(X);
disp('Completing new rows ...')
for i=1:mx
    id=find(M(i,:)==1);
    Xt=X(i,id);
    Zt=Z(:,id);
    Dr(i,:)=Xt*Zt'*inv(Zt*Zt'+lambda*eye(d));
end
Xr=Dr*Z.*(1-M)+X.*M;
end
