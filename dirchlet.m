%figure 2.5
close all
clear
x1 = linspace(0,1,101);
x2 = linspace(0,1,101);
[X1,X2] = ndgrid(x1,x2);
i = 0;
for a = [0.1,1,10]
    alpha = a*ones(3,1);
    X3 = a - X1 - X2;
    bad = (X1+X2 > 1); X1(bad) = NaN; X2(bad) = NaN; X3(bad) = NaN;
    i = i+1;
    betaConst = exp(sum(gammaln(alpha))-gammaln(sum(alpha)));
    F = real((X1.^(alpha(1)-1) .* X2.^(alpha(2)-1) .* X3.^(alpha(3)-1)) / betaConst);
    subplot(1,3,i)
    surf(X1,X2,F)
    view(100,20)
end
colorbar