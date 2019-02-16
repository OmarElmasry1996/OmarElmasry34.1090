function [ xNormalised,yNormalised,mu2,sDeviation2] = normalise(x,column,y)

mu = 0;
sDeviation = 0;
xNormalised = zeros(size(x));               %normailising
yNormalised= zeros(size(y));

for i = 1 : column
    mu = mean(x(:,i));
    sDeviation = std(x(:,i));                         
    xNormalised(:,i) = (x(:,i) - mu)/sDeviation;
end
mu2 = mean(y);
sDeviation2 = std(y);
yNormalised = (y-mu2)/sDeviation2;


end

