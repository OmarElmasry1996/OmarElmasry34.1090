function [  xNormalised,yNormalised,maximum ] = normaliseMax( x,column,y )

maximum = 0;
maximum2 = 0;
xNormalised = zeros(size(x));               %normailising
yNormalised= zeros(size(y));

for i = 1 : column
    maximum = max(x(:,i));
    xNormalised(:,i) = x(:,i)/maximum;
end
maximum2 = max(y);
yNormalised = y/maximum2;


end




