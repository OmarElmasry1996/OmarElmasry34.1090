function [ cost,thetas,h,iterations] = costFunction( xFeatures,thetas,m,yNormalised )
 

% a = 0.0001;
a = 0.01;
iterations = 3000;
cost = zeros(1,iterations);
count = 0;
for z = 1:iterations
    
     h = 1./(1+exp(-xFeatures*thetas));



    for i = 1 : length(thetas)
        
        thetas(i,1) = thetas(i,1) - a*(1/m)*(h-yNormalised)'*xFeatures(:,i);
    
    end
    
    cost(z) = (1/m)*sum((-yNormalised'*log(h))-(1-yNormalised)'*log(1-h));
    count = count + 1
%     cost(z) = -(1/m)*sum(yNormalised.*log(h)+(1-yNormalised).*log(1-h));

end
% for j=1:iterations
%     
% h = 1./(1+exp(-xFeatures*thetas));
% cost(j) = -(1/m)*sum(yNormalised.*log(h)+(1-yNormalised).*log(1-h));
% thetas = thetas+(a/length(yNormalised))*xFeatures'*(yNormalised-h);
% 
% end
% 
% grad=zeros(size(thetas,1),1);
% 
% for i=1:length(thetas)
%     grad(i)=(1/m)*sum((h-yNormalised)'*xFeatures(:,i));
% end


end

