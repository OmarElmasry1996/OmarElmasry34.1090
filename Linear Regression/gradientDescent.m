function [ cost,thetas ] = gradientDescent( xFeatures,thetas,m,y )

a = 0.001;
cost = zeros(1,2000);

for z = 1:2000
    
    h = (xFeatures*thetas);
    thetas=thetas-(a/m)*xFeatures'*(xFeatures*thetas-y);

%     for i = 1 : length(thetas)
%         
% %         thetas(i,1) = thetas(i,1) - a*(1/m)*sum((h-y)'*xFeatures(:,i));
%     
%     end
%     
    cost(z) = (1/(2*m))*sum((xFeatures*thetas - y).^2);
    
end

end

