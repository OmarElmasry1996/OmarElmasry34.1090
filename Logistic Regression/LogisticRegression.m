clc
clear all
close all

% file = readtable('house_prices_data_training_data.csv');
% data = table2array(file(1:18000-1,3:21));
% file2 = readtable('house_data_complete.csv');
% data2 = table2array(file2(18000:21614-1,3:21));

ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
data = read(ds);



x1 = data{1:175,1:3}; %3 features 
y1 = data{1:175,14}; % price

x2 = x1.^2; % 2nd hypothesis squaring same features 
% y2 = y1.^2; 

x3 = data{1:175,4:7}; % 3rd hypothesis 4 features 
x4 = x3.^2; %4th hypothesis squaring same features

xTest1 = data{176:end,1:3};
yTest1 = data{176:end,14};

xTest2 = xTest1.^2;
% yTest2 = yTest1.^2;

xTest3 = data{176:end,4:7};
xTest4 = xTest3.^2;


[row column1] = size(x1);               %1st and 2nd hypothesis
[rowTest column1] = size(xTest1);      

[row column2] = size(x3);                    %3rd and 4th hypothesis
[rowTest column2] = size(xTest3);

m = row;
mTest = rowTest;

thetas1 = randn(column1+1,1);
thetas2 = randn(column1+1,1);
thetas3 = randn(column2+1,1);
thetas4 = randn(column2+1,1);


    


[ xNormalised1,yNormalised1,max1] = normaliseMax(x1,column1,y1);       
[ xNormalisedTest1,yTestNormalised1] = normaliseMax(xTest1,column1,yTest1);

[ xNormalised2,yNormalised2,max2 ] = normaliseMax(x2,column1,y1);       
[ xNormalisedTest2,yTestNormalised2] = normaliseMax(xTest2,column1,yTest1);

[ xNormalised3,yNormalised3,max3 ] = normaliseMax(x3,column2,y1);       
[ xNormalisedTest3,yTestNormalised3] = normaliseMax(xTest3,column2,yTest1);

[ xNormalised4,yNormalised4,max4 ] = normaliseMax(x4,column2,y1);       
[ xNormalisedTest4,yTestNormalised4] = normaliseMax(xTest4,column2,yTest1);


x0 = ones(m,1);
xFeatures1 = [x0 xNormalised1];
xFeatures2 = [x0 xNormalised2];
xFeatures3 = [x0 xNormalised3];
xFeatures4 = [x0 xNormalised4];

normEq1 = inv(xFeatures1'*xFeatures1)*xFeatures1'*y1;
normEq2 = inv(xFeatures2'*xFeatures2)*xFeatures2'*y1;
normEq3 = inv(xFeatures3'*xFeatures3)*xFeatures3'*y1;
normEq4 = inv(xFeatures4'*xFeatures4)*xFeatures4'*y1;




[ cost1,thetas1,h1,iterations ] = costFunction( xFeatures1,thetas1,m,yNormalised1 ); %cost function
[ cost2,thetas2,h2,iterations ] = costFunction( xFeatures2,thetas2,m,yNormalised2 );       %cost function
[ cost3,thetas3,h3,iterations ] = costFunction( xFeatures3,thetas3,m,yNormalised3 );       %cost function
[ cost4,thetas4,h4,iterations ] = costFunction( xFeatures4,thetas4,m,yNormalised4 );       %cost function


hTotal = [h1,h2,h3,h4];

x00 = ones(mTest,1);
xNormalisedTest1 = [ x00 xNormalisedTest1];
xNormalisedTest2 = [ x00 xNormalisedTest2];
xNormalisedTest3 = [ x00 xNormalisedTest3];
xNormalisedTest4 = [ x00 xNormalisedTest4];


% xx = ones(mTest,1);
% xTestFinal1 = [ xx xTest1];
% xTestFinal2 = [ xx xTest2];
% xTestFinal3 = [ xx xTest3];
% xTestFinal4 = [ xx xTest4];


 predict1 = 1./(1+exp(-xNormalisedTest1*thetas1));
 predict2 = 1./(1+exp(-xNormalisedTest2*thetas2));
 predict3 = 1./(1+exp(-xNormalisedTest3*thetas3));
 predict4 = 1./(1+exp(-xNormalisedTest4*thetas4));
% 
% diff1 = abs(predict1-yTestNormalised1);
% diff2 = abs(predict2-yTestNormalised2);
% diff3 = abs(predict3-yTestNormalised3);
% diff4 = abs(predict4-yTestNormalised4);


z = [1:iterations];

figure()
plot(z,cost1);

figure()
plot(z,cost2);

figure()
plot(z,cost3);

figure()
plot(z,cost4);






    
    