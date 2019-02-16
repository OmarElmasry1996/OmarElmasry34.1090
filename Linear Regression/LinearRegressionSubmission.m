clc
clear all
close all

% file = readtable('house_prices_data_training_data.csv');
% data = table2array(file(1:18000-1,3:21));
% file2 = readtable('house_data_complete.csv');
% data2 = table2array(file2(18000:21614-1,3:21));

ds = datastore('house_data_complete.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
data = read(ds);



x1 = data{1:17999,5:7}; %3 features 
y1 = data{1:17999,3}; % price

x2 = x1.^2; % 2nd hypothesis squaring same features 
% y2 = y1.^2; 

x3 = data{1:17999,13:16}; % 3rd hypothesis 4 features 
x4 = x3.^2; %4th hypothesis squaring same features

xTest1 = data{18000:end,5:7};
yTest1 = data{18000:end,3};

xTest2 = xTest1.^2;
% yTest2 = yTest1.^2;

xTest3 = data{18000:end,13:16};
xTest4 = xTest3.^2;


[row column1] = size(x1);               %1st and 2nd hypothesis
[rowTest column1] = size(xTest1);      

[row column2] = size(x3);                    %3rd and 4th hypothesis
[rowTest column2] = size(xTest3);

m = row;
mTest = rowTest;

thetas1 = randn(column1+1,1);
htheta1 = thetas1;

thetas2 = randn(column1+1,1);
htheta2 = thetas2;

thetas3 = randn(column2+1,1);
htheta3 = thetas3;

thetas4 = randn(column2+1,1);
htheta4 = thetas4;

[ xNormalised1 , yNormalised1 ] = normalise(x1,column1,y1);       
[ xNormalisedTest1,yNormalisedTest1,mu1,sDeviation1] = normalise(xTest1,column1,yTest1);

[ xNormalised2 , yNormalised2 ] = normalise(x2,column1,y1);       
[ xNormalisedTest2,yNormalisedTest2,mu2,sDeviation2] = normalise(xTest2,column1,yTest1);

[ xNormalised3 , yNormalised3 ] = normalise(x3,column2,y1);       
[ xNormalisedTest3,yNormalisedTest3,mu3,sDeviation3] = normalise(xTest3,column2,yTest1);

[ xNormalised4 , yNormalised4 ] = normalise(x4,column2,y1);       
[ xNormalisedTest4,yNormalisedTest4,mu4,sDeviation4] = normalise(xTest4,column2,yTest1);


x0 = ones(m,1);
xFeatures1 = [x0 xNormalised1];
xFeatures2 = [x0 xNormalised2];
xFeatures3 = [x0 xNormalised3];
xFeatures4 = [x0 xNormalised4];

normEq1 = inv(xFeatures1'*xFeatures1)*xFeatures1'*yNormalised1;
normEq2 = inv(xFeatures2'*xFeatures2)*xFeatures2'*yNormalised2;
normEq3 = inv(xFeatures3'*xFeatures3)*xFeatures3'*yNormalised3;
normEq4 = inv(xFeatures4'*xFeatures4)*xFeatures4'*yNormalised4;




[ cost1,thetas1 ] = gradientDescent( xFeatures1,thetas1,m,yNormalised1 ); %cost function
[ cost2,thetas2 ] = gradientDescent( xFeatures2,thetas2,m,yNormalised2 );       %cost function
[ cost3,thetas3 ] = gradientDescent( xFeatures3,thetas3,m,yNormalised3 );       %cost function
[ cost4,thetas4 ] = gradientDescent( xFeatures4,thetas4,m,yNormalised4 );       %cost function


x00 = ones(mTest,1);
xNormalisedTest1 = [ x00 xNormalisedTest1];
xNormalisedTest2 = [ x00 xNormalisedTest2];
xNormalisedTest3 = [ x00 xNormalisedTest3];
xNormalisedTest4 = [ x00 xNormalisedTest4];


hFinal1 = abs(xNormalisedTest1*thetas1);
[prediction1] = denormaliseSTD(hFinal1,mu1,sDeviation1);
difference1 = abs(yTest1 - prediction1);


hFinal2 = abs(xNormalisedTest2*thetas2);
[prediction2] = denormaliseSTD(hFinal2,mu2,sDeviation2);
difference2 = abs(yTest1 - prediction2);


hFinal3 = abs(xNormalisedTest3*thetas3);
[prediction3] = denormaliseSTD(hFinal3,mu3,sDeviation3);
difference3 = abs(yTest1 - prediction3);


hFinal4 = abs(xNormalisedTest4*thetas4);
[prediction4] = denormaliseSTD(hFinal4,mu4,sDeviation4);
difference4 = abs(yTest1 - prediction4);






z = [1:2000];
figure()
plot(z,cost1);

figure()
plot(z,cost2);

figure()
plot(z,cost3);

figure()
plot(z,cost4);






    
    
