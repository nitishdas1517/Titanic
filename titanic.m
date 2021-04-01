clear ; close all; clc
data=csvread('train.csv');
X=data([2:end],[3,7,8,9,11]);
y=data([2:end],[2]);

[m n]=size(X);
X=[ones(m,1) X];
initial_theta=zeros(n+1,1);

[cost, grad]=costFunction(initial_theta,X,y);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options)
  
test=csvread('test.csv');
Xtest=test([2:end],[2,6,7,8,10]);
m_test=size(Xtest,1);
Xtest=[ones(m_test,1) Xtest];

p = round(sigmoid(X*theta))
psize=size(p);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
save("p.csv","p");

