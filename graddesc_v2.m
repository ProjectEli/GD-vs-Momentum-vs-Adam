%% Gradient descent method
% Jaehyeok Park, 2019/12/09, All rights reserved

%% Init
close all; clear; clc;

f = @(x1,x2) x1.^2 + x1.*x2 + 3*x2.^2;
g = @(x1,x2) [2*x1.^1 + x1, x2 + 6*x2]; % const

%cd = [uint8(hot(100)*255) uint8(ones(100,1))].';


%% Definitions
% define parameters
maxN = 10000; % max iterations

tolerance = 1e-6;
mindx = 1e-6; % min walk

alpha = 0.01; % learning rate
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

% initial value
x0 = [-4.8 -4.1];
% x0 = [0.01 0.1];
Nx0 = length(x0); % length of vector

% initial gradient
gnormGD =  inf;
xGD = x0;
dx = inf; % initial walk;

% predefined parameter for GD
vxGD1 = x0(1);% zeros(1,maxN);
vxGD2 = x0(2);% zeros(1,maxN);
vfGD = f(x0(1),x0(2)); % zeros(1,maxN);

% momentum method
gamma = 0.9;
vMom = 0;
gnormMom = inf;
xMom = x0;
vxMom1 = x0(1);
vxMom2 = x0(2);
vfMom = f(vxMom1,vxMom2);

% predefine for Adam
gnormAdam = inf;
m = 0;
v = 0;
xAdam = x0;
vxAdam1 = x0(1); % Adam
vxAdam2 = x0(2);
vfAdam = f(vxAdam1,vxAdam2);

%% target function
[X, Y] = meshgrid(-5:0.05:5,-5:0.05:5);

figure(1);
surf(X,Y,f(X,Y)); view(84,31); 
shading interp;
hold on;
pGD= plot3(vxGD1,vxGD2,vfGD,'r','LineWidth',3,'DisplayName','GD');
pMom = plot3(vxMom1,vxMom2,vfMom,'g','LineWidth',3,'DisplayName','Momentum');
pAdam = plot3(vxAdam1,vxAdam2,vfAdam,'k','LineWidth',3,'DisplayName','Adam');
hold off;
legend('show');
%linkdata on;

pGD.XDataSource = 'vxGD1';
pGD.YDataSource = 'vxGD2';
pGD.ZDataSource = 'vfGD';
% set(pGD.Edge,'ColorBinding','interpolated','ColorData',colormap('parula'));

pMom.XDataSource = 'vxMom1';
pMom.YDataSource = 'vxMom2';
pMom.ZDataSource = 'vfMom';

pAdam.XDataSource = 'vxAdam1';
pAdam.YDataSource = 'vxAdam2';
pAdam.ZDataSource = 'vfAdam';

for k = 1:maxN
%     disp(k);
    % Conv check
    if gnormGD < tolerance
        refreshdata; drawnow; % update plot
        disp('GD converged');
        disp(['iters =' num2str(k)]);
        break;
    elseif gnormMom < tolerance
        refreshdata; drawnow; % update plot
        disp('Momentum converged'); 
        disp(['iters =' num2str(k)]);
        break;
    elseif gnormAdam < tolerance
        refreshdata; drawnow; % update plot
        disp('Adam converged'); 
        disp(['iters =' num2str(k)]);
        break;
    end
    
    % GD method
    grad = g(xGD(1),xGD(2));
    gnormGD = norm(grad);
    xnewGD = xGD - alpha*grad;
    
    if ~isfinite(xnewGD)
        disp(['Stopped at ' num2str(k) 'th iteration: xGD is not finite']);break;
%     else
%         disp([num2str(k) char(9) 'GD norm=' num2str(gnormGD,2)]);
    end
    
    vxGD1 = [vxGD1 xnewGD(1)];
    vxGD2 = [vxGD2 xnewGD(2)];
    vfGD =  [vfGD f(xnewGD(1),xnewGD(2))];
    
    dx = norm(xnewGD-xGD); % update variables
    xGD=xnewGD;
    
    % Momentum
    gradMom = g(xMom(1),xMom(2));
    gnormMom = norm(gradMom);
    
    vMom = gamma*vMom + alpha * gradMom;
    xMom = xMom - vMom;
    
    vxMom1 = [vxMom1 xMom(1)];
    vxMom2 = [vxMom2 xMom(2)];
    vfMom = [vfMom f(xMom(1),xMom(2))];
    
    % Adam
    gradAdam = g(xAdam(1),xAdam(2));
    gnormAdam = norm(gradAdam);
    
    m = beta1*m + (1-beta1)* gradAdam; % biased first moment
    v = beta2*v + (1-beta2)* gradAdam.^2; % biased second moment
    
    m_hat = m/(1-beta1^k); % bias-corrected first raw moment
    v_hat = v/(1-beta2^k); % bias-corrected second raw moment
    
    xAdam = xAdam - alpha*m_hat./(sqrt(v_hat)+epsilon); % update adam step
    
    if ~isfinite(xAdam)
        disp(['Stopped at ' num2str(k) 'th iteration: xAdam is not finite']);break;
%     else
%         disp([num2str(k) char(9) 'Adam norm=' num2str(gnormAdam,2)]);
    end
    
    vxAdam1 = [vxAdam1 xAdam(1)];
    vxAdam2 = [vxAdam2 xAdam(2)];
    vfAdam = [vfAdam f(xAdam(1),xAdam(2))];
    
    if mod(k,50) == 0
        refreshdata; drawnow; % update plot
    end
end
xoptGD = xGD;
foptGD = f(xoptGD(1),xoptGD(2));
xoptAdam = xAdam;
foptAdam = f(xoptAdam(1),xoptAdam(2));
Nfin = k;