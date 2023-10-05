function [data] = Datas()
data.T=2;% Simulation duration
%data.T=1.73;% el chido debe ser este
data.L = 10; % String length
data.E = 5E4; % Elasticity
data.rho = 1500; % Density
data.c = sqrt(data.E / data.rho);%velocity
%data.XI = 101; % Element index
end