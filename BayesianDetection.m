% Implementation of the paper ` RECOVERING CLIPPED OFDM SYMBOLS WITH
% BAYESIAN INFERENCE ' 2000

%% Basic Setting
clear all
close all

N = 2048; % Signal length
Modulation = 'QAM'; % QAM or PSK
Mod_M = 16; % M used in modulation. e.g., 64 QAM
A = 4.42; % Linear Range of the amplifier
M = 5; % Maximum delay slots
SNR_dB = 30
eps = 1e-4;

% Transmission and receive
% Frequency Domain Signal
S = randi(Mod_M,1,N) -1;
if strcmp(Modulation,'QAM')
    X = qammod(S,Mod_M);
elseif strcmp(Modulation,'PSK')
    X = pskmod(S,Mod_M);
end
x = sqrt(N) * ifft(X);
%x = ifft(X);
scatterplot(X)
title('X')

% Clipping Distortion
x_clip_pos = abs(x) > A;
x_linear_pos = abs(x) <= A;
x_out = x .* x_linear_pos + A * exp(1j* angle(x)) .* x_clip_pos;
X_out = 1/sqrt(N) * fft(x_out);
%X_out = fft(x_out);
scatterplot(X_out)
title('X out')
%plot([abs(x)' abs(x_out)'])

% AWGN channel
h = [exp(1-(0:M-1)/M*5)/exp(1) zeros(1,N-M)];

%H = 1/sqrt(N) * fft(h); % Channel Frequency Respond
H = fft(h); % Channel Frequency Respond
SNR = 10^(SNR_dB/10);
Var_W = max(abs(X))/ sqrt(SNR);


W = sqrt(Var_W/2)* randn(1,N) + 1j*sqrt(Var_W/2)* randn(1,N); % Complex AWG noise
%W = 0;
Y = X_out .* H + W; % Received Frequency Signal

scatterplot(Y)
title('Y')
% Equalization
% Assuming perfect detection of H
Y_bar = Y ./ H;
scatterplot(Y_bar)
title('Y bar')

y_bar = sqrt(N) * ifft(Y_bar);
%y_bar = ifft(Y_bar);


U_table = (1:Mod_M) -1;
if strcmp(Modulation,'QAM')
    S_table = qammod(U_table,Mod_M);
elseif strcmp(Modulation,'PSK')
    S_table = pskmod(U_table,Mod_M);
end
%% ML Estimation

%% MAP Iterative Estimation
% Setting prior
Gamma_a_0 = 10;
Gamma_b_0 = 5;

% Initialization
S0 = randi(Mod_M,1,N) -1;
if strcmp(Modulation,'QAM')
    X0 = qammod(S0,Mod_M);
elseif strcmp(Modulation,'PSK')
    X0 = pskmod(S0,Mod_M);
end

z0 = randi(2,1,N) -1;

Var0 = 1/gamrnd(Gamma_a_0,Gamma_b_0);

% Update


iter_Max = 4;
X_old = X0;
z_old = z0;
Var_old = Var0;
for iter = 1:iter_Max
    fprintf('Iteration No. %d\n',iter);
    x_old = sqrt(N) * ifft(X_old);
    %x_old = ifft(X_old);

    % Update z
    z_prob = zeros(1,N);
    for n = 1:1:N
       z_prob(n) = 1/(1+ exp( -1/ (1*Var_old) * ( abs( y_bar(n) - A * exp(1j* angle(x_old(n))) )^2 - abs( y_bar(n) - x_old(n) )^2 )));
    end
    z_new = binornd(1,z_prob);
    
    % Update x
    X_new = X_old;
    for l = 1:1:N
        
        pkl_Su = zeros(1,Mod_M);
        for u = 1:1:Mod_M
           
            Xnk_GivenX = X_old;
            Xnk_GivenX(l) = S_table(u);
            xnk_GivenX = sqrt(N) * ifft(Xnk_GivenX);
            %xnk_GivenX = ifft(Xnk_GivenX);
            
            sum_lu = 0;
            for n = 1:1:N
                
                if z_new(n) == 0
                    sum_lu = sum_lu + abs(y_bar(n) - xnk_GivenX(n))^2;
                else
                    sum_lu = sum_lu + abs(y_bar(n) - A*exp(1j*angle(xnk_GivenX(n))) )^2;
                end
            end
            
            pkl_Su(u) = sum_lu;
        end
        
        [prob_max,u_max] = max(pkl_Su);
        X_new(l) = S_table(u_max);
    end
    x_new = sqrt(N) * ifft(X_new);
    %x_new = ifft(X_new);
    
    % Update sigma^2
    Gamma_a_N = Gamma_a_0 + N/2;
    Gamma_b_N = Gamma_b_0;
    for n = 1:1:N
       Gamma_b_N = Gamma_b_N + 1/2* abs( y_bar(n) - x_new(n)*(1-z_new(n)) - z_new(n)*A*exp(1j*angle(x_new(n))))^2; 
    end
    Var_new = 1/gamrnd(Gamma_a_N,Gamma_b_N);
    
    % renew
    z_old = z_new;
    x_old = x_new;
    X_old = X_new;
    Var_old = Var_new;
    
    % Stat
    ACC_z = 1 - sum( abs(x_clip_pos - z_new) <= eps )/N;
    ACC_X = 1 - sum( abs(X - X_new) <= eps)/N;
end

%ACC_z = sum(x_clip_pos == z_new)/N
scatterplot(X_new)
title('X estimated')
IBO_dB = 10*log10(A^2 / mean(abs(x).^2))
real_Var_eps = Var_W / N * sum( 1./ abs(H) .^2)

%% Heuristic method for Comparison
X_comp = zeros(1,N);
for n = 1:1:N
    [min_distance,min_u] = min( abs(Y_bar(n) - S_table) );
    X_comp(n) = S_table(min_u);
end

SER_comp = sum( abs(X_comp - X) > eps)/N

%% Figure Generating
IBO_dB = 10*log10(A^2 / mean(abs(x).^2))
real_Var_eps = Var_W / N * sum( 1./ abs(H) .^2)

figure
semilogy(1/N:1/N:1,abs(H))
axis([0 1 0.1 1.2*max(abs(H))])
grid on
axis on
title('Channel Transfer Function')
xlabel('Normalized Frequency')
ylabel('|H_l|')

scatterplot(X)
title('Constellation')
grid on
axis on

scatterplot(X_out)
title('Clipped Signal (Frequency Domain)')
grid on
axis on

scatterplot(Y_bar)
title('Equalized Signal (Frequency Domain)')
grid on
axis on

d = [0 0 0.05 0.1 0.15 0.2 0.3 0.4 0.6 8]*1;
SNR_dB_test = [20 20.5 21 21.5 22 23 24 25 27 30];
SER_test = [0.089 0.08 0.063 0.051 0.032 0.018  0.011 0.005 6.3e-4 NaN];
%SER_test = SER_test .* (10.^(d));
SER_comp_test = [0.155 0.151 0.141 0.132 0.126 0.11 0.099 0.084 0.064 0.04];
%SER_comp_test = SER_comp_test.* (10.^(-d));
figure
semilogy(SNR_dB_test,SER_test,'*-')
%legend('Presented Method')
hold on 
semilogy(SNR_dB_test,SER_comp_test,'r^-')
legend('Presented Method','Heuristic Method')
axis([20 30 2e-4 0.25])
grid on
axis on
title('Symbol Error Rate vs SNR')
xlabel('SNR (dB)')
ylabel('SER')

%% Spark Running time
Npart = [1 2 3 5 7 10 15 20];
T_average = [1.89 1.76 1.51 1.49 1.53 1.79 1.98 2.32];
plot(Npart,T_average)
title('Average Run Time vs Partition Number')
xlabel('Partition Number')
ylabel('Average Run Time')