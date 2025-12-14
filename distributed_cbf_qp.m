% 13 December 2025
% Implementation of the distributed RSDD problem, complete with guide and details
% This code is for a university course on multi-agent systems.
% As the original paper is rather unclear, I added many comments to
% showcase the choices you can take.
% If you find this useful, check out my other projects!
% (I am mainly a mechanical engineer)

% 1. Reset MATLAB
close all
clear

% 2. Define objects (structures)
% 2.1. Define agent
% Initial & final state
agents.num = 4;
pos_init = 5*[1 1; 1 -1; -1 -1; -1 1].'-0*(rand(2,4)-0.5);
axisVector = 10*[-1 1 -1 1];

% Harder example
% agents.num = 10;
% R = 15;
% pos_init = R*[cos(2*pi/(agents.num/2)*(0:(agents.num/2-1)));...
%     sin(2*pi/(agents.num/2)*(0:(agents.num/2-1)))];
% pos_init = [pos_init(1,:) -pos_init(1,:)
%     pos_init(2,:) -pos_init(2,:)] + (rand(2,agents.num)-0.5)*1;
% axisVector = 20*[-1 1 -1 1];

% pos_init = [-5.4 -5.1; 5.5 5.4].';
pos_final = -pos_init;

% Agents' parameters
% Define agents' state: position, velocity, and acceleration
agents.pos_init = pos_init;
agents.vel_init = zeros(2,agents.num);
agents.pos_final = pos_final;
agents.vel_final = zeros(2,agents.num);
agents.pos = agents.pos_init;
agents.vel = agents.vel_init;
agents.accel = zeros(2,agents.num);
%agents.accel_max = 10*ones(1,agents.num);
agents.accel_max = [1*ones(1,agents.num/2) 10*ones(1,agents.num/2)];
agents.accel_min = -agents.accel_max;
% Define agents' additional parameters
agents.diameter = 1*ones(1,agents.num); % Safety diameter
agents.connection_graph = zeros(agents.num); % System's connection_graph
agents.lambda = 10*rand(agents.num,agents.num,3); % System's lambda dual variable 
%agents.lambda = zeros(agents.num);
%agents.lambda = 20*mod((1:agents.num)+(1:agents.num)',2)-10;
agents.M = 35*ones(1,agents.num); % Penalty constant
agents.mu = zeros(agents.num); % System's mu dual variable
% CBF constants: gamma*b(x)^beta, b(x) is the CBF
agents.gamma = 0.05*ones(1,agents.num); % CBF's gamma constant
%agents.gamma = rand(1,agents.num);
agents.beta = 3*ones(1,agents.num); % CBF's beta constant

% 2.2. Define simulation time
% The simulation run from current_time to simulate_time
time.current_time = 0;
time.simulate_time = 10;
time.number_of_time_interval = 100;
time.time_interval = time.simulate_time/time.number_of_time_interval;
time.time_space = linspace(time.current_time,time.simulate_time,...
    time.number_of_time_interval);

% 2.3. Define the nominal controller (in this case a PID controller)
% There are 2 PID controllers, one for position and one for velocity
% The efficency of the simulation seems to be heavily dependent on the
% nominal controller choosen
% With PID controller, the required detection distance between agents is
% essentially infinity (will be explained later)
PID = struct;
% For position
PID.pos.value = 0;
PID.pos.P = 2;
PID.pos.I = 1.5;
PID.pos.D = 0;
PID.pos.previousError = 0;
PID.pos.integral = 0;
% For velocity
PID.vel.value = 0;
PID.vel.P = 3;
PID.vel.I = 2;
PID.vel.D = 0;
PID.vel.previousError = 0;
PID.vel.integral = 0;

% 2.4. Define additional system variable
options = optimoptions(@quadprog,'Display','off');
rho_plot = zeros(1,time.number_of_time_interval);
rho_sum_i = zeros(1,agents.num);
collision_flag = 0;

% 2.5. Open video
video = VideoWriter("Agents trajectory","Motion JPEG AVI");
open(video);

% 3. Start simulation
% Start main time loop
for t = 1:time.number_of_time_interval
    % 3.1. Calculate nominal controller
    u_nom = nominal_controller(agents,PID,time);
    % 3.2. Update the connection graph
    agents = update_graph(agents);
    controller = zeros(2,agents.num);
    iter = 1;
    % The gamma lambda "constant" for subgradient method
    % This is one of the most essential part of the algorithm
    gamma_lambda = 0.05;
    % Subgradient method's relaxation coefficient
    beta = 0.2;
    % Each time iteration pick a random lambda matrix
    agents.lambda = 10*rand(agents.num,agents.num,3);
    collision_flag = 0;
    % 3.3. Start Algorithm 1 main loop
    while (norm(agents.lambda(:,:,1)-agents.lambda(:,:,2))>1e-5)
        % 3.3.1. Calculate gamma lambda
        % Some choices for gamma lambda
        % The current choice is the fastest (might be because it becomes
        % small very quickly? Not sure.)
        % Search the Stanford pdf on subgradient method for more details
        %             disp(abs(lambda_norm(2)-lambda_norm(1)));
        %             if (abs(lambda_norm(2)-lambda_norm(1))>1e-2)
        %                 gamma_lambda = 0.5;
        %             elseif (abs(lambda_norm(2)-lambda_norm(1))>1e-5)
        %                 gamma_lambda = 0.05;
        %             else
        %                 gamma_lambda = 1/iter;
        %             end
        %gamma_lambda = (iter+20)/(iter^2+300);
        gamma_lambda = gamma_lambda/sqrt(iter);
        % Iterate for each agent
        for i = 1:agents.num
            % 3.3.2. Find all agents within detection distance
            connected = find(agents.connection_graph(:,i)==1).';
            connected_size = length(connected);
            % 3.3.3. Calculate the QP-CBF
            % More details on this can be found in Ames 2016/2017 paper
            u_max = [agents.accel_max(i)*ones(1,2) 999*ones(1,connected_size)];
            u_min = [agents.accel_min(i)*ones(1,2) zeros(1,connected_size)];
            A = zeros(connected_size,2+connected_size);
            b = zeros(connected_size,1);
            u0 = u_nom(:,i);
            H = eye(2+connected_size);
            f = [-u0.' agents.M(i)*ones(1,connected_size)];
            connected_num1 = 1;
            for e = connected
                delta_p = agents.pos(:,i)-agents.pos(:,e);
                delta_v = agents.vel(:,i)-agents.vel(:,e);
                accel_max_sum = agents.accel_max(i)+agents.accel_max(e);
                Dij = agents.diameter(i) + agents.diameter(e);
                hij = sqrt(2*accel_max_sum*(norm(delta_p)-Dij)) + ...
                    delta_p.'*delta_v/norm(delta_p);
                Aij = -delta_p.';
                A(connected_num1,1:2) = Aij; % Calculate A matrix
                A(connected_num1,connected_num1+2) = -1;
                bij = agents.gamma(i)*hij^agents.beta(i)*norm(delta_p) ...
                    - (delta_v.'*delta_p)^2/(norm(delta_p))^2 ...
                    + norm(delta_v)^2 ...
                    + accel_max_sum*delta_v.'*delta_p ...
                    /sqrt(2*accel_max_sum*(norm(delta_p)-Dij));
%                 % If bij is not real (collision), recalculate
%                 if (~isreal(bij))
%                     disp("Error: Possible collision!")
%                     agents.lambda = 10*rand(agents.num,agents.num,3);
%                     collision_flag = 1;
%                     gamma_lambda = 0.5;
%                     break;
%                 end
                b(connected_num1) = bij; % Calculate b vector
                connected_num1 = connected_num1 + 1;
            end
%             if (collision_flag == 1)
%                 break;
%             end
            % Update b vector with difference between lambda variables
            b = b - (agents.lambda(i,connected,1).' - agents.lambda(connected,i,1));
            % Finally, use quadprog
            [u_star,~,exitflag,~,mu] = quadprog(H,f,A,b,[],[],u_min,u_max,[],options);
            rho_sum_i(i) = sum((u_star(3:end).^2+agents.M(i)*u_star(3:end)));
            %disp(A*u_star-b)
            agents.mu(i,connected) = mu.ineqlin(1:connected_size);
            controller(:,i) = u_star(1:2);
        end
        agents.lambda(:,:,3) = agents.lambda(:,:,2);
        agents.lambda(:,:,2) = agents.lambda(:,:,1);
        % 3.3.4. Update lambda variables with newly-found mu variables 
        for i = 1:agents.num
            connected = find(agents.connection_graph(:,i)==1).';
            for l = connected
                % With relaxation and past variables
                agents.lambda(i,l,1) = agents.lambda(i,l,2) - ...
                    gamma_lambda*(agents.mu(i,l) - agents.mu(l,i)) + ...
                    beta*(agents.lambda(i,l,2) - agents.lambda(i,l,3));
                % Given in the paper
%                 agents.lambda(i,l,1) = agents.lambda(i,l,2) - ...
%                     gamma_lambda*(agents.mu(i,l) - agents.mu(l,i));
            end
        end
        %disp(agents.lambda(:,:,1));
        disp(num2str(norm(agents.lambda(:,:,2)-agents.lambda(:,:,1))));
        iter = iter + 1;
    end
    disp("Error next time step:")
    rho_plot(t) = sum(rho_sum_i);
    % Update simulation with (backward?) Euler
    agents.accel = controller;
    agents.vel = agents.vel + agents.accel*time.time_interval;
    agents.pos = agents.pos + agents.vel*time.time_interval;
    plot_position(agents,video,axisVector)
end
figure()
plot(time.time_space,rho_plot,'r')

% Update the graph
function agents = update_graph(agents)
for i = 1:agents.num
    for j = i+1:agents.num
        %Dij = (agents.diameter(i) + agents.diameter(j))^3;
        % Ames gave a formula for safety diameter. In my experience, that
        % formula always gives a relatively large diameter value that is
        % essentially infinity
        % The formula would be more useful if the velocity can be bounded
        Dij = inf;
        if (norm(agents.pos(:,i)-agents.pos(:,j),2) < Dij)
            agents.connection_graph(i,j) = 1;
            agents.connection_graph(j,i) = 1;
        else
            agents.connection_graph(i,j) = 0;
            agents.connection_graph(j,i) = 0;
        end
    end
end
end

% define PID controller
function PID = PID_controller(agents,PID,time,varType)
dt = time.time_interval;
if varType == "pos"
    P = PID.pos.P;
    I = PID.pos.I;
    D = PID.pos.D;
    error = agents.pos_final - agents.pos;
    PID.pos.integral = PID.pos.integral + error*dt;
    derivative = (error - PID.pos.previousError)/dt;
    PID.pos.value = P*error + I*PID.pos.integral + D*derivative;
    PID.pos.previousError = error;
elseif varType == "vel"
    P = PID.vel.P;
    I = PID.vel.I;
    D = PID.vel.D;
    error = agents.vel_final - agents.vel;
    PID.vel.integral = PID.vel.integral + error*dt;
    derivative = (error - PID.vel.previousError)/dt;
    PID.vel.value = P*error + I*PID.vel.integral + D*derivative;
    PID.vel.previousError = error;
end
end

function controller = nominal_controller(agents,PID,time)
PID = PID_controller(agents,PID,time,"pos");
PID = PID_controller(agents,PID,time,"vel");
controller = PID.pos.value + PID.vel.value;
end

% Plot the simulation
function plot_position(agents,video,axisVector)
plot(agents.pos(1,:),agents.pos(2,:),'o',...
    'MarkerSize',20,"MarkerEdgeColor", "b");
grid on;
axis(axisVector);
pause(0.1);
writeVideo(video,getframe);
end
