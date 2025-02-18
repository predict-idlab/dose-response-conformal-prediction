                            % MAIN PROGRAM FOR BENCHMARK PATIENT SIMULATOR
                                       % Ghent University, MAY 2024
                                       % ERC AMICAS 2022-2027

clearvars; clear memory; close all; clc;
delete(gcp('nocreate'));

% variable initialization
BIS_all= []; propofol_all = []; Remifentanil_all = []; Atracurium_all = []; Dopamine_all = []; SNP_all = [];
inner_loop_id = [];

Ts = 1200/60; % sampling time 5 minutes (30/60 min)
ST = 0.1*Ts; % simulation step size 

% initialize patient independent models
% hemodyn_model_AMICAS      % hemodynamic models
% dist_model_AMICAS         % disturbance models

%hemodyn_vars = evalin('base', 'hemodyn_model_AMICAS whos');

% Capture the state of the workspace before running the script
beforeVars = evalin('base', 'whos');
% Run the script
evalin('base', 'hemodyn_model_AMICAS');
% Capture the state of the workspace after running the script
afterVars = evalin('base', 'whos');
% Identify the new variables introduced by the script
newVars = setdiff({afterVars.name}, {beforeVars.name});
% Get detailed information for new variables
hemodyn_vars = afterVars(ismember({afterVars.name}, newVars));

% Capture the state of the workspace before running the script
beforeVars = evalin('base', 'whos');
% Run the script
evalin('base', 'dist_model_AMICAS');
% Capture the state of the workspace after running the script
afterVars = evalin('base', 'whos');
% Identify the new variables introduced by the script
newVars = setdiff({afterVars.name}, {beforeVars.name});
% Get detailed information for new variables
dist_model_vars = afterVars(ismember({afterVars.name}, newVars));


%% Choose patient database -> database_type = 1 (for 12 patient database - young patients), database_type = 2 (for 24 patient database - old patients)
load patient_database_generated_combined.mat

Patients = [];
for i = 1 : size(patients,1)
    age = patients(i,1);
    height = patients(i,2);
    weight = patients(i,3);
    sex = patients(i,4);
    Remifentanil = patients(i,5);
    Atracurium = patients(i,6);
    Dopamine = patients(i,7);
    SNP = patients(i,8);

    Patients = [Patients; Patient_appended(i, age, height, weight, sex,Remifentanil,Atracurium,Dopamine,SNP)];
end

[noOfPatients,~] = size(Patients)
%noOfPatients = 10

%% Choose simulation time (min) - See user manual
Tsim = 300; 

lower_bound_propofol = 0.05;
upper_bound_propofol = 0.5;
propofol_interval = 0.01;
propofol_loop_vector = lower_bound_propofol:propofol_interval:upper_bound_propofol;
propofol_loops = length(propofol_loop_vector);

%% Choose surface model type ->  RSM_type=1 (for Greco) / RSM_type=2 (for Minto) / RSM_type=3 (for Reduced Greco)
RSM_type = 2;

%% Choose disturbance profile -> See user manual
Dist_type = 1;

%% Choose anesthesiologist in loop or not -> Anest_loop = 1 / Anest_loop = 2 
Anest_loop = 1;

anestS=timeseries(anestS); 
no_anest=timeseries(no_anest);



BIS_all_results = cell(1, noOfPatients); % Cell array to store results for each patient
inner_loop_id_results = cell(1, noOfPatients);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%first loop for parallel - will be overwritten
patient = Patients(1);
Remifentanil = patient.Remifentanil; %binornd(1,0.3) * unifrnd(0,0.4); % [ug/kg/min]  
Atracurium = patient.Atracurium; %binornd(1,0.3) * unifrnd(0,29.5);       % [ug/kg/min] 
Dopamine = patient.Dopamine; %binornd(1,0.3) * unifrnd(0,20);         % [ug/kg/min] 
SNP = patient.SNP; %binornd(1,0.3) * unifrnd(0,10);  
Propofol = propofol_loop_vector(1);% [ug/kg/min]

inputStepR = [zeros(1,10) Remifentanil * ones(1,Tsim)]; %% 
inputStepA = [zeros(1,10) Atracurium * ones(1,Tsim)]; %% 
inputStepD = [zeros(1,10) Dopamine * ones(1,Tsim)]; %% 
inputStepS = [zeros(1,10) SNP * ones(1,Tsim)]; %% 

Reminput=timeseries(inputStepR); 
Atracinput=timeseries(inputStepA); 
Dopinput=timeseries(inputStepD); 
SNPinput=timeseries(inputStepS); 
inputStepP = [zeros(1,10) Propofol * ones(1,Tsim)]; %% 
% input profiles
Propinput=timeseries(inputStepP);

% Capture the state of the workspace before running the script
beforeVars = evalin('base', 'whos');
% Run the script
evalin('base', 'anesthesia_model_AMICAS');
% Capture the state of the workspace after running the script
afterVars = evalin('base', 'whos');
% Identify the new variables introduced by the script
newVars = setdiff({afterVars.name}, {beforeVars.name});
% Get detailed information for new variables
anesthesia_vars = afterVars(ismember({afterVars.name}, newVars));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parallel Computing Setup
parpool; % Start parallel pool

for index = 1:noOfPatients % Parallelize over patients
    index
    patient = Patients(index);

    %% Choose input values
    %Propofol_value = 0.05 + (0.0005*patient.age^2+0.3*(patient.bmi/40)^3+0.4*exp(patient.lbm/40))/25 + normrnd(0,0.1);
    %Propofol_local = max([0.05 Propofol_value]);       % [mg/kg/min] 

    Remifentanil = patient.Remifentanil; %binornd(1,0.3) * unifrnd(0,0.4); % [ug/kg/min]  
    Atracurium = patient.Atracurium; %binornd(1,0.3) * unifrnd(0,29.5);       % [ug/kg/min] 
    Dopamine = patient.Dopamine; %binornd(1,0.3) * unifrnd(0,20);         % [ug/kg/min] 
    SNP = patient.SNP; %binornd(1,0.3) * unifrnd(0,10);              % [ug/kg/min]

    inputStepR = [zeros(1,10) Remifentanil * ones(1,Tsim)]; %% 
    inputStepA = [zeros(1,10) Atracurium * ones(1,Tsim)]; %% 
    inputStepD = [zeros(1,10) Dopamine * ones(1,Tsim)]; %% 
    inputStepS = [zeros(1,10) SNP * ones(1,Tsim)]; %% 

    Reminput=timeseries(inputStepR); 
    Atracinput=timeseries(inputStepA); 
    Dopinput=timeseries(inputStepD); 
    SNPinput=timeseries(inputStepS); 

    % Create batch simulation inputs
    simInputs = Simulink.SimulationInput.empty(propofol_loops, 0);

    loop_id_prop = 1;

    simInputs = [];

    for loop_id = 1:propofol_loops
        Propofol = propofol_loop_vector(loop_id);
        %loop_id_prop+propofol_loops*(index-1)

        %Remifentanil = 0;     % [ug/kg/min] 
        %Atracurium = 0;       % [ug/kg/min] 
        %Dopamine = 0;         % [ug/kg/min] 
        %SNP = 0;              % [ug/kg/min]

        %initialize_inputs;

        inputStepP = [zeros(1,10) Propofol * ones(1,Tsim)]; %% 

        % input profiles
        Propinput=timeseries(inputStepP); 

        %% run simulation for every patient
    
        % Initialise patient dependent model
        anesthesia_model_AMICAS   % anesthetic models
        

        % Initialize simulation input object
        simIn = Simulink.SimulationInput('simulator_AMICAS_M2022a.slx');
        
        % Set simulation parameters and variables
        simIn = simIn.setVariable('Propinput', Propinput);
        simIn = simIn.setVariable('Reminput', Reminput);
        simIn = simIn.setVariable('Atracinput', Atracinput);
        simIn = simIn.setVariable('Dopinput', Dopinput);
        simIn = simIn.setVariable('SNPinput', SNPinput);
        simIn = simIn.setVariable('Tsim', Tsim);
        simIn = simIn.setVariable('Ts', Ts);
        simIn = simIn.setVariable('ST', ST);

        simIn = simIn.setVariable('Dist_type', Dist_type);
        simIn = simIn.setVariable('Anest_loop', Anest_loop);
        simIn = simIn.setVariable('RSM_type', RSM_type);
        simIn = simIn.setVariable('patient', patient);

        all_vars = [anesthesia_vars; hemodyn_vars; dist_model_vars];

        for i = 1:length(all_vars)
            varName = all_vars(i).name;
            varValue = evalin('base', varName); % Get the variable value from the base workspace
            simIn = simIn.setVariable(varName, varValue); % Add it to simIn
        end

        
        simInputs = [simInputs, simIn];
    
    end

    % Run Batch Simulations
    simOuts = parsim(simInputs, 'ShowProgress', 'on');

    % Collect Results for This Patient
    BIS_patient = []; % Temporary storage for BIS
    inner_loop_id_patient = []; % Temporary storage for loop IDs


    % Collect Results
    for k = 1:length(simOuts)
        if simOuts(k).ErrorMessage
            disp(['Simulation error for patient ', num2str(index), ', dose ', num2str(k)]);
            continue;
        end
        % Store BIS and other outputs
        simOutput = simOuts(k);
        BIS_patient = [BIS_patient, simOutput.BIS];
        inner_loop_id_patient = [inner_loop_id_patient, ...
                                 repelem(k + length(propofol_loop_vector) * (index - 1), length(simOutput.BIS.Time))];
    end

    BIS_all_results{index} = BIS_patient;
    inner_loop_id_results{index} = inner_loop_id_patient;

end

% Combine results from all iterations after the parfor loop
BIS_all = horzcat(BIS_all_results{:});
inner_loop_id = horzcat(inner_loop_id_results{:});

% Assuming 2 columns of data
numCols = noOfPatients;

% Extract Time (assuming all entries share the same Time vector)
timeData = BIS_all(1).Time; % Use the Time from the first entry

% Extract Data for all patients and store it in a matrix or cell array
dataMatrix = arrayfun(@(x) BIS_all(x).Data, 1:noOfPatients*propofol_loops, 'UniformOutput', false);
dataMatrix = horzcat(dataMatrix{:}); % Combine into a single matrix

% Convert wide-format to long-format
% Create a PatientID column
[PatientID, Time, BISData] = deal([]); % Initialize arrays for long format
for i = 1:noOfPatients*propofol_loops
    PatientID = [PatientID; repmat(ceil(i/propofol_loops), size(timeData, 1), 1)];
    Time = [Time; timeData];
    BISData = [BISData; dataMatrix(:, i)];
end

% Create the long-format table
BIS_csv_table = table(PatientID, Time, BISData, transpose(inner_loop_id), 'VariableNames', {'PatientID', 'Time', 'BISData', 'inner_loop'});

writetable(BIS_csv_table, 'BIS_counterfactual.csv');

