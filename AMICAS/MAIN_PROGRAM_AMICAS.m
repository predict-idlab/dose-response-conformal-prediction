                            % MAIN PROGRAM FOR BENCHMARK PATIENT SIMULATOR
                                       % Ghent University, MAY 2024
                                       % ERC AMICAS 2022-2027

clearvars; clear memory; close all; clc;

% variable initialization
BIS_all= []; propofol_all = []; Remifentanil_all = []; Atracurium_all = []; Dopamine_all = []; SNP_all = [];

Ts = 1200/60; % sampling time 5 minutes (30/60 min)
ST = 0.1*Ts; % simulation step size 

% initialize patient independent models
hemodyn_model_AMICAS      % hemodynamic models
dist_model_AMICAS         % disturbance models

%% Choose patient database -> database_type = 1 (for 12 patient database - young patients), database_type = 2 (for 24 patient database - old patients)
database_type = 1;
Patients = initialize_patients(database_type);

%[noOfPatients,~] = size(Patients)
noOfPatients = 1 ;

%% Choose simulation time (min) - See user manual
Tsim = 300; 

for index = 1 : noOfPatients
    index;

    patient = Patients(index);

    %% Choose input values
    Propofol_value = 0.05 + (0.0005*patient.age^2+0.3*(patient.bmi/40)^3+0.4*exp(patient.lbm/40))/25 + normrnd(0,0.1);
    Propofol = 0.15 ;%max([0.05 Propofol_value]);       % [mg/kg/min] 
    %Remifentanil = 0;     % [ug/kg/min] 
    %Atracurium = 0;       % [ug/kg/min] 
    %Dopamine = 0;         % [ug/kg/min] 
    %SNP = 0;              % [ug/kg/min]

    Remifentanil = 0; %binornd(1,0.3) * unifrnd(0,0.4);    % [ug/kg/min]  
    Atracurium = 0;% binornd(1,0.3) * unifrnd(0,29.5);     % [ug/kg/min] 
    Dopamine = 0;% binornd(1,0.3) * unifrnd(0,20);         % [ug/kg/min] 
    SNP = 0;% binornd(1,0.3) * unifrnd(0,10);              % [ug/kg/min]
    
    initialize_inputs;
    
    %% Choose surface model type ->  RSM_type=1 (for Greco) / RSM_type=2 (for Minto) / RSM_type=3 (for Reduced Greco)
    RSM_type = 1;
    
    %% Choose disturbance profile -> See user manual
    Dist_type = 1;
    
    %% Choose anesthesiologist in loop or not -> Anest_loop = 1 / Anest_loop = 2 
    Anest_loop = 1;
    
    %% run simulation for every patient

    

    % Initialise patient dependent model
    anesthesia_model_AMICAS   % anesthetic models
    
    % run simulation
    sim('simulator_AMICAS_M2022a.slx');
    
    % save variables
    BIS_all = [BIS_all, BIS];
    propofol_all = [propofol_all, Propofol];
    Remifentanil_all = [Remifentanil_all, Remifentanil];
    Atracurium_all = [Atracurium_all, Atracurium];
    Dopamine_all = [Dopamine_all, Dopamine];
    SNP_all = [SNP_all, SNP];
    %RASS_all = [RASS_all, RASS]; 
    %CO_all = [CO_all, CO]; 
    %MAP_all = [MAP_all, MAP]; 
    %NMB_all = [NMB_all, NMB];
    
end
% Assuming 2 columns of data
numCols = noOfPatients;

% Extract Time (assuming all entries share the same Time vector)
timeData = BIS_all(1).Time; % Use the Time from the first entry

% Extract Data for all patients and store it in a matrix or cell array
dataMatrix = arrayfun(@(x) BIS_all(x).Data, 1:noOfPatients, 'UniformOutput', false);
dataMatrix = horzcat(dataMatrix{:}); % Combine into a single matrix

% Convert wide-format to long-format
% Create a PatientID column
[PatientID, Time, BISData] = deal([]); % Initialize arrays for long format
for i = 1:noOfPatients
    PatientID = [PatientID; repmat(i, size(timeData, 1), 1)];
    Time = [Time; timeData];
    BISData = [BISData; dataMatrix(:, i)];
end

% Create the long-format table
BIS_csv_table = table(PatientID, Time, BISData, 'VariableNames', {'PatientID', 'Time', 'BISData'});

% Extract attributes dynamically (assuming Patients is a struct array)
attributeNames = fieldnames(Patients(1:noOfPatients)); % List of attributes in Patients
for i = 1:numel(attributeNames)
    attribute = attributeNames{i}; % Current attribute name
    attributeValues = arrayfun(@(x) x.(attribute), Patients); % Extract values
    
    % Expand values to match rows in longTable
    expandedValues = arrayfun(@(id) attributeValues(id), BIS_csv_table.PatientID);
    
    % Add the attribute as a new column in longTable
    BIS_csv_table.(attribute) = expandedValues;
end


% Expand values to match rows in longTable
expandedValues = arrayfun(@(id) propofol_all(id), BIS_csv_table.PatientID);
% Add the propofol as a new column in longTable
BIS_csv_table.('Propofol') = expandedValues;

% Expand values to match rows in longTable
expandedValues = arrayfun(@(id) Remifentanil_all(id), BIS_csv_table.PatientID);
% Add the propofol as a new column in longTable
BIS_csv_table.('Remifentanil') = expandedValues;

% Expand values to match rows in longTable
expandedValues = arrayfun(@(id) Atracurium_all(id), BIS_csv_table.PatientID);
% Add the propofol as a new column in longTable
BIS_csv_table.('Atracurium') = expandedValues;

% Expand values to match rows in longTable
expandedValues = arrayfun(@(id) Dopamine_all(id), BIS_csv_table.PatientID);
% Add the propofol as a new column in longTable
BIS_csv_table.('Dopamine') = expandedValues;

% Expand values to match rows in longTable
expandedValues = arrayfun(@(id) SNP_all(id), BIS_csv_table.PatientID);
% Add the propofol as a new column in longTable
BIS_csv_table.('SNP') = expandedValues;


% Adding the other variables

%for i = 1:noOfPatients
%BIS_csv_table.(['Propofol']) = repmat(Propofol, height(BIS_csv_table), 1); % Add propofol

BIS_csv_table.(['Dist_type']) = repmat(Dist_type, height(BIS_csv_table), 1); % Add propofol

%writetable(BIS_csv_table, 'BIS.csv');

%% Results - use plotValues(signal, type, safetyLimits - bool, showLegend - bool)

figure; 
subplot(3,2,1)
plotValues(BIS_all, OutputSignalType.BIS, false, false);
%subplot(3,2,2) 
%plotValues(RASS_all, OutputSignalType.RASS, false, false);
%subplot(3,2,3) 
%plotValues(CO_all, OutputSignalType.CO, false, false);
%subplot(3,2,4)
%plotValues(MAP_all, OutputSignalType.MAP, false, false);
%subplot(3,2,5) 
%plotValues(NMB_all, OutputSignalType.NMB, false, false);


