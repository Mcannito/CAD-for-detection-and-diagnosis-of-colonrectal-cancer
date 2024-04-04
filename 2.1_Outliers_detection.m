%% LAB 2 - VALUTAZIONE OUTLIERS

clc
clear all
close all


load('test_roi_data_normalized.mat')




%% NaN

% Trova il numero di valori NaN nella matrice
numero_nan = sum(sum(isnan(test_roi_data_normalized)));

% Visualizza il numero di valori NaN
disp(['Il numero di valori NaN nella matrice è: ', num2str(numero_nan)]);

% Trova le posizioni dei valori NaN nella matrice
[row_nan, col_nan] = find(isnan(test_roi_data_normalized));

% Mostra le righe che contengono i valori NaN e la decisione di eliminarle
disp("Le ROI contenenti i valori NaN sono:");  %%%3 NaN  sono nella stessa riga il che vuol dire che eliminiamo 4 ROI
disp(row_nan);    

% Decisione di rimuovere le righe con valori NaN
test_roi_data_cleaned = test_roi_data_normalized;
test_roi_data_cleaned(row_nan, :) = [];

disp("Decidiamo di rimuovere le righe contenenti i valori NaN.");





%% OUTLIERS CON ISOUTLIER sulle RIGHE - boxplot 
% Identify outliers in the specific columns
outliers = isoutlier(test_roi_data_cleaned(:, 5:48), 'quartiles'); %%Usiamo il boxplot per trovare gli outliers

% Conta il numero di '1' per ogni riga
count_ones = sum(outliers == 1, 2);

% Calcola il numero massimo di outliers consentiti per una riga
outliers_consentiti = round(0.5 * size(test_roi_data_cleaned, 2)); 

% Trova le righe con più del 75% di outliers
roi_outlier = find(count_ones > outliers_consentiti);

% Rimuovi le righe con più del 75% di outliers dalla matrice
%roi_data_cleaned(roi_outlier, :) = [];

% Calcola il numero totale di '1' nelle righe
numero_totali_di_uno = sum(count_ones);
disp(['Il numero totale di outliers nel dataset è: ', num2str(numero_totali_di_uno)]);

% Conta il numero di righe rimosse
%numero_righe_rimosse = length(roi_outlier);
%disp(['Il numero totale di righe rimosse è: ', num2str(numero_righe_rimosse)]);

%% OUTLIERS CON ISOUTLIER SULLE COLONNE
% Inizializza un vettore per la somma degli '1' per ogni colonna
somma_degli_1_per_colonna = zeros(1, size(test_roi_data_cleaned(:, 5:48), 2));

% Inizializza un vettore per la percentuale di outliers per ogni colonna
percentuale_outliers_per_colonna = zeros(1, size(test_roi_data_cleaned(:, 5:48), 2));

% Calcola la somma degli '1' e la percentuale di outliers per ogni colonna
for i = 1:size(test_roi_data_cleaned(:, 5:48), 2)
    % Trova gli outliers nella colonna corrente
    outliers_colonna_corrente = isoutlier(test_roi_data_cleaned(:, i), 'quartiles');

    % Calcola la somma degli '1' per la colonna corrente
    somma_degli_1_per_colonna(i) = sum(outliers_colonna_corrente);

    % Calcola la percentuale di outliers per la colonna corrente
    percentuale_outliers_per_colonna(i) = (sum(outliers_colonna_corrente) / size(test_roi_data_cleaned, 1)) * 100;
    
   
end



save('test_roi_data_cleaned.mat', 'test_roi_data_cleaned') %%NOTARE CHE CONTIENE SOLO 44 COLONNE, CI SONO SOLO LE VARIABILI DI INTERESSE E LA CLASSE