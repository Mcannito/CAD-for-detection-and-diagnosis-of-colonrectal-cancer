
%LAB 01 DATA SCIENCE 
clc 
clear all 
close all 
 
% PUNTO 1 
 
%Specifica il percorso della cartella che vuoi aprire 
directory = 'C:\Users\miche\Desktop\DSM\RettiLAB\RettiLAB'; 
%  
% % Utilizza la funzione 'dir' per ottenere una lista dei file all'interno della cartella 
fileList = dir(directory); 
dataStructure = struct('FolderName', {}, 'Mask', {}, 'T2', {}, 'ADC', {}); 
 
% % Ciclo for per scorrere ogni cartella 
 for i = 3:length(fileList) % Inizia da 3 per saltare le cartelle . e .. 
    currentFolder = fileList(i).name; 
    if fileList(i).isdir % Controlla se è una cartella 
         currentDir = fullfile(directory, currentFolder); 
         filesInFolder = dir(fullfile(currentDir, '*.nii')); % Ottieni i file con estensione .nii 
  
         currentData.FolderName = currentFolder; % Salva il nome della cartella  
         % Inizializza i campi Mask, T2 e ADC 
         currentData.Mask = []; 
         currentData.T2 = []; 
         currentData.ADC = []; 
 
        for j = 1:length(filesInFolder) % Ciclo per leggere ciascun file .nii 
             currentFile = filesInFolder(j).name; 
             fullPath = fullfile(currentDir, currentFile); 
             niiData = niftiread(fullPath); % Leggi il file .nii 
  
             disp(['Lettura di: ', currentFile]); % Stampa il nome del file per il debug 
  
% Determina il campo in cui inserire i dati in base al nome del file 
             if contains(currentFile, 'mask') 
                 currentData.Mask = niiData; 
            elseif contains(currentFile, 'T2') 
                 currentData.T2 = niiData; 
             elseif contains(currentFile, 'adc') 
                 currentData.ADC = niiData; 
             end 
         end 
%  
         disp(['mask: ', num2str(isempty(currentData.Mask))]); % Stampa se il campo Mask è vuoto 
         disp(['adc: ', num2str(isempty(currentData.ADC))]); % Stampa se il campo ADC è vuoto 
         % Aggiungi la struttura corrente alla struttura principale 
         dataStructure(end + 1) = currentData; 
     end 
 end 

save('dataStructure.mat', 'dataStructure', '-v7.3') 

load('dataStructure.mat')

%Carico matrice normalizzata 
load('dataStructure_norm.mat')
     
% Inizializza le matrici per i risultati 
results_norm = struct('Patient', {}, 'MeanT2', {}, 'StandardDeviationT2', {}, 'Mean_adc', {}, 'StandardDeviation_adc', {}); 
 
% Ciclo su tutti gli elementi della struttura 
for k = 1:numel(dataStructure_norm) 
    currentPatient = dataStructure_norm(k); 
    mask = dataStructure_norm(k).Mask; % La prima immagine è la maschera binaria 
    t2Image = dataStructure_norm(k).T2; 
    adcImage = dataStructure_norm(k).ADC; 
    mask = double(mask); 
 
    % Moltiplica l'immagine per la maschera binaria 
    maskedT2 = t2Image .* mask; 
    maskedADC = adcImage .* mask; 
 
    % Calcola la media e la deviazione standard 
    maskedT2(maskedT2 == 0) = NaN; % Imposta i valori esterni all'area desiderata su NaN per non influenzare le statistiche
    maskedADC(maskedADC == 0) = NaN;
    meanT2 = mean(maskedT2(:), 'omitnan');
    stdT2 = std(maskedT2(:), 'omitnan');
    meanADC = mean(maskedADC(:), 'omitnan');
    stdADC = std(maskedADC(:), 'omitnan');
 % Salva i risultati nella struttura 
    results_norm(k).Patient = currentPatient.FolderName; 
    results_norm(k).MeanT2 = meanT2; 
    results_norm(k).StandardDeviationT2 = stdT2; 
    results_norm(k).Mean_adc = meanADC; 
    results_norm(k).StandardDeviation_adc = stdADC; 
end

% Converto stringa che indica pz in numero 
for i = 1:length(results_norm) 
    results_norm(i).Patient= str2double(results_norm(i).Patient); 
end 
 
% Estrai i valori dalla prima colonna 
valori = [results_norm.Patient]; 
 
% Ordina gli indici in base ai valori della prima colonna 
[~, indici_ordinati] = sort(valori); 
 
% Riordina la struttura in base agli indici ordinati 
results_norm = results_norm(indici_ordinati); 
 
% Salva i risultati in un file mat 
save('results_norm.mat', 'results_norm', '-v7.3'); 
 
%Divido pz in centri: 
% Centro A: 5-231 (1-41) 
% Centro A: 1054-1171 (42-63) 
% Centro A: 2002-2049 (64-89) 
 
% Estrai le tre strutture separate 
centroA = results_norm(1:41); 
centroB = results_norm(42:63); 
centroC = results_norm(64:89); 
 
% Salva i risultati in un file mat 
save('centroA.mat', 'centroA', '-v7.3'); 
save('centroB.mat', 'centroB', '-v7.3'); 
save('centroC.mat', 'centroC', '-v7.3'); 
% PUNTO 2 
 
% Faccio clustering con Kmeans per valutare possibilità di fare block 
% sampling  
 
% Clustering con Kmeans 
 
% Preparazione dei dati per il clustering 
X = [];  % Inizializza una matrice vuota per i dati 
 
% Estrai e appiattisci le immagini da results_norm 
for i = 1:numel(results_norm) 
    mT2 = results_norm(i).MeanT2; 
    stdT2 = results_norm(i).StandardDeviationT2; 
    madc = results_norm(i).Mean_adc; 
    stdadc = results_norm(i).StandardDeviation_adc; 
    % Inserisci i dati in X 
    X = [X; mT2, stdT2, madc, stdadc]; 
end 
 
% Specifica il numero di cluster desiderato (K) 
K = 3;   % uno per ogni centro 
 
% Esegui il clustering K-Means 
[idx, centroids] = kmeans(X, K); 
 
% Visualizza i risultati del clustering 
% idx conterrà le etichette dei cluster assegnate a ciascun dato 
% e centroids conterrà i centroidi dei cluster
cl1=find(idx==1); 
cl2=find(idx==2); 
cl3=find(idx==3); 
 
% Non possiamo fare block samling in quanto non c'è corrispondenza nella 
% divisione tra cluster e centri, possiamo divididere tra conbstruction e 
% test set prendendo da ogni cluster il 70% dei pz per il construction e il 
% 30% per il test set: faccio sampling per clustering