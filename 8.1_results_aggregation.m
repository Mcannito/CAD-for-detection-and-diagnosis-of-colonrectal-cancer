%% LAB 8

clc
clear all
close all

load("RFModel_5.mat")
load("y_val_pred_rf_5.mat")
load('validation_fs_5.mat')
% Rimuovi l'ultima colonna da validation_fs_8
validation_fs_5_senza_ultima_colonna = validation_fs_5(:, 1:end-1);

% Aggiungi y_val_pred_knn_5 come nuova ultima colonna
validation_set_rf_5 = [validation_fs_5_senza_ultima_colonna, y_val_pred_rf_5];

% Estrai la prima colonna come vettore di ID paziente
id_pazienti = validation_set_rf_5(:, 1);

% Trova gli ID pazienti unici
pazienti_unici = unique(id_pazienti);

%% Considerare solo la fetta centrale

fette = validation_set_rf_5(:, 2);

% Inizializza il vettore per contare le fette per ogni paziente
num_fette_per_paziente = zeros(size(pazienti_unici));

% Itera attraverso ciascun paziente
for i = 1:length(pazienti_unici)
    % Trova le fette corrispondenti al paziente corrente
    fette_paziente = fette(id_pazienti == pazienti_unici(i));

    % Conta il numero di fette crescenti per il paziente corrente
    num_fette_per_paziente(i) = sum(diff(fette_paziente) > 0) + 1;
end

% Inizializza la cella per memorizzare le fette risultanti
fette_risultanti = cell(length(pazienti_unici), 1);

% Itera attraverso ciascun paziente
for i = 1:length(pazienti_unici)
    % Trova le fette corrispondenti al paziente corrente
    fette_paziente = fette(id_pazienti == pazienti_unici(i));
    
    % Se il numero di fette è dispari, prendi la fetta centrale
    if mod(num_fette_per_paziente(i), 2) == 1
        indice_fetta_centrale = ceil(num_fette_per_paziente(i) / 2);
        fette_risultanti{i} = fette_paziente(indice_fetta_centrale);
    else
        % Se il numero di fette è pari, prendi entrambe le fette centrali
        indice_fette_centrali = [num_fette_per_paziente(i) / 2, num_fette_per_paziente(i) / 2 + 1];
        fette_risultanti{i} = fette_paziente(indice_fette_centrali);
    end
end

% classifico paziente basandomi sulla fetta centrale se fette dispari, su
% fette centrali se pari: se una fetta vale 0 e l'altra 1 lo classifico
% come 1, situazione più sicura


% Inizializza la cella per memorizzare le classi risultanti
classi_risultanti_fetta_centrale = cell(length(pazienti_unici), 1);

% Itera attraverso ciascun paziente
for i = 1:length(pazienti_unici)
    % Trova le classi corrispondenti al paziente corrente
    classi_paziente = validation_set_rf_5(id_pazienti == pazienti_unici(i), end);
    
    % Se il numero di fette è dispari, prendi la classe della fetta centrale
    if mod(num_fette_per_paziente(i), 2) == 1
        indice_fetta_centrale = ceil(num_fette_per_paziente(i) / 2);
        classi_risultanti_fetta_centrale{i} = classi_paziente(indice_fetta_centrale);
    else
        % Se il numero di fette è pari, valuta le condizioni per assegnare la classe
        if all(classi_paziente == 0)
            % Se entrambe le fette hanno classe 0, assegna 0
            classi_risultanti_fetta_centrale{i} = 0;
        else
            % Altrimenti, assegna 1
            classi_risultanti_fetta_centrale{i} = 1;
        end
    end
end

% Converti la cella delle classi in un array
classi_risultanti_fetta_centrale = cell2mat(classi_risultanti_fetta_centrale);
classi_risultanti_fetta_centrale=[pazienti_unici classi_risultanti_fetta_centrale];


%% Maggioranza delle fette

% Inizializza la cella per memorizzare le classi risultanti
classi_risultanti_maggioranza_fette = zeros(length(pazienti_unici), 1);

% Itera attraverso ciascun paziente
for i = 1:length(pazienti_unici)
    % Trova le classi corrispondenti al paziente corrente
    classi_paziente = validation_set_rf_5(id_pazienti == pazienti_unici(i), end);
    
    % Conta il numero di fette associate a ciascuna classe
    conteggio_classi = histcounts(classi_paziente, [0, 1, 2]); % Se le classi sono 0 e 1
    
    % Trova la classe con il maggior numero di fette
    [~, indice_classe_massima] = max(conteggio_classi);
    
    % Assegna al paziente la classe con il maggior numero di fette
    classi_risultanti_maggioranza_fette(i) = indice_classe_massima - 1; % Sottrai 1 solo se necessario
end
classi_risultanti_maggioranza_fette=[pazienti_unici classi_risultanti_maggioranza_fette];

%% Considero solo risultato della fetta più grande
load("construction_set.mat")
bounding_boxes_all = struct();

for i = 1:length(construction_set)
    id_paziente = construction_set(i).FolderName;
    mask = construction_set(i).Mask;
    mask=double(mask);

    [height, width, num_slices] = size(mask);

    bounding_boxes_patient = cell(num_slices,1);

    for slice = 1:num_slices
        props = regionprops(mask(:, :, slice), 'BoundingBox');
        if ~isempty(props)
            bounding_box = props.BoundingBox;
            
            bounding_boxes_patient{slice} = bounding_box;
        end
    end

    bounding_boxes_all(i).PatientID = id_paziente;
    bounding_boxes_all(i).BoundingBoxes = bounding_boxes_patient;
end



dim_roi = [];

% Secondo ciclo for per il calcolo delle ROI all'interno delle bounding boxes
for i = 1:length(construction_set)
    id_paziente = construction_set(i).FolderName;
    mask = construction_set(i).Mask;
    t2 = construction_set(i).T2;
    adc = construction_set(i).ADC;
    [height, width, num_slices] = size(mask);

    % Estrai la maschera binaria del tumore
    tumor_mask = double(mask);

    for slice = 1:num_slices
        bounding_box = bounding_boxes_all(i).BoundingBoxes{slice};

        if ~isempty(bounding_box)
            roi_x = round(bounding_box(1));
            roi_y = round(bounding_box(2));
            roi_width = round(bounding_box(3));
            roi_height = round(bounding_box(4));

                % Crea una maschera binaria basata sulla bounding box
                roi_mask = tumor_mask(roi_y : roi_y + roi_height - 1, roi_x : roi_x + roi_width - 1, slice);

                % Applica la maschera alla T2 e all'ADC
                current_ROI_T2 = t2(roi_y : roi_y + roi_height - 1, roi_x : roi_x + roi_width - 1, slice) .* roi_mask;
                current_ROI_T2(roi_mask == 0) = NaN;

                area_roi=roi_width*roi_height;

                current_ROI_ADC = adc(roi_y : roi_y + roi_height - 1, roi_x : roi_x + roi_width - 1, slice) .* roi_mask;
                current_ROI_ADC(roi_mask == 0) = NaN;
                current_roi_data = zeros(1, 7);
                current_roi_data(1) = convertCharsToStrings(id_paziente);
                current_roi_data(2) = slice;
                current_roi_data(3) = roi_x;
                current_roi_data(4) = roi_y;
                current_roi_data(5) = roi_height;
                current_roi_data(6) = roi_width;
                current_roi_data(7) = area_roi;

                dim_roi = [dim_roi; current_roi_data];

        end
    end
end
%save("dim_roi.mat", "dim_roi")

 % prelevo da dim_roi solo i dati relativi ai pazienti presenti nel
 % validation, valuto per ogni paziente quale roi risulta più grande e
 % classifico paziente in base alla classe di quella roi

pazienti_tot=unique(dim_roi(:,1));

% Inizializza una matrice per memorizzare le informazioni sulla fetta con area maggiore per ciascun paziente
info_fette_maggiori = [];

% Itera sui pazienti unici
for i = 1:length(pazienti_unici)
    paziente = pazienti_unici(i);

    % Trova indici delle righe corrispondenti al paziente corrente
    indici_paziente = find(dim_roi(:, 1) == paziente);

    % Trova indice della riga con area massima
    [~, indice_fetta_maggiore] = max(dim_roi(indici_paziente, 3));

    % Ottieni le informazioni sulla fetta con area maggiore per il paziente corrente
    info_fetta_maggiore = dim_roi(indici_paziente(indice_fetta_maggiore), :);

    % Aggiungi le informazioni alla matrice
    info_fette_maggiori = [info_fette_maggiori; info_fetta_maggiore];
end

% Inizializza una matrice per memorizzare i risultati
classi_risultanti_fetta_maggiore = zeros(size(info_fette_maggiori, 1), 2);

% Itera sulle righe in info_fette_maggiori
for i = 1:size(info_fette_maggiori, 1)
    % Estrai informazioni paziente e fetta
    paziente_corrente = info_fette_maggiori(i, 1);
    fetta_corrente = info_fette_maggiori(i, 2);

    % Stampa i valori per il debugging
    fprintf('Paziente corrente: %d, Fetta corrente: %d\n', paziente_corrente, fetta_corrente);

    % Trova la riga corrispondente in validation_set_knn_8
    idx = find(validation_set_rf_5(:, 1) == paziente_corrente & ...
               validation_set_rf_5(:, 2) == fetta_corrente);

    % Se viene trovata una corrispondenza, assegna la classe
    if ~isempty(idx)
        classe_corrente = validation_set_rf_5(idx, end);
        % Memorizza le informazioni in classi_risultanti_fetta_maggiore
        classi_risultanti_fetta_maggiore(i, :) = [paziente_corrente, classe_corrente];
    end
end

%% Maggioranza pesata delle fette



% valuto area totale del tumore per ogni pazinete sommando l'area delle
% singole fette del pazinete e poi divido area singola fetta per area
% totale, ottengo così una percentuale per ogni fetta. A questo punto
% valuto quale percentuale ha classe 1 e quale classe 0, classifico
% paziente in base a percentuale maggiore

% Identificatori dei pazienti nel set di validazione
pazienti_validazione = validation_set_rf_5;

% Trova gli indici delle righe in dim_roi corrispondenti ai pazienti nel set di validazione
indici_pazienti_validazione = ismember(dim_roi(:, 1), pazienti_validazione);

% Estrai solo le righe corrispondenti ai pazienti nel set di validazione
dim_roi_validazione = dim_roi(indici_pazienti_validazione, :);

% Trova gli indici delle righe in dim_roi_validazione corrispondenti ai pazienti unici
indici_pazienti_unici = ismember(dim_roi_validazione(:, 1), pazienti_unici);

% Estrai solo le righe corrispondenti ai pazienti unici
dim_roi_ridotta = dim_roi_validazione(indici_pazienti_unici, :);

dim_roi_ridotta=dim_roi_ridotta(:,[1 2 end]);

% Estrai le colonne paziente fette e area dalla matrice
pazienti = dim_roi_ridotta(:, 1);
fette = dim_roi_ridotta(:, 2);
aree = dim_roi_ridotta(:, 3);

% Usa la funzione accumarray per ottenere l'area totale per ogni paziente
somma_area = accumarray(pazienti, aree, [], @sum);

% Crea una matrice 25x1 con pazienti unici e aree totali
area_totale_per_paziente = [pazienti_unici, somma_area(pazienti_unici)];

% Calcola la percentuale di area per ogni fetta
percentuali_area = aree ./ somma_area(pazienti);

% Crea la matrice risultante con paziente, fetta e percentuale di area
percentuale_fetta = [pazienti, fette, percentuali_area];

% Voglio classificare basandomi su percentuale area 

percentuale_fetta=sortrows(percentuale_fetta,1);

class=validation_set_rf_5(:,end);

pesi_fette=[percentuale_fetta class];

% Ordina la matrice in base all'identificatore del paziente (colonna 1)
pesi_fette_ordinati = sortrows(pesi_fette, 1);

risultati_per_paziente = [];

% Ciclo attraverso ogni paziente
pazienti_unici = unique(pesi_fette_ordinati(:, 1));
for i = 1:length(pazienti_unici)
    paziente_corrente = pazienti_unici(i);
    
    % Estrai le righe corrispondenti al paziente corrente
    righe_paziente = pesi_fette_ordinati(pesi_fette_ordinati(:, 1) == paziente_corrente, :);
    
    % Somma le percentuali di area per ogni classe
    somma_percentuali_classe_1 = sum(righe_paziente(righe_paziente(:, 4) == 1, 3));
    somma_percentuali_classe_0 = sum(righe_paziente(righe_paziente(:, 4) == 0, 3));
    % Aggiungi i risultati alla matrice risultati_per_paziente
    risultati_per_paziente = [risultati_per_paziente; paziente_corrente, somma_percentuali_classe_1, somma_percentuali_classe_0];
end

classi_risultanti_maggioranza_pesata=zeros(25,2);
classi_risultanti_maggioranza_pesata(:,1)=pazienti_unici;
for i=1:length(risultati_per_paziente)
    if risultati_per_paziente(i,2)>risultati_per_paziente(i,3)
        classi_risultanti_maggioranza_pesata(i,2)=1;
    else
        classi_risultanti_maggioranza_pesata(i,2)=0;
    end
end




%% Classe vera 
 
% Specifica il percorso del tuo file Excel 
file_path = 'Database.xlsx'; 
 
% Estrai i dati dalle colonne desiderate (ad esempio, colonne A e B) 
data = xlsread(file_path, 'pazienti', 'A:B'); 
 
% Crea un vettore classe con due colonne 
vettore_excel = [data(:, 1), data(:, 2)]; 
 
classe = [data(:, 1)]; 
 
% Filtra vettore_excel solo per i pazienti presenti in pazienti_unici 
indice_pazienti = ismember(vettore_excel(:, 2), pazienti_unici); 
vettore_filtrato = vettore_excel(indice_pazienti, :); 
 
%% Classe predetta maggioranza pesata 
classe_predetta = classi_risultanti_maggioranza_pesata(:, 2); 
 
% Supponiamo che la classe vera sia nella prima colonna di vettore_filtrato 
classe_vera = vettore_filtrato(:, 1); 
 
% Calcolare le metriche 
confusion_mat = confusionmat(classe_vera, classe_predetta); 
precision = confusion_mat(2, 2) / sum(confusion_mat(:, 2)); 
recall = confusion_mat(2, 2) / sum(confusion_mat(2, :)); 
f1_score = 2 * (precision * recall) / (precision + recall); 
 
accuracy = sum(diag(confusion_mat)) / sum(confusion_mat(:)); 
 
% Calcolare il NPV (Negative Predictive Value) 
npv = confusion_mat(1, 1) / sum(confusion_mat(1, :)); 
 
disp('--- Performance Metrics maggioranza pesata---'); 
fprintf('Precision: %.4f\n', precision); 
fprintf('Recall: %.4f\n', recall); 
fprintf('F1 Score: %.4f\n', f1_score); 
fprintf('Accuracy: %.4f\n', accuracy); 
fprintf('NPV (Negative Predictive Value): %.4f\n', npv); 
 
%% Classe predetta fetta maggiore 
classe_predetta = classi_risultanti_fetta_maggiore(:, 2); 
 
% Supponiamo che la classe vera sia nella prima colonna di vettore_filtrato 
classe_vera = vettore_filtrato(:, 1); 
 
% Calcolare le metriche 
confusion_mat = confusionmat(classe_vera, classe_predetta); 
precision = confusion_mat(2, 2) / sum(confusion_mat(:, 2)); 
recall = confusion_mat(2, 2) / sum(confusion_mat(2, :)); 
f1_score = 2 * (precision * recall) / (precision + recall); 
 
accuracy = sum(diag(confusion_mat)) / sum(confusion_mat(:)); 
 
% Calcolare il NPV (Negative Predictive Value) 
npv = confusion_mat(1, 1) / sum(confusion_mat(1, :)); 
 
disp('--- Performance Metrics fetta maggiore ---'); 
fprintf('Precision: %.4f\n', precision); 
fprintf('Recall: %.4f\n', recall); 
fprintf('F1 Score: %.4f\n', f1_score); 
fprintf('Accuracy: %.4f\n', accuracy); 
fprintf('NPV (Negative Predictive Value): %.4f\n', npv); 
 
 
%% Classe predetta fetta centrale 
classe_predetta = classi_risultanti_fetta_centrale(:, 2); 
 
% Supponiamo che la classe vera sia nella prima colonna di vettore_filtrato 
classe_vera = vettore_filtrato(:, 1); 
 
% Calcolare le metriche 
confusion_mat = confusionmat(classe_vera, classe_predetta); 
precision = confusion_mat(2, 2) / sum(confusion_mat(:, 2)); 
recall = confusion_mat(2, 2) / sum(confusion_mat(2, :)); 
f1_score = 2 * (precision * recall) / (precision + recall); 
 
accuracy = sum(diag(confusion_mat)) / sum(confusion_mat(:)); 
 
% Calcolare il NPV (Negative Predictive Value) 
npv = confusion_mat(1, 1) / sum(confusion_mat(1, :)); 
 
disp('--- Performance Metrics fetta centrale ---'); 
fprintf('Precision: %.4f\n', precision); 
fprintf('Recall: %.4f\n', recall); 
fprintf('F1 Score: %.4f\n', f1_score); 
fprintf('Accuracy: %.4f\n', accuracy); 
fprintf('NPV (Negative Predictive Value): %.4f\n', npv); 
 
%% Classe predetta maggioranza fette 
classe_predetta = classi_risultanti_maggioranza_fette(:, 2); 
 
% Supponiamo che la classe vera sia nella prima colonna di vettore_filtrato 
classe_vera = vettore_filtrato(:, 1); 
 
% Calcolare le metriche 
confusion_mat = confusionmat(classe_vera, classe_predetta); 
precision = confusion_mat(2, 2) / sum(confusion_mat(:, 2)); 
recall = confusion_mat(2, 2) / sum(confusion_mat(2, :)); 
f1_score = 2 * (precision * recall) / (precision + recall); 
 
accuracy = sum(diag(confusion_mat)) / sum(confusion_mat(:)); 
 
% Calcolare il NPV (Negative Predictive Value) 
npv = confusion_mat(1, 1) / sum(confusion_mat(1, :)); 
 
disp('--- Performance Metrics maggioranza fette ---'); 
fprintf('Precision: %.4f\n', precision); 
fprintf('Recall: %.4f\n', recall); 
fprintf('F1 Score: %.4f\n', f1_score); 
fprintf('Accuracy: %.4f\n', accuracy); 
fprintf('NPV (Negative Predictive Value): %.4f\n', npv);


