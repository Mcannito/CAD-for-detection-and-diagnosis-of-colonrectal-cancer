%% Lab 08 test

clc
clear all
close all

load("test_tumor_data_cleaned.mat")
load('test_set.mat')

load('RFModel_5.mat')
%% Classe vera 
test_tumor_data_cleaned = test_data_cleaned;

% Specifica il percorso del tuo file Excel 
file_path = 'Database.xlsx'; 
 
% Estrai i dati dalle colonne desiderate (ad esempio, colonne A e B) 
data = xlsread(file_path, 'pazienti', 'A:B'); 
 
% Crea un vettore classe con due colonne 
vettore_excel = [data(:, 1), data(:, 2)]; 
  
% Estrai la prima colonna come vettore di ID paziente 
id_pazienti = test_tumor_data_cleaned(:, 1); 
 
% Trova gli ID pazienti unici 
pazienti_unici = unique(id_pazienti);

% Filtra vettore_excel solo per i pazienti presenti in pazienti_unici 
indice_pazienti = ismember(vettore_excel(:, 2), pazienti_unici); 
vettore_filtrato = vettore_excel(indice_pazienti, :); 

classe_vera = vettore_filtrato(:,1);

%% Predizioni modello  

test_tumor_data_cleaned = sortrows(test_tumor_data_cleaned, 1);

% Assumi che roi_data_normalized contenga le features, e l'ultima colonna contenga le classi
colonne = [39, 5, 8, 18, 46]; 
features = test_tumor_data_cleaned(:, colonne);

% Usa la rete neurale per fare previsioni
previsioni = predict(RFModel_5, features);
previsioni = str2double(previsioni);

% Rimuovi l'ultima colonna da validation_fs_8
test_tumor_data_cleaned_senza_colonna = test_tumor_data_cleaned(:, 1:end-1);

% Aggiungi y_val_pred_knn_5 come nuova ultima colonna
test_tumor_data_cleaned = [test_tumor_data_cleaned_senza_colonna, previsioni];

% Estrai la prima colonna come vettore di ID paziente
id_pazienti = test_tumor_data_cleaned(:, 1);

% Trova gli ID pazienti unici
pazienti_unici = unique(id_pazienti);

%% FETTA MAGGIORE
bounding_boxes_all = struct();

for i = 1:length(test_set)
    id_paziente = test_set(i).FolderName;
    mask = test_set(i).Mask;
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
for i = 1:length(test_set)
    id_paziente = test_set(i).FolderName;
    mask = test_set(i).Mask;
    t2 = test_set(i).T2;
    adc = test_set(i).ADC;
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
    idx = find(test_tumor_data_cleaned(:, 1) == paziente_corrente & ...
               test_tumor_data_cleaned(:, 2) == fetta_corrente);

    % Se viene trovata una corrispondenza, assegna la classe
    if ~isempty(idx)
        classe_corrente = test_tumor_data_cleaned(idx, end);
        % Memorizza le informazioni in classi_risultanti_fetta_maggiore
        classi_risultanti_fetta_maggiore(i, :) = [paziente_corrente, classe_corrente];
    end
end


%% Classe predetta fetta maggiore 
classe_predetta = classi_risultanti_fetta_maggiore(:, 2); 
 
 
 
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





