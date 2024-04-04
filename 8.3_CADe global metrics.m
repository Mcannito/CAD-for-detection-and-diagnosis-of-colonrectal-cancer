clc
clear all
close all

load('newDataStructure.mat')

%% Classe

% Specifica il percorso del tuo file Excel
file_path = 'Database.xlsx';

% Estrai i dati dalle colonne desiderate (ad esempio, colonne A e B)
data = xlsread(file_path, 'pazienti', 'A:B');

% Crea un vettore classe con due colonne
vettore_excel = [data(:, 1), data(:, 2)];

classe = [data(:, 1)];

%% ESTRAZIONE BOUNDING BOXES
bounding_boxes_all = struct();

for i = 1:length(newDataStructure)
    id_paziente = newDataStructure(i).FolderName;
    mask = newDataStructure(i).Mask;
   
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

%% Estrazione features 

% Nascondi tutti gli avvisi
warning('off');

% Inizializza tumor_data come matrice vuota invece che cell array
auto_tumor_data = [];

% Secondo ciclo for per il calcolo delle ROI all'interno delle bounding boxes
for i = 1:length(newDataStructure)
    id_paziente = str2double(newDataStructure(i).FolderName);
    mask = newDataStructure(i).Mask;
    t2 = newDataStructure(i).T2;
    adc = newDataStructure(i).ADC;
    [height, width, num_slices] = size(mask);

    % Estrai la maschera binaria del tumore
    tumor_mask = mask;

    for slice = 1:num_slices
        bounding_box = bounding_boxes_all(i).BoundingBoxes{slice};

        if ~isempty(bounding_box)
            roi_x = round(bounding_box(1));
            roi_y = round(bounding_box(2));
            roi_width = round(bounding_box(3));
            roi_height = round(bounding_box(4));

            % Verifica se le coordinate di ROI superano le dimensioni di t2 e adc
            if roi_y + roi_height - 1 > height || roi_x + roi_width - 1 > width
                % Gestisci l'errore assegnando 0 come valore speciale
                current_roi_data = zeros(1, 47);
                current_roi_data(1) = convertCharsToStrings(id_paziente);
                current_roi_data(2) = slice;
                auto_tumor_data = [auto_tumor_data; current_roi_data];

            else
                % Crea una maschera binaria basata sulla bounding box
                roi_mask = tumor_mask(roi_y : roi_y + roi_height - 1, roi_x : roi_x + roi_width - 1, slice);

                % Applica la maschera alla T2 e all'ADC
                current_ROI_T2 = t2(roi_y : roi_y + roi_height - 1, roi_x : roi_x + roi_width - 1, slice) .* roi_mask;
                current_ROI_T2(roi_mask == 0) = NaN;

                current_ROI_ADC = adc(roi_y : roi_y + roi_height - 1, roi_x : roi_x + roi_width - 1, slice) .* roi_mask;
                current_ROI_ADC(roi_mask == 0) = NaN;


                % Calcola GrayLimits e NumLevels per l'immagine T2
                min_val_t2 = min(current_ROI_T2(:));
                max_val_t2 = max(current_ROI_T2(:));
                GrayLimits_T2 = [min_val_t2, max_val_t2];
                NumLevels_T2 = 16;

                % Calcola GrayLimits e NumLevels per l'immagine ADC
                min_val_adc = min(current_ROI_ADC(:));
                max_val_adc = max(current_ROI_ADC(:));
                GrayLimits_ADC = [min_val_adc, max_val_adc];
                NumLevels_ADC = 16;

                % Calcola le caratteristiche di primo ordine per l'immagine T2
                mean_val_t2 = nanmean(current_ROI_T2(:));
                std_dev_val_t2 = nanstd(double(current_ROI_T2(:)));
                kurtosis_val_t2 = kurtosis(current_ROI_T2(:));
                skewness_val_t2 = skewness(current_ROI_T2(:));

                % Estrai le caratteristiche di texture per l'immagine T2
                features_t2 = TextureFeat(current_ROI_T2, GrayLimits_T2, NumLevels_T2);

                % Calcola le caratteristiche di primo ordine per l'immagine ADC
                mean_val_adc = nanmean(current_ROI_ADC(:));
                std_dev_val_adc = nanstd(double(current_ROI_ADC(:)));
                kurtosis_val_adc = kurtosis(current_ROI_ADC(:));
                skewness_val_adc = skewness(current_ROI_ADC(:));

                % Estrai le caratteristiche di texture per l'immagine ADC
                features_adc = TextureFeat(current_ROI_ADC, GrayLimits_ADC, NumLevels_ADC);

                % Assegna le caratteristiche alle posizioni corrispondenti di current_roi_data
                current_roi_data=zeros(1,47);
                current_roi_data(1) = id_paziente;
                current_roi_data(2) = slice;                        
                current_roi_data(3) = mean_val_t2;
                current_roi_data(4) = std_dev_val_t2;
                current_roi_data(5) = kurtosis_val_t2;
                current_roi_data(6) = skewness_val_t2;
                current_roi_data(7:24) = features_t2;
                current_roi_data(25) = mean_val_adc;
                current_roi_data(26) = std_dev_val_adc;
                current_roi_data(27) = kurtosis_val_adc;
                current_roi_data(28) = skewness_val_adc;
                current_roi_data(29:46) = features_adc;


                % Aggiungi la classe da vettore_excel
                idx = find(vettore_excel(:, 2) == id_paziente);

                if ~isempty(idx)
                    current_roi_data(47) = vettore_excel(idx, 1);
                else
                    current_roi_data(47) = 0; % Assegna 0 se non trovi l'id paziente corrispondente
                end

                auto_tumor_data = [auto_tumor_data; current_roi_data];                
               
            end
        end
    end
end

auto_tumor_data = sortrows(auto_tumor_data, 1);

%% PLOT DI ESEMPIO
% Extract information for the first patient
first_patient_boxes = bounding_boxes_all(62).BoundingBoxes;
num_slices_first_patient = length(first_patient_boxes);

% Load the first patient's data
first_patient_id = bounding_boxes_all(62).PatientID;
first_patient_index = 62;
first_patient_data = newDataStructure(first_patient_index);
first_patient_mask = double(first_patient_data.Mask);

% Plot each slice with its bounding box in a separate figure
for slice = 1:num_slices_first_patient
    figure;
    
    % Display the original image
    imshow(first_patient_mask(:, :, slice), []);
    title(['Patient ' num2str(first_patient_id) ' - Slice ' num2str(slice)]);
    
    % Overlay bounding box on the image
    hold on;
    bounding_box = first_patient_boxes{slice};
    if ~isempty(bounding_box)
        rectangle('Position', bounding_box, 'EdgeColor', 'r', 'LineWidth', 2);
    end
    hold off;
end

%% Normalizzazione tumor_data
% Trova le dimensioni della matrice roi_data
[row_count, col_count] = size(auto_tumor_data);

% Inizializza una copia della matrice roi_data normalizzata
auto_tumor_data_normalized = auto_tumor_data;

% Normalizza le colonne tranne la quinta fino alla penultima
for col = 3:(col_count-1) % Esclude le prime 2 colonne e l'ultima colonna
    % Trova il minimo e il massimo per la colonna corrente
    col_min = min(auto_tumor_data(:, col));
    col_max = max(auto_tumor_data(:, col));

    % Normalizza i valori della colonna corrente
    auto_tumor_data_normalized(:, col) = (auto_tumor_data(:, col) - col_min) / (col_max - col_min);
end

% Restituisci la matrice con le prime 4 colonne e l'ultima invariate
auto_tumor_data_normalized = [auto_tumor_data(:, 1:2), auto_tumor_data_normalized(:, 3:(end-1)), auto_tumor_data(:, end)];

% Salvataggio dei dati normalizzati in un file
save('tumor_data_normalized.mat', 'auto_tumor_data_normalized');
save('tumor_data.mat', 'tumor_data');

%% NaN
% Trova il numero di valori NaN nella matrice
numero_nan = sum(sum(isnan(auto_tumor_data_normalized)));

% Visualizza il numero di valori NaN
disp(['Il numero di valori NaN nella matrice è: ', num2str(numero_nan)]);

% Trova le posizioni dei valori NaN nella matrice
[row_nan, col_nan] = find(isnan(auto_tumor_data_normalized));

% Mostra le righe che contengono i valori NaN e la decisione di eliminarle
disp("Le ROI contenenti i valori NaN sono:");  
disp(row_nan);    



disp("Decidiamo di rimuovere le righe contenenti i valori NaN.");

%% OUTLIERS CON ISOUTLIER sulle RIGHE 
% Identify outliers in the specific columns
outliers = isoutlier(auto_tumor_data(:, 2:46), 'quartiles'); 

% Conta il numero di '1' per ogni riga
count_ones = sum(outliers == 1, 2);

% Calcola il numero massimo di outliers consentiti per una riga
outliers_consentiti = round(0.5* size(auto_tumor_data(:,2:38), 2)); 

% Trova le righe con più del 50% di outliers
roi_outlier = find(count_ones > outliers_consentiti);


% Calcola il numero totale di '1' nelle righe
numero_totali_di_uno = sum(count_ones);
disp(['Il numero totale di outliers nel dataset è: ', num2str(numero_totali_di_uno)]);


%% OUTLIERS CON ISOUTLIER SULLE COLONNE
% Inizializza un vettore per la somma degli '1' per ogni colonna
somma_degli_1_per_colonna = zeros(1, size(auto_tumor_data(:, 2:46), 2));

% Inizializza un vettore per la percentuale di outliers per ogni colonna
percentuale_outliers_per_colonna = zeros(1, size(auto_tumor_data(:, 2:38), 2));

% Calcola la somma degli '1' e la percentuale di outliers per ogni colonna
for i = 1:size(auto_tumor_data(:, 2:38), 2)
    % Trova gli outliers nella colonna corrente
    outliers_colonna_corrente = isoutlier(auto_tumor_data(:, i), 'quartiles');

    % Calcola la somma degli '1' per la colonna corrente
    somma_degli_1_per_colonna(i) = sum(outliers_colonna_corrente);

    % Calcola la percentuale di outliers per la colonna corrente
    percentuale_outliers_per_colonna(i) = (sum(outliers_colonna_corrente) / size(auto_tumor_data, 1)) * 100;
end


save('tumor_data_cleaned.mat', 'tumor_data_cleaned')


%% APPLICAZIONE MODELLO 

% Specifica il percorso del tuo file Excel 
file_path = 'Database.xlsx'; 
 
% Estrai i dati dalle colonne desiderate (ad esempio, colonne A e B) 
data = xlsread(file_path, 'pazienti', 'A:B'); 
 
% Crea un vettore classe con due colonne 
vettore_excel = [data(:, 1), data(:, 2)]; 
  
% Estrai la prima colonna come vettore di ID paziente 
id_pazienti = auto_tumor_data(:, 1); 
 
% Trova gli ID pazienti unici 
pazienti_unici = unique(id_pazienti);

% Filtra vettore_excel solo per i pazienti presenti in pazienti_unici 
indice_pazienti = ismember(vettore_excel(:, 2), pazienti_unici); 
vettore_filtrato = vettore_excel(indice_pazienti, :); 

classe_vera = vettore_filtrato(:,1);
%% prova
load('net_9.mat')
load('RFModel_5.mat')
% Assumi che roi_data_normalized contenga le features, e l'ultima colonna contenga le classi
colonne = [37, 4, 6, 16, 44]; 
features = auto_tumor_data_normalized(:, colonne);

% Usa la rete neurale per fare previsioni
previsioni = predict(RFModel_5, features);
previsioni = str2double(previsioni);


% Rimuovi l'ultima colonna da validation_fs_8
auto_tumor_data_senzacol = auto_tumor_data_normalized(:, 1:end-1);

% Aggiungi y_val_pred_knn_5 come nuova ultima colonna
auto_tumor_data_normalized = [auto_tumor_data_senzacol, previsioni];

% Estrai la prima colonna come vettore di ID paziente
id_pazienti = auto_tumor_data_normalized(:, 1);

% Trova gli ID pazienti unici
pazienti_unici = unique(id_pazienti);

%% FETTA MAGGIORE
bounding_boxes_all = struct();

for i = 1:length(newDataStructure)
    id_paziente = newDataStructure(i).FolderName;
    mask = newDataStructure(i).Mask;
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
for i = 1:length(newDataStructure)
    id_paziente = newDataStructure(i).FolderName;
    mask = newDataStructure(i).Mask;
    t2 = newDataStructure(i).T2;
    adc = newDataStructure(i).ADC;
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
    idx = find(auto_tumor_data_normalized(:, 1) == paziente_corrente & ...
               auto_tumor_data_normalized(:, 2) == fetta_corrente);

    % Se viene trovata una corrispondenza, assegna la classe
    if ~isempty(idx)
        classe_corrente = auto_tumor_data_normalized(idx, end);
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
