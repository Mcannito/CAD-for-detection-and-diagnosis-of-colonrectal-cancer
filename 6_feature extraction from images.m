%% Lab 6

clc
clear all
close all

% Nascondi tutti gli avvisi
warning('off');

load('construction_set.mat');
% Convert FolderName field to numeric array and sort the structure
[~, idx] = sort(cell2mat({construction_set.FolderName}));
construction_set = construction_set(idx);



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

% Extract information for the first patient
first_patient_boxes = bounding_boxes_all(62).BoundingBoxes;
num_slices_first_patient = length(first_patient_boxes);

% Load the first patient's data
first_patient_id = bounding_boxes_all(62).PatientID;
first_patient_index = 62;
first_patient_data = construction_set(first_patient_index);
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


%% Estrazione features 
% Inizializza tumor_data come matrice vuota invece che cell array
tumor_data = [];

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

            % Verifica se le coordinate di ROI superano le dimensioni di t2 e adc
            if roi_y + roi_height - 1 > height || roi_x + roi_width - 1 > width
                % Gestisci l'errore assegnando 0 come valore speciale
                current_roi_data = zeros(1, 39);
                current_roi_data(1) = convertCharsToStrings(id_paziente);
                current_roi_data(2) = slice;
                tumor_data = [tumor_data; current_roi_data];

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

                % Estrai le caratteristiche di texture per l'immagine T2
                features_t2 = TextureFeat(current_ROI_T2, GrayLimits_T2, NumLevels_T2);

                % Estrai le caratteristiche di texture per l'immagine ADC
                features_adc = TextureFeat(current_ROI_ADC, GrayLimits_ADC, NumLevels_ADC);

                % Assegna le caratteristiche alle posizioni corrispondenti di current_roi_data
                current_roi_data = zeros(1, 39);
                current_roi_data(1) = convertCharsToStrings(id_paziente);
                current_roi_data(2) = slice;
                current_roi_data(3:20) = features_t2;
                current_roi_data(21:38) = features_adc;

                % Aggiungi la classe da vettore_excel
                idx = find(vettore_excel(:, 2) == id_paziente);

                if ~isempty(idx)
                    current_roi_data(39) = vettore_excel(idx, 1);
                else
                    current_roi_data(39) = 0; % Assegna 0 se non trovi l'id paziente corrispondente
                end

                tumor_data = [tumor_data; current_roi_data];                
               
            end
        end
    end
end

tumor_data = sortrows(tumor_data, 1);

% Calcola le dimensioni della maschera
[mask_height, mask_width] = size(roi_mask);

% Ridimensiona le immagini T2 e ADC
resized_ROI_T2 = imresize(current_ROI_T2, [mask_height, mask_width]);
resized_ROI_ADC = imresize(current_ROI_ADC, [mask_height, mask_width]);

% Visualizza le immagini
figure;

% Visualizza la maschera nella prima colonna
subplot(1, 3, 1);
imshow(roi_mask);
axis image;
title('Mask Slice 30 - Paziente 2049');

% Visualizza l'immagine T2 nella seconda colonna
subplot(1, 3, 2);
imshow(resized_ROI_T2, 'Colormap', gray);
axis image;
colorbar;
title('Current ROI T2');

% Visualizza l'immagine ADC nella terza colonna
subplot(1, 3, 3);
imshow(resized_ROI_ADC, 'Colormap', gray);
axis image;
colorbar;
title('Current ROI ADC');

%% Normalizzazione tumor_data
% Trova le dimensioni della matrice roi_data
[row_count, col_count] = size(tumor_data);

% Inizializza una copia della matrice roi_data normalizzata
tumor_data_normalized = tumor_data;

% Normalizza le colonne tranne la quinta fino alla penultima
for col = 3:(col_count-1) % Esclude le prime 2 colonne e l'ultima colonna
    % Trova il minimo e il massimo per la colonna corrente
    col_min = min(tumor_data(:, col));
    col_max = max(tumor_data(:, col));

    % Normalizza i valori della colonna corrente
    tumor_data_normalized(:, col) = (tumor_data(:, col) - col_min) / (col_max - col_min);
end

% Restituisci la matrice con le prime 4 colonne e l'ultima invariate
tumor_data_normalized = [tumor_data(:, 1:2), tumor_data_normalized(:, 3:(end-1)), tumor_data(:, end)];

% Salvataggio dei dati normalizzati in un file
save('tumor_data_normalized.mat', 'tumor_data_normalized');
save('tumor_data.mat', 'tumor_data');

%% NaN
% Trova il numero di valori NaN nella matrice
numero_nan = sum(sum(isnan(tumor_data_normalized)));

% Visualizza il numero di valori NaN
disp(['Il numero di valori NaN nella matrice è: ', num2str(numero_nan)]);

% Trova le posizioni dei valori NaN nella matrice
[row_nan, col_nan] = find(isnan(tumor_data_normalized));

% Mostra le righe che contengono i valori NaN e la decisione di eliminarle
disp("Le ROI contenenti i valori NaN sono:");  
disp(row_nan);    

% Decisione di rimuovere le righe con valori NaN, che è la fetta 14 del
% paziente 44
tumor_data_cleaned = tumor_data_normalized;
tumor_data_cleaned(row_nan, :) = [];

disp("Decidiamo di rimuovere le righe contenenti i valori NaN.");

%% OUTLIERS CON ISOUTLIER sulle RIGHE 
% Identify outliers in the specific columns
outliers = isoutlier(tumor_data_cleaned(:, 2:38), 'quartiles'); 

% Conta il numero di '1' per ogni riga
count_ones = sum(outliers == 1, 2);

% Calcola il numero massimo di outliers consentiti per una riga
outliers_consentiti = round(0.5* size(tumor_data_cleaned(:,2:38), 2)); 

% Trova le righe con più del 50% di outliers
roi_outlier = find(count_ones > outliers_consentiti);


% Calcola il numero totale di '1' nelle righe
numero_totali_di_uno = sum(count_ones);
disp(['Il numero totale di outliers nel dataset è: ', num2str(numero_totali_di_uno)]);


%% OUTLIERS CON ISOUTLIER SULLE COLONNE
% Inizializza un vettore per la somma degli '1' per ogni colonna
somma_degli_1_per_colonna = zeros(1, size(tumor_data_cleaned(:, 2:38), 2));

% Inizializza un vettore per la percentuale di outliers per ogni colonna
percentuale_outliers_per_colonna = zeros(1, size(tumor_data_cleaned(:, 2:38), 2));

% Calcola la somma degli '1' e la percentuale di outliers per ogni colonna
for i = 1:size(tumor_data_cleaned(:, 2:38), 2)
    % Trova gli outliers nella colonna corrente
    outliers_colonna_corrente = isoutlier(tumor_data_cleaned(:, i), 'quartiles');

    % Calcola la somma degli '1' per la colonna corrente
    somma_degli_1_per_colonna(i) = sum(outliers_colonna_corrente);

    % Calcola la percentuale di outliers per la colonna corrente
    percentuale_outliers_per_colonna(i) = (sum(outliers_colonna_corrente) / size(tumor_data_cleaned, 1)) * 100;
end


save('tumor_data_cleaned.mat', 'tumor_data_cleaned')



