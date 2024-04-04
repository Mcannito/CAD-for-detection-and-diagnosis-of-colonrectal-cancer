%% lab 2 snellito
clc
clear all
close all

load('test_set.mat');
%% plot esempio foto
% Estrai le informazioni dal primo paziente
patientData = test_set(1);

% Numero di fette del primo paziente
numSlices = size(patientData.Mask, 3);

% Per ogni fetta, crea una figura con le 3 immagini
for sliceIndex = 1:numSlices
    % Crea una nuova figura
    figure('Name', ['Fetta ', num2str(sliceIndex)], 'NumberTitle', 'off');
    
    % Immagine Mask
    subplot(1,3,1);
    imshow(patientData.Mask(:,:,sliceIndex), []);
    title('Mask');
    
    % Immagine T2
    subplot(1,3,2);
    imshow(patientData.T2(:,:,sliceIndex), []);
    title('T2');
    
    % Immagine ADC
    subplot(1,3,3);
    imshow(patientData.ADC(:,:,sliceIndex), []);
    title('ADC');
    
    
end
%% BOUNDING BOXES

bounding_boxes_all = struct();

for i = 1:length(test_set)
    id_paziente = test_set(i).FolderName;
    mask = test_set(i).Mask;
    [height, width, num_slices] = size(mask);

    bounding_boxes_patient = cell(num_slices,1);

    for slice = 1:num_slices
        props = regionprops(mask(:, :, slice), 'BoundingBox');
        if ~isempty(props)
            bounding_box = props.BoundingBox;
            bounding_box_old = bounding_box;
            amplification_factor = 0.4;
            bounding_box(1) = bounding_box(1) - bounding_box(3) * amplification_factor;
            bounding_box(2) = bounding_box(2) - bounding_box(4) * amplification_factor;
            bounding_box(3) = bounding_box(3) * (1 + 2 * amplification_factor);
            bounding_box(4) = bounding_box(4) * (1 + 2 * amplification_factor);
            bounding_boxes_patient{slice} = bounding_box;
        end
    end

    bounding_boxes_all(i).PatientID = id_paziente;
    bounding_boxes_all(i).BoundingBoxes = bounding_boxes_patient;
end

id_paziente = construction_set(8).FolderName;
mask = construction_set(8).Mask;
[height, width, num_slices] = size(mask);

bounding_boxes_patient = cell(num_slices,1);

for slice = 1:num_slices
    props = regionprops(mask(:, :, slice), 'BoundingBox');
    if ~isempty(props)
        bounding_box_old = props.BoundingBox;
        amplification_factor = 0.4;

        % Creazione di una nuova figura per ogni slice
        figure;

        % Plot della maschera con la vecchia bounding box
        imshow(mask(:, :, slice));
        title(['Patient ID: ', num2str(id_paziente), ' - Slice: ', num2str(slice)]);
        hold on;

        % Plot della vecchia bounding box
        if ~isempty(bounding_box_old) && length(bounding_box_old) == 4
            rectangle('Position', bounding_box_old, 'EdgeColor', 'r', 'LineWidth', 2);
        end

        % Calcolo delle nuove coordinate della bounding box
        if ~isempty(bounding_box_old) && length(bounding_box_old) == 4
            x = bounding_box_old(1) - bounding_box_old(3) * amplification_factor;
            y = bounding_box_old(2) - bounding_box_old(4) * amplification_factor;
            width = bounding_box_old(3) * (1 + 2 * amplification_factor);
            height = bounding_box_old(4) * (1 + 2 * amplification_factor);
            bounding_box_new = [x, y, width, height];

            % Plot della nuova bounding box
            rectangle('Position', bounding_box_new, 'EdgeColor', 'g', 'LineWidth', 2);
        end

        hold off;

        bounding_boxes_patient{slice} = bounding_box_old;
    end
end





%% ROI DATA CON ESTRAZIONE FEATURES
% Inizializza roi_data come matrice vuota invece che cell array
test_roi_data = [];



% Secondo ciclo for per il calcolo delle ROI all'interno delle bounding boxes
for i = 1:length(test_set)
    id_paziente = test_set(i).FolderName;
    mask = test_set(i).Mask;
    t2 = test_set(i).T2;
    adc = test_set(i).ADC;
    [height, width, num_slices] = size(mask);
    % Preallocazione delle dimensioni t2 e adc
    [t2_height, t2_width, t2_slices] = size(t2);
    [adc_height, adc_width, adc_slices] = size(adc);

    for slice = 1:num_slices
        
        bounding_box = bounding_boxes_all(i).BoundingBoxes{slice};

        if ~isempty(bounding_box)
            box_width = bounding_box(3);
            box_height = bounding_box(4);
            roi_width = 5;
            roi_height = 5;

            num_ROI_width = floor(box_width / roi_width);
            num_ROI_height = floor(box_height / roi_height);

            for row = 1:num_ROI_height
                for col = 1:num_ROI_width
                    roi_x = round(bounding_box(1) + (col - 1) * roi_width);
                    roi_y = round(bounding_box(2) + (row - 1) * roi_height);

                    % Verifica se le coordinate di ROI superano le dimensioni di t2 e adc
                    if roi_y + roi_height - 1 > t2_height || roi_x + roi_width - 1 > t2_width || ...
                        roi_y + roi_height - 1 > adc_height || roi_x + roi_width - 1 > adc_width
                        % Gestisci l'errore assegnando 0 come valore speciale
                        current_roi_data = zeros(1, 49);
                        current_roi_data(1) = id_paziente;
                        current_roi_data(2) = slice;
                        current_roi_data(3) = roi_x;
                        current_roi_data(4) = roi_y;
                        test_roi_data = [test_roi_data; current_roi_data];
                    else
                        current_ROI_T2 = t2(roi_y : roi_y + roi_height - 1, roi_x : roi_x + roi_width - 1, slice);
                        current_ROI_ADC = adc(roi_y : roi_y + roi_height - 1, roi_x : roi_x + roi_width - 1, slice);

                        % Calcola GrayLimits e NumLevels per l'immagine T2
                        min_val_t2 = min(current_ROI_T2, [], 'all');
                        max_val_t2 = max(current_ROI_T2, [], 'all');
                        GrayLimits_T2 = [min_val_t2, max_val_t2];
                        NumLevels_T2 = 16;

                        % Calcola GrayLimits e NumLevels per l'immagine ADC
                        min_val_adc = min(current_ROI_ADC, [], 'all');
                        max_val_adc = max(current_ROI_ADC, [], 'all');
                        GrayLimits_ADC = [min_val_adc, max_val_adc];
                        NumLevels_ADC = 16;

                        % Calcola le caratteristiche di primo ordine per l'immagine T2
                        mean_val_t2 = mean(current_ROI_T2(:));
                        std_dev_val_t2 = std(double(current_ROI_T2(:)));
                        kurtosis_val_t2 = kurtosis(current_ROI_T2(:));
                        skewness_val_t2 = skewness(current_ROI_T2(:));

                        % Estrai le caratteristiche di texture per l'immagine T2
                        features_t2 = TextureFeat(current_ROI_T2, GrayLimits_T2, NumLevels_T2);

                        % Calcola le caratteristiche di primo ordine per l'immagine ADC
                        mean_val_adc = mean(current_ROI_ADC(:));
                        std_dev_val_adc = std(double(current_ROI_ADC(:)));
                        kurtosis_val_adc = kurtosis(current_ROI_ADC(:));
                        skewness_val_adc = skewness(current_ROI_ADC(:));

                        % Estrai le caratteristiche di texture per l'immagine ADC
                        features_adc = TextureFeat(current_ROI_ADC, GrayLimits_ADC, NumLevels_ADC);

                        % Assegna le caratteristiche alle posizioni corrispondenti di current_roi_data
                        current_roi_data=zeros(1,49);
                        current_roi_data(1) = id_paziente;
                        current_roi_data(2) = slice;
                        current_roi_data(3) = roi_x;
                        current_roi_data(4) = roi_y;
                        current_roi_data(5) = mean_val_t2;
                        current_roi_data(6) = std_dev_val_t2;
                        current_roi_data(7) = kurtosis_val_t2;
                        current_roi_data(8) = skewness_val_t2;
                        current_roi_data(9:26) = features_t2;
                        current_roi_data(27) = mean_val_adc;
                        current_roi_data(28) = std_dev_val_adc;
                        current_roi_data(29) = kurtosis_val_adc;
                        current_roi_data(30) = skewness_val_adc;
                        current_roi_data(31:48) = features_adc;

                        % Calcola il numero totale di pixel e di pixel del tumore nella ROI
                        current_roi_mask = mask(roi_y:roi_y + roi_height - 1, roi_x:roi_x + roi_width - 1, slice);
                        total_roi_pixels = numel(current_roi_mask);
                        tumor_roi_pixels = sum(current_roi_mask, 'all');

                        % Calcola le proporzioni
                        tumor_roi_percentage = tumor_roi_pixels / total_roi_pixels;
                        non_tumor_roi_percentage = 1 - tumor_roi_percentage;

                        % Assegna la classe in base alle proporzioni calcolate
                        if tumor_roi_percentage >= 0.9
                            roi_class = 1; % Tumore
                        elseif non_tumor_roi_percentage >= 0.9
                            roi_class = 0; % Non tumore
                        else
                            roi_class = 2; % Bordo
                        end

                        % Aggiungi i dati della ROI alla matrice roi_data
                        current_roi_data(49) = roi_class; % Aggiungi la classe
                        test_roi_data = [test_roi_data; current_roi_data];
                    end
                end
            end
        end
    end
    
end

% Assuming the first patient is construction_set(1)
id_paziente = construction_set(1).FolderName;
mask = construction_set(1).Mask;
t2 = construction_set(1).T2;
adc = construction_set(1).ADC;
[height, width, num_slices] = size(mask);

slice = 11;

bounding_box = bounding_boxes_all(1).BoundingBoxes{slice};

if ~isempty(bounding_box)
    % Plot the mask with the bounding box and ROIs
    figure;
    subplot(1, 3, 1);
    imshow(mask(:, :, slice));
    hold on;
    rectangle('Position', bounding_box, 'EdgeColor', 'r', 'LineWidth', 2);

    % Overlay ROIs on the mask
    box_width = bounding_box(3);
    box_height = bounding_box(4);
    roi_width = 5;
    roi_height = 5;
    num_ROI_width = floor(box_width / roi_width);
    num_ROI_height = floor(box_height / roi_height);

    for row = 1:num_ROI_height
        for col = 1:num_ROI_width
            roi_x = round(bounding_box(1) + (col - 1) * roi_width);
            roi_y = round(bounding_box(2) + (row - 1) * roi_height);
            rectangle('Position', [roi_x, roi_y, roi_width, roi_height], 'EdgeColor', 'g', 'LineWidth', 1);
        end
    end

    title(['Patient ID: ', num2str(id_paziente), ' - Slice: ', num2str(slice)]);

    % Plot the T2 image
    subplot(1, 3, 2);
    imshow(t2(:, :, slice), []);
    title('T2 Image');

    % Overlay ROIs on the T2 image
    for row = 1:num_ROI_height
        for col = 1:num_ROI_width
            roi_x = round(bounding_box(1) + (col - 1) * roi_width);
            roi_y = round(bounding_box(2) + (row - 1) * roi_height);
            rectangle('Position', [roi_x, roi_y, roi_width, roi_height], 'EdgeColor', 'g', 'LineWidth', 1);
        end
    end

    % Plot the ADC image
    subplot(1, 3, 3);
    imshow(adc(:, :, slice), []);
    title('ADC Image');

    % Overlay ROIs on the ADC image
    for row = 1:num_ROI_height
        for col = 1:num_ROI_width
            roi_x = round(bounding_box(1) + (col - 1) * roi_width);
            roi_y = round(bounding_box(2) + (row - 1) * roi_height);
            rectangle('Position', [roi_x, roi_y, roi_width, roi_height], 'EdgeColor', 'g', 'LineWidth', 1);
        end
    end

    hold off;
end








%% NORMALIZZAZIONE DELLE FEATURES
% Trova le dimensioni della matrice roi_data
[row_count, col_count] = size(test_roi_data);

% Inizializza una copia della matrice roi_data normalizzata
test_roi_data_normalized = test_roi_data;

% Normalizza le colonne tranne la quinta fino alla penultima
for col = 5:(col_count-1) % Esclude le prime 4 colonne e l'ultima colonna
    % Trova il minimo e il massimo per la colonna corrente
    col_min = min(test_roi_data(:, col));
    col_max = max(test_roi_data(:, col));

    % Normalizza i valori della colonna corrente
    test_roi_data_normalized(:, col) = (test_roi_data(:, col) - col_min) / (col_max - col_min);
end

% Restituisci la matrice con le prime 4 colonne e l'ultima invariate
test_roi_data_normalized = [test_roi_data(:, 1:4), test_roi_data_normalized(:, 5:(end-1)), test_roi_data(:, end)];


% Salvataggio dei dati normalizzati in un file
save('test_roi_data_normalized.mat', 'test_roi_data_normalized');
save('test_roi_data.mat', 'test_roi_data')



