clc
clear all
close all

load('roi_data_cleaned.mat')

%% Calcola le occorrenze di ciascun valore
last_col = roi_data_cleaned(:,end);

values = unique(last_col);
count = histcounts(last_col, [values; max(values)+1]);

% Calcola le percentuali
percentuali = count / (length(last_col)) * 100;

% Visualizza i risultati
disp("Percentuale delle classi rispetto al totale:");
for i = 1:length(values)
    disp(['Classe ', num2str(values(i)), ': ', num2str(percentuali(i)), '%']);
end

% Supponendo che l'ultima colonna contenga la classe
classe_colonna = size(roi_data_cleaned, 2);

% Trova gli indici delle righe con classe 0
indici_classe_0 = find(roi_data_cleaned(:, classe_colonna) == 0);

% Seleziona casualmente un numero di righe di classe 0 per mantenere
percentuale_righe_da_mantenere = 0.4;
num_righe_da_mantenere = round(percentuale_righe_da_mantenere * length(indici_classe_0));
indici_righe_da_mantenere = randsample(indici_classe_0, num_righe_da_mantenere);

% Creazione di una matrice con solo le righe di classe 0 selezionate casualmente
roi_data_classe_0_mantenute = roi_data_cleaned(indici_righe_da_mantenere, :);

% Rimuovi le righe di classe 0 selezionate dalla matrice originale
roi_data_cleaned(indici_righe_da_mantenere, :) = [];
roi_data_cleaned_undersampled = roi_data_cleaned;
% Salva le matrici modificate
save('roi_data_cleaned_undersampled.mat', 'roi_data_cleaned_undersampled');
save('roi_data_classe_0_mantenute.mat', 'roi_data_classe_0_mantenute');

y = roi_data_cleaned_undersampled(:, end); 
% Verifica il bilanciamento delle classi 
num_campioni_classe_0 = sum(y == 0); % la classe maggioritaria 0 
num_campioni_classe_1 = sum(y == 1); % la classe minoritaria 1

perc0 = num_campioni_classe_0/length(y)*100;
perc1 = num_campioni_classe_1/length(y)*100;

% Mostra a schermo le percentuali
fprintf('Percentuale di campioni per la classe 0: %.2f%%\n', perc0);
fprintf('Percentuale di campioni per la classe 1: %.2f%%\n', perc1);


%%rimuovo righe di classe 2, le metto in un array che poi andrà nel
%%validation

ultima_colonna_indice = size(roi_data_cleaned_undersampled, 2);

% Trova le righe in cui l'ultima colonna è uguale a 2
roi_class2 = roi_data_cleaned_undersampled(roi_data_cleaned_undersampled(:, ultima_colonna_indice) == 2, :);

% Trova le righe in cui l'ultima colonna non è uguale a 2
roi_data_cleaned_undersampled = roi_data_cleaned_undersampled(roi_data_cleaned_undersampled(:, ultima_colonna_indice) ~= 2, :);
%% DIVISIONE TRAINING E VALIDATION 
% Estrai la colonna degli ID dei pazienti
id_pazienti = roi_data_cleaned_undersampled(:, 1);

% Trova gli ID univoci dei pazienti
id_univoci = unique(id_pazienti);

% Calcola il numero totale di pazienti
num_pazienti = length(id_univoci);

% Calcola il numero di pazienti da includere nel set di allenamento e convalida
percentuale_training = 0.7;
num_pazienti_training = round(num_pazienti * percentuale_training);

% Mescola casualmente gli ID dei pazienti
id_univoci_mischiati = id_univoci(randperm(num_pazienti));

% Seleziona i primi num_pazienti_training ID per il training set
id_training = id_univoci_mischiati(1:num_pazienti_training);

% Gli ID rimanenti vanno nel set di convalida
id_convalida = id_univoci_mischiati(num_pazienti_training+1:end);

% Filtra il dataset in base agli ID di training e convalida
training_set = ismember(id_pazienti, id_training);
validation_set = ismember(id_pazienti, id_convalida);

% Estrai le righe corrispondenti ai set di allenamento e convalida
training_set = roi_data_cleaned_undersampled(training_set, :);
validation_set = roi_data_cleaned_undersampled(validation_set, :);

% Visualizza le dimensioni dei set di allenamento e convalida
disp(['Dimensione del training set: ', num2str(size(training_set))]);
disp(['Dimensione del validation set: ', num2str(size(validation_set))]);

validation_set = [validation_set; roi_data_classe_0_mantenute; roi_class2];

% Ordina gli array in base alla prima colonna
training_set = sortrows(training_set, 1);
validation_set = sortrows(validation_set, 1);

% Salva il training set come file MAT
save('training_set.mat', 'training_set');

% Salva il validation set come file MAT
save('validation_set.mat', 'validation_set');








