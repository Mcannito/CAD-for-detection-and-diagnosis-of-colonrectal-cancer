clc
clear all
close all

% Carica i dati
load('training_fs4.mat');
load('validation_fs4.mat');
load('training_fs6.mat');
load('validation_fs6.mat');
load('training_fs10.mat');
load('validation_fs10.mat');

%% Feature Set 4 (FS4)
X_train_4 = training_fs4(:, 5:end-1);
y_train_4 = training_fs4(:, end);

X_val_4 = validation_fs4(:, 5:end-1);
y_val_true_4 = validation_fs4(:, end);
X = X_train_4; % Assumendo che i dati di addestramento siano in 'X_train_4'
Y = y_train_4; % Assumendo che le etichette di addestramento siano in 'y_train_4'

% Definisci una griglia di valori per il numero di alberi
numTrees = [1, 2, 3, 4, 5, 6]; % Puoi aggiungere altri valori a seconda delle tue esigenze

% Inizializza vettori per immagazzinare le prestazioni
accuracy = zeros(length(numTrees), 1);

% Loop sul numero di alberi
for i = 1:length(numTrees)
    % Configura il modello random forest
    model = fitcensemble(X, Y, 'Method', 'Bag', 'NumLearningCycles', numTrees(i));

    % Esegui la cross-validazione e ottieni l'errore di classificazione medio
    cvmodel = crossval(model, 'KFold', 10);
    accuracy(i) = kfoldLoss(cvmodel, 'LossFun', 'ClassifError');
end

% Trova il numero ottimale di alberi che massimizza le prestazioni
[optimalNumTrees, idx] = min(accuracy);

% Visualizza i risultati
figure;
plot(numTrees, accuracy, '-o');
xlabel('Numero di Alberi');
ylabel('Errore di Classificazione Medio');
title('Valutazione del Numero di Alberi in un Random Forest');
grid on;

fprintf('Il numero ottimale di alberi è: %d\n', numTrees(idx));

optimalNumTree = numTrees(idx);

% Random Forest

RFModel_4 = TreeBagger(optimalNumTree, X_train_4, y_train_4, 'Method', 'classification');
y_train_pred_rf_4 = str2double(predict(RFModel_4, X_train_4));
y_train_pred_rf_4 = round(y_train_pred_rf_4);
y_val_pred_rf_4 = str2double(predict(RFModel_4, X_val_4));
y_val_pred_rf_4 = round(y_val_pred_rf_4);



% Performance Metrics and Confusion Matrix for RF (FS4)
confmat_rf_train_4 = confusionmat(y_train_4, y_train_pred_rf_4);
accuracy_rf_train_4 = sum(diag(confmat_rf_train_4)) / sum(confmat_rf_train_4(:));
precision_rf_train_4 = confmat_rf_train_4(2, 2) / sum(confmat_rf_train_4(:, 2));
recall_rf_train_4 = confmat_rf_train_4(2, 2) / sum(confmat_rf_train_4(2, :));
f1_score_rf_train_4 = 2 * (precision_rf_train_4 * recall_rf_train_4) / (precision_rf_train_4 + recall_rf_train_4);
npv_rf_train_4 = confmat_rf_train_4(1, 1) / sum(confmat_rf_train_4(1, :));

confmat_rf_val_4 = confusionmat(y_val_true_4, y_val_pred_rf_4);
accuracy_rf_val_4 = sum(diag(confmat_rf_val_4)) / sum(confmat_rf_val_4(:));
precision_rf_val_4 = confmat_rf_val_4(2, 2) / sum(confmat_rf_val_4(:, 2));
recall_rf_val_4 = confmat_rf_val_4(2, 2) / sum(confmat_rf_val_4(2, :));
f1_score_rf_val_4 = 2 * (precision_rf_val_4 * recall_rf_val_4) / (precision_rf_val_4 + recall_rf_val_4);
npv_rf_val_4 = confmat_rf_val_4(1, 1) / sum(confmat_rf_val_4(1, :));



disp('--- Feature Set 4 (FS4) ---');
disp('Random Forest Performance (Training Set):');
disp(['Accuracy: ', num2str(accuracy_rf_train_4)]);
disp(['Precision: ', num2str(precision_rf_train_4)]);
disp(['Recall: ', num2str(recall_rf_train_4)]);
disp(['F1 Score: ', num2str(f1_score_rf_train_4)]);
disp(['NPV: ', num2str(npv_rf_train_4)]);
disp('Confusion Matrix - Random Forest (Training Set):');
disp(confmat_rf_train_4);

disp('Random Forest Performance (Validation Set):');
disp(['Accuracy: ', num2str(accuracy_rf_val_4)]);
disp(['Precision: ', num2str(precision_rf_val_4)]);
disp(['Recall: ', num2str(recall_rf_val_4)]);
disp(['F1 Score: ', num2str(f1_score_rf_val_4)]);
disp(['NPV: ', num2str(npv_rf_val_4)]);
disp('Confusion Matrix - Random Forest (Validation Set):');
disp(confmat_rf_val_4);



%% Feature Set 6 (FS6)
X_train_6 = training_fs6(:, 5:end-1);
y_train_6 = training_fs6(:, end);

X_val_6 = validation_fs6(:, 5:end-1);
y_val_true_6 = validation_fs6(:, end);

X = X_train_6; % Assumendo che i dati di addestramento siano in 'X_train_4'
Y = y_train_6; % Assumendo che le etichette di addestramento siano in 'y_train_4'

% Definisci una griglia di valori per il numero di alberi
numTrees = [1, 2, 3, 4, 5, 6]; % Puoi aggiungere altri valori a seconda delle tue esigenze

% Inizializza vettori per immagazzinare le prestazioni
accuracy = zeros(length(numTrees), 1);

% Loop sul numero di alberi
for i = 1:length(numTrees)
    % Configura il modello random forest
    model = fitcensemble(X, Y, 'Method', 'Bag', 'NumLearningCycles', numTrees(i));

    % Esegui la cross-validazione e ottieni l'errore di classificazione medio
    cvmodel = crossval(model, 'KFold', 10);
    accuracy(i) = kfoldLoss(cvmodel, 'LossFun', 'ClassifError');
end

% Trova il numero ottimale di alberi che massimizza le prestazioni
[optimalNumTrees, idx] = min(accuracy);

% Visualizza i risultati
figure;
plot(numTrees, accuracy, '-o');
xlabel('Numero di Alberi');
ylabel('Errore di Classificazione Medio');
title('Valutazione del Numero di Alberi in un Random Forest');
grid on;

fprintf('Il numero ottimale di alberi è: %d\n', numTrees(idx));

optimalNumTree2 = numTrees(idx);
% Random Forest

RFModel_7 = TreeBagger(optimalNumTree2, X_train_6, y_train_6, 'Method', 'classification');
y_train_pred_rf_6 = str2double(predict(RFModel_7, X_train_6));
y_train_pred_rf_6 = round(y_train_pred_rf_6);
y_val_pred_rf_6 = str2double(predict(RFModel_7, X_val_6));
y_val_pred_rf_6 = round(y_val_pred_rf_6);



% Performance Metrics and Confusion Matrix for RF (FS7)
confmat_rf_train_7 = confusionmat(y_train_6, y_train_pred_rf_6);
accuracy_rf_train_7 = sum(diag(confmat_rf_train_7)) / sum(confmat_rf_train_7(:));
precision_rf_train_7 = confmat_rf_train_7(2, 2) / sum(confmat_rf_train_7(:, 2));
recall_rf_train_7 = confmat_rf_train_7(2, 2) / sum(confmat_rf_train_7(2, :));
f1_score_rf_train_7 = 2 * (precision_rf_train_7 * recall_rf_train_7) / (precision_rf_train_7 + recall_rf_train_7);
npv_rf_train_7 = confmat_rf_train_7(1, 1) / sum(confmat_rf_train_7(1, :));

confmat_rf_val_7 = confusionmat(y_val_true_6, y_val_pred_rf_6);
accuracy_rf_val_7 = sum(diag(confmat_rf_val_7)) / sum(confmat_rf_val_7(:));
precision_rf_val_7 = confmat_rf_val_7(2, 2) / sum(confmat_rf_val_7(:, 2));
recall_rf_val_7 = confmat_rf_val_7(2, 2) / sum(confmat_rf_val_7(2, :));
f1_score_rf_val_7 = 2 * (precision_rf_val_7 * recall_rf_val_7) / (precision_rf_val_7 + recall_rf_val_7);
npv_rf_val_7 = confmat_rf_val_7(1, 1) / sum(confmat_rf_val_7(1, :));



disp('--- Feature Set 6 (FS6) ---');
disp('Random Forest Performance (Training Set):');
disp(['Accuracy: ', num2str(accuracy_rf_train_7)]);
disp(['Precision: ', num2str(precision_rf_train_7)]);
disp(['Recall: ', num2str(recall_rf_train_7)]);
disp(['F1 Score: ', num2str(f1_score_rf_train_7)]);
disp(['NPV: ', num2str(npv_rf_train_7)]);
disp('Confusion Matrix - Random Forest (Training Set):');
disp(confmat_rf_train_7);

disp('Random Forest Performance (Validation Set):');
disp(['Accuracy: ', num2str(accuracy_rf_val_7)]);
disp(['Precision: ', num2str(precision_rf_val_7)]);
disp(['Recall: ', num2str(recall_rf_val_7)]);
disp(['F1 Score: ', num2str(f1_score_rf_val_7)]);
disp(['NPV: ', num2str(npv_rf_val_7)]);
disp('Confusion Matrix - Random Forest (Validation Set):');
disp(confmat_rf_val_7);



%% Feature Set 10 (FS10)
X_train_10 = training_fs10(:, 5:end-1);
y_train_10 = training_fs10(:, end);

X_val_10 = validation_fs10(:, 5:end-1);
y_val_true_10 = validation_fs10(:, end);

X = X_train_10; % Assumendo che i dati di addestramento siano in 'X_train_4'
Y = y_train_10; % Assumendo che le etichette di addestramento siano in 'y_train_4'

% Definisci una griglia di valori per il numero di alberi
numTrees = [1, 2, 3, 4, 5, 6]; % Puoi aggiungere altri valori a seconda delle tue esigenze

% Inizializza vettori per immagazzinare le prestazioni
accuracy = zeros(length(numTrees), 1);

% Loop sul numero di alberi
for i = 1:length(numTrees)
    % Configura il modello random forest
    model = fitcensemble(X, Y, 'Method', 'Bag', 'NumLearningCycles', numTrees(i));

    % Esegui la cross-validazione e ottieni l'errore di classificazione medio
    cvmodel = crossval(model, 'KFold', 10);
    accuracy(i) = kfoldLoss(cvmodel, 'LossFun', 'ClassifError');
end

% Trova il numero ottimale di alberi che massimizza le prestazioni
[optimalNumTrees, idx] = min(accuracy);

% Visualizza i risultati
figure;
plot(numTrees, accuracy, '-o');
xlabel('Numero di Alberi');
ylabel('Errore di Classificazione Medio');
title('Valutazione del Numero di Alberi in un Random Forest');
grid on;

fprintf('Il numero ottimale di alberi è: %d\n', numTrees(idx));

optimalNumTree3 = numTrees(idx);
% Random Forest

RFModel_10 = TreeBagger(optimalNumTree3, X_train_10, y_train_10, 'Method', 'classification');
y_train_pred_rf_12 = str2double(predict(RFModel_10, X_train_10));
y_train_pred_rf_12 = round(y_train_pred_rf_12);
y_val_pred_rf_12 = str2double(predict(RFModel_10, X_val_10));
y_val_pred_rf_12 = round(y_val_pred_rf_12);



% Performance Metrics and Confusion Matrix for RF (FS12)
confmat_rf_train_12 = confusionmat(y_train_10, y_train_pred_rf_12);
accuracy_rf_train_12 = sum(diag(confmat_rf_train_12)) / sum(confmat_rf_train_12(:));
precision_rf_train_12 = confmat_rf_train_12(2, 2) / sum(confmat_rf_train_12(:, 2));
recall_rf_train_12 = confmat_rf_train_12(2, 2) / sum(confmat_rf_train_12(2, :));
f1_score_rf_train_12 = 2 * (precision_rf_train_12 * recall_rf_train_12) / (precision_rf_train_12 + recall_rf_train_12);
npv_rf_train_12 = confmat_rf_train_12(1, 1) / sum(confmat_rf_train_12(1, :));

confmat_rf_val_12 = confusionmat(y_val_true_10, y_val_pred_rf_12);
accuracy_rf_val_12 = sum(diag(confmat_rf_val_12)) / sum(confmat_rf_val_12(:));
precision_rf_val_12 = confmat_rf_val_12(2, 2) / sum(confmat_rf_val_12(:, 2));
recall_rf_val_12 = confmat_rf_val_12(2, 2) / sum(confmat_rf_val_12(2, :));
f1_score_rf_val_12 = 2 * (precision_rf_val_12 * recall_rf_val_12) / (precision_rf_val_12 + recall_rf_val_12);
npv_rf_val_12 = confmat_rf_val_12(1, 1) / sum(confmat_rf_val_12(1, :));


disp('--- Feature Set 10 (FS10) ---');
disp('Random Forest Performance (Training Set):');
disp(['Accuracy: ', num2str(accuracy_rf_train_12)]);
disp(['Precision: ', num2str(precision_rf_train_12)]);
disp(['Recall: ', num2str(recall_rf_train_12)]);
disp(['F1 Score: ', num2str(f1_score_rf_train_12)]);
disp(['NPV: ', num2str(npv_rf_train_12)]);
disp('Confusion Matrix - Random Forest (Training Set):');
disp(confmat_rf_train_12);

disp('Random Forest Performance (Validation Set):');
disp(['Accuracy: ', num2str(accuracy_rf_val_12)]);
disp(['Precision: ', num2str(precision_rf_val_12)]);
disp(['Recall: ', num2str(recall_rf_val_12)]);
disp(['F1 Score: ', num2str(f1_score_rf_val_12)]);
disp(['NPV: ', num2str(npv_rf_val_12)]);
disp('Confusion Matrix - Random Forest (Validation Set):');
disp(confmat_rf_val_12);





