%% Lab 4 - ann

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
rng(1);

%% Feature Set 4 (FS4)
X_train_4 = training_fs4(:, 5:end-1);
y_train_4 = training_fs4(:, end);

X_val_4 = validation_fs4(:, 5:end-1);
y_val_true_4 = validation_fs4(:, end);


% Neural Network (ANN)
hiddenLayerSize = [9 5 5 2];
X_train_ann_4 = training_fs4(:, 5:end-1)';
y_train_ann_4 = training_fs4(:, end)';
net_4 = patternnet(hiddenLayerSize, 'trainscg', 'mse');
net_4.trainParam.epochs = 1000;
net_4.trainParam.lr = 0.001; 
net_4.trainParam.max_fail = 100; % Modifica il numero di controlli di validazione come desiderato
net_4.divideParam.trainRatio = 80/100;
net_4.divideParam.valRatio = 20/100;
net_4.divideParam.testRatio = 0/100;

[net_4, tr4] = train(net_4, X_train_ann_4, y_train_ann_4);

y_train_pred_ann_4 = round(net_4(X_train_ann_4));
y_val_pred_ann_4 = round(net_4(X_val_4'));

% Performance Metrics and Confusion Matrix for ANN (FS4)
confmat_ann_train_4 = confusionmat(y_train_ann_4, y_train_pred_ann_4);
accuracy_ann_train_4 = sum(diag(confmat_ann_train_4)) / sum(confmat_ann_train_4(:));
precision_ann_train_4 = confmat_ann_train_4(2, 2) / sum(confmat_ann_train_4(:, 2));
recall_ann_train_4 = confmat_ann_train_4(2, 2) / sum(confmat_ann_train_4(2, :));
f1_score_ann_train_4 = 2 * (precision_ann_train_4 * recall_ann_train_4) / (precision_ann_train_4 + recall_ann_train_4);
npv_ann_train_4 = confmat_ann_train_4(1, 1) / sum(confmat_ann_train_4(1, :));
figure;
confusionchart(y_train_ann_4, y_train_pred_ann_4, 'Title', 'Training Set 4 features');

confmat_ann_val_4 = confusionmat(y_val_true_4, y_val_pred_ann_4);
accuracy_ann_val_4 = sum(diag(confmat_ann_val_4)) / sum(confmat_ann_val_4(:));
precision_ann_val_4 = confmat_ann_val_4(2, 2) / sum(confmat_ann_val_4(:, 2));
recall_ann_val_4 = confmat_ann_val_4(2, 2) / sum(confmat_ann_val_4(2, :));
f1_score_ann_val_4 = 2 * (precision_ann_val_4 * recall_ann_val_4) / (precision_ann_val_4 + recall_ann_val_4);
npv_ann_val_4 = confmat_ann_val_4(1, 1) / sum(confmat_ann_val_4(1, :));
figure;
confusionchart(y_val_true_4, y_val_pred_ann_4, 'Title', 'Validation Set 4 features');

disp('--- Feature Set 4 (FS4) ---');
disp('Neural Network (ANN) Performance (Training Set):');
disp(['Accuracy: ', num2str(accuracy_ann_train_4)]);
disp(['Precision: ', num2str(precision_ann_train_4)]);
disp(['Recall: ', num2str(recall_ann_train_4)]);
disp(['F1 Score: ', num2str(f1_score_ann_train_4)]);
disp(['NPV: ', num2str(npv_ann_train_4)]);
disp('Confusion Matrix - Neural Network (Training Set):');
disp(confmat_ann_train_4);

disp('Neural Network (ANN) Performance (Validation Set):');
disp(['Accuracy: ', num2str(accuracy_ann_val_4)]);
disp(['Precision: ', num2str(precision_ann_val_4)]);
disp(['Recall: ', num2str(recall_ann_val_4)]);
disp(['F1 Score: ', num2str(f1_score_ann_val_4)]);
disp(['NPV: ', num2str(npv_ann_val_4)]);
disp('Confusion Matrix - Neural Network (Validation Set):');
disp(confmat_ann_val_4);

%% Feature Set 7 (FS6)
X_train_7 = training_fs6(:, 5:end-1);
y_train_7 = training_fs6(:, end);

X_val_7 = validation_fs6(:, 5:end-1);
y_val_true_7 = validation_fs6(:, end);

% Neural Network (ANN)
hiddenLayerSize = [9 5 5 2];
X_train_ann_7 = training_fs6(:, 5:end-1)';
y_train_ann_7 = training_fs6(:, end)';
net_6 = patternnet(hiddenLayerSize, 'trainscg', 'mse');
net_6.trainParam.epochs = 1000;
net_6.trainParam.lr = 0.001; 
net_6.trainParam.max_fail = 100; % Modifica il numero di controlli di validazione come desiderato
net_6.divideParam.trainRatio = 80/100;
net_6.divideParam.valRatio = 20/100;
net_6.divideParam.testRatio = 0/100;

[net_6, tr6] = train(net_6, X_train_ann_7, y_train_ann_7);
save('net_6.mat', 'net_6')
y_train_pred_ann_7 = round(net_6(X_train_ann_7));
y_val_pred_ann_7 = round(net_6(X_val_7'));

% Performance Metrics and Confusion Matrix for ANN (FS7)
confmat_ann_train_7 = confusionmat(y_train_ann_7, y_train_pred_ann_7);
accuracy_ann_train_7 = sum(diag(confmat_ann_train_7)) / sum(confmat_ann_train_7(:));
precision_ann_train_7 = confmat_ann_train_7(2, 2) / sum(confmat_ann_train_7(:, 2));
recall_ann_train_7 = confmat_ann_train_7(2, 2) / sum(confmat_ann_train_7(2, :));
f1_score_ann_train_7 = 2 * (precision_ann_train_7 * recall_ann_train_7) / (precision_ann_train_7 + recall_ann_train_7);
npv_ann_train_7 = confmat_ann_train_7(1, 1) / sum(confmat_ann_train_7(1, :));
figure;
confusionchart(y_train_ann_7, y_train_pred_ann_7, 'Title', 'Training Set 6 features');


confmat_ann_val_7 = confusionmat(y_val_true_7, y_val_pred_ann_7);
accuracy_ann_val_7 = sum(diag(confmat_ann_val_7)) / sum(confmat_ann_val_7(:));
precision_ann_val_7 = confmat_ann_val_7(2, 2) / sum(confmat_ann_val_7(:, 2));
recall_ann_val_7 = confmat_ann_val_7(2, 2) / sum(confmat_ann_val_7(2, :));
f1_score_ann_val_7 = 2 * (precision_ann_val_7 * recall_ann_val_7) / (precision_ann_val_7 + recall_ann_val_7);
npv_ann_val_7 = confmat_ann_val_7(1, 1) / sum(confmat_ann_val_7(1, :));
figure;
confusionchart(y_val_true_7, y_val_pred_ann_7, 'Title', 'Validation Set 6 features');


disp('--- Feature Set 6 (FS6) ---');
disp('Neural Network (ANN) Performance (Training Set):');
disp(['Accuracy: ', num2str(accuracy_ann_train_7)]);
disp(['Precision: ', num2str(precision_ann_train_7)]);
disp(['Recall: ', num2str(recall_ann_train_7)]);
disp(['F1 Score: ', num2str(f1_score_ann_train_7)]);
disp(['NPV: ', num2str(npv_ann_train_7)]);
disp('Confusion Matrix - Neural Network (Training Set):');
disp(confmat_ann_train_7);

disp('Neural Network (ANN) Performance (Validation Set):');
disp(['Accuracy: ', num2str(accuracy_ann_val_7)]);
disp(['Precision: ', num2str(precision_ann_val_7)]);
disp(['Recall: ', num2str(recall_ann_val_7)]);
disp(['F1 Score: ', num2str(f1_score_ann_val_7)]);
disp(['NPV: ', num2str(npv_ann_val_7)]);
disp('Confusion Matrix - Neural Network (Validation Set):');
disp(confmat_ann_val_7);

%% Feature Set 12 (FS12)
X_train_10 = training_fs10(:, 5:end-1);
y_train_10 = training_fs10(:, end);

X_val_10 = validation_fs10(:, 5:end-1);
y_val_true_10 = validation_fs10(:, end);

% Neural Network (ANN)
hiddenLayerSize = [10 10 2];
X_train_ann_12 = training_fs10(:, 5:end-1)';
y_train_ann_12 = training_fs10(:, end)';
net_10 = patternnet(hiddenLayerSize, 'trainscg', 'mse');
net_10.trainParam.epochs = 1000;
net_10.trainParam.lr = 0.001; 
net_10.trainParam.max_fail = 100; % Modifica il numero di controlli di validazione come desiderato
net_10.divideParam.trainRatio = 80/100;
net_10.divideParam.valRatio = 20/100;
net_10.divideParam.valRatio = 0/100;

[net_10, tr10] = train(net_10, X_train_ann_12, y_train_ann_12);

y_train_pred_ann_12 = round(net_10(X_train_ann_12));
y_val_pred_ann_12 = round(net_10(X_val_10'));

% Performance Metrics and Confusion Matrix for ANN (FS12)
confmat_ann_train_12 = confusionmat(y_train_ann_12, y_train_pred_ann_12);
accuracy_ann_train_12 = sum(diag(confmat_ann_train_12)) / sum(confmat_ann_train_12(:));
precision_ann_train_12 = confmat_ann_train_12(2, 2) / sum(confmat_ann_train_12(:, 2));
recall_ann_train_12 = confmat_ann_train_12(2, 2) / sum(confmat_ann_train_12(2, :));
f1_score_ann_train_12 = 2 * (precision_ann_train_12 * recall_ann_train_12) / (precision_ann_train_12 + recall_ann_train_12);
npv_ann_train_12 = confmat_ann_train_12(1, 1) / sum(confmat_ann_train_12(1, :));
figure;
confusionchart(y_train_ann_12, y_train_pred_ann_12, 'Title', 'Training Set 10 features');

confmat_ann_val_12 = confusionmat(y_val_true_10, y_val_pred_ann_12);
accuracy_ann_val_12 = sum(diag(confmat_ann_val_12)) / sum(confmat_ann_val_12(:));
precision_ann_val_12 = confmat_ann_val_12(2, 2) / sum(confmat_ann_val_12(:, 2));
recall_ann_val_12 = confmat_ann_val_12(2, 2) / sum(confmat_ann_val_12(2, :));
f1_score_ann_val_12 = 2 * (precision_ann_val_12 * recall_ann_val_12) / (precision_ann_val_12 + recall_ann_val_12);
npv_ann_val_12 = confmat_ann_val_12(1, 1) / sum(confmat_ann_val_12(1, :));
figure;
confusionchart(y_val_true_10, y_val_pred_ann_12, 'Title', 'Validation Set 10 features');

disp('--- Feature Set 10 (FS10) ---');
disp('Neural Network (ANN) Performance (Training Set):');
disp(['Accuracy: ', num2str(accuracy_ann_train_12)]);
disp(['Precision: ', num2str(precision_ann_train_12)]);
disp(['Recall: ', num2str(recall_ann_train_12)]);
disp(['F1 Score: ', num2str(f1_score_ann_train_12)]);
disp(['NPV: ', num2str(npv_ann_train_12)]);
disp('Confusion Matrix - Neural Network (Training Set):');
disp(confmat_ann_train_12);

disp('Neural Network (ANN) Performance (Validation Set):');
disp(['Accuracy: ', num2str(accuracy_ann_val_12)]);
disp(['Precision: ', num2str(precision_ann_val_12)]);
disp(['Recall: ', num2str(recall_ann_val_12)]);
disp(['F1 Score: ', num2str(f1_score_ann_val_12)]);
disp(['NPV: ', num2str(npv_ann_val_12)]);
disp('Confusion Matrix - Neural Network (Validation Set):');
disp(confmat_ann_val_12);
save('net_10.mat', 'net_10')
