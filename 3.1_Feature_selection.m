%% Lab 3 - discretizzazione e feature selection
clc
clear all
close all

load('training_set.mat')
load('validation_set.mat')
load('feature_discr.mat')


%% Feature selection after discretization

% Seleziona le features discretizzate e la classe 
features_discretized = feature_discr; 
labels = training_set(:, end); 
 
% Esegui la feature selection con fscmrmr 
[selected_features_indices, scores] = fscmrmr(features_discretized, labels); 


% Creare un istogramma 
figure; 
bar(scores(selected_features_indices), 'barwidth', 0.5); 
title('Punteggi FSCmRMR per Features'); 
xticks(1:length(selected_features_indices));
xticklabels(cellstr(num2str(selected_features_indices')));
xtickangle(45)
ylabel('Punteggio FSCmRMR'); 
grid on; 
 
% Le feature selezionate sono la numero 1 22 2 24; 42 23; 21 15 31 18
threshold = 0.016;
selected_indices_above_threshold = selected_features_indices(scores(selected_features_indices) > threshold);

% Estrai colonne selezionate sopra la soglia da training_set
training_fs10 = [training_set(:, 1:4), training_set(:, selected_indices_above_threshold + 4), training_set(:,end)];

% Estrai colonne selezionate sopra la soglia da validation_set
validation_fs10 = [validation_set(:, 1:4), validation_set(:, selected_indices_above_threshold + 4), validation_set(:,end)];

save('training_fs10.mat', 'training_fs10')
save('validation_fs10.mat', 'validation_fs10')
