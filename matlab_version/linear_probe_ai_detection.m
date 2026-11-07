%% linear_probe_ai_detection.m
% Linear probe (difference-of-means direction) for detecting AI-generated vs
% human-written text from an LLM's hidden states.
%
% Requires: MATLAB R2016b+ (local functions in scripts),
%           Statistics and Machine Learning Toolbox (cvpartition, pca, perfcurve).
%
% Pipeline (matches the diagram):
%   1-2. texts -> observer LLM -> mean-pooled hidden state h from one layer
%   3.   direction  d = (mu_AI - mu_Human) / ||mu_AI - mu_Human||
%   4.   projection score  s = h' * d
%   5.   threshold  tau = midpoint of the two class-mean scores (TRAIN set only)
%   6.   classify:  s > tau -> AI,  else Human

clear; clc; close all;

%% ------------------------------ configuration ------------------------------
MODE      = "synthetic";          % "synthetic" | "real"
MAT_FILE  = "hidden_states.mat";  % produced by extract_hidden_states.py
TEST_FRAC = 0.30;                 % 70/30 split
SEED      = 1;
rng(SEED);

% Synthetic-data settings
D         = 768;                  % GPT-2 small hidden size
N_PER_CLS = 400;
SHIFT     = 4;                    % size of the class shift (noise std per dim = 1)
N_LAYERS  = 13;                   % embeddings + 12 blocks (synthetic layer sweep)

%% ------------------------------ get the data -------------------------------
% H: N x D hidden states (steps 1-2), y: 0 = Human, 1 = AI
% H_all: N x D x nLayers (all layers, for the bonus sweep; may be empty)
if MODE == "synthetic"
    % Simulated hidden states: isotropic Gaussian noise; AI class is shifted by
    % SHIFT along ONE random unit direction u (the "linear representation").
    y = [zeros(N_PER_CLS,1); ones(N_PER_CLS,1)];
    N = numel(y);

    % Layer profile: separability peaks in the middle layers (inverted U),
    % mimicking what is typically seen in real LLMs.
    layers  = 0:N_LAYERS-1;
    shifts  = 0.4*SHIFT + 0.6*SHIFT*exp(-(layers-7).^2 / (2*2.5^2));
    LAYER   = 8;                                  % layer used for the main analysis

    H_all = zeros(N, D, N_LAYERS);
    for l = 1:N_LAYERS
        u = randn(D,1);  u = u/norm(u);           % random direction for this layer
        H_all(:,:,l) = randn(N, D) + (y * shifts(l)) * u';
    end
    H = H_all(:,:,LAYER+1);                       % layer index 0 is column 1
elseif MODE == "real"
    S = load(MAT_FILE);
    H = double(S.H);   y = double(S.y(:));
    LAYER = S.LAYER;
    if isfield(S,'H_all'), H_all = double(S.H_all); else, H_all = []; end
else
    error('MODE must be "synthetic" or "real".');
end
fprintf('MODE=%s | N=%d samples, D=%d dims, layer=%d\n', MODE, size(H,1), size(H,2), LAYER);

%% ------------------------ 70/30 stratified split ---------------------------
% cvpartition with a class vector stratifies on the labels.
cv     = cvpartition(y, 'HoldOut', TEST_FRAC);
idxTr  = training(cv);   idxTe = test(cv);

%% -------- steps 3-6 on the chosen layer (d and tau from TRAIN only) --------
[d, tau] = fit_probe(H(idxTr,:), y(idxTr));       % steps 3 and 5
s_all    = H * d;                                  % step 4: s = h' * d for every text
s_te     = s_all(idxTe);   y_te = y(idxTe);
y_hat    = s_te > tau;                             % step 6: s > tau -> AI (1)
acc      = mean(y_hat == y_te);

[fpr, tpr, ~, auc] = perfcurve(y_te, s_te, 1);    % positive class = AI

fprintf('Train n=%d | Test n=%d\n', nnz(idxTr), nnz(idxTe));
fprintf('tau = %.4f\n', tau);
fprintf('TEST accuracy = %.2f %%   TEST AUC = %.4f\n', 100*acc, auc);

blue = [0.20 0.30 0.90];   red = [0.90 0.15 0.15];

%% ------------- Figure 1: PCA 2-D scatter + direction d as arrow ------------
mu_all = mean(H, 1);
[coeff, ~, ~, ~, ~, ~] = pca(H, 'NumComponents', 2);   % PCA on all hidden states
Z      = (H - mu_all) * coeff;                          % N x 2 coordinates
d2     = coeff' * d;                                    % d projected into the PCA plane
arrowLen = 0.6 * max(abs(Z(:)));                        % scale for visibility only
d2s    = arrowLen * d2 / max(norm(d2), eps);            % direction preserved, norm rescaled

figure('Name','PCA scatter','Color','w'); hold on; box on;
scatter(Z(y==0,1), Z(y==0,2), 18, blue, 'filled', 'MarkerFaceAlpha', 0.5);
scatter(Z(y==1,1), Z(y==1,2), 18, red,  'filled', 'MarkerFaceAlpha', 0.5);
quiver(0, 0, d2s(1), d2s(2), 0, 'k', 'LineWidth', 2.5, 'MaxHeadSize', 0.8);
text(d2s(1), d2s(2), '  d', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('PC 1'); ylabel('PC 2');
title(sprintf('PCA of hidden states (layer %d); arrow = projection of d (|proj| = %.2f of |d|)', ...
      LAYER, norm(d2)));
legend({'Human','AI','direction d'}, 'Location','best');
axis equal; grid on;

%% -------------- Figure 2: histograms of projection scores s ----------------
figure('Name','Projection scores','Color','w'); hold on; box on;
edges = linspace(min(s_te), max(s_te), 30);
histogram(s_te(y_te==0), edges, 'FaceColor', blue, 'FaceAlpha', 0.5, 'Normalization', 'pdf');
histogram(s_te(y_te==1), edges, 'FaceColor', red,  'FaceAlpha', 0.5, 'Normalization', 'pdf');
xline(tau, '--k', sprintf('\\tau = %.2f', tau), 'LineWidth', 2, 'LabelOrientation','horizontal');
xlabel('projection score  s = h^T d'); ylabel('density');
title(sprintf('Test-set projection scores (accuracy %.1f%%)', 100*acc));
legend({'Human','AI','\tau'}, 'Location','best');

%% ------------------------- Figure 3: ROC curve -----------------------------
figure('Name','ROC','Color','w'); hold on; box on;
plot(fpr, tpr, 'LineWidth', 2.5, 'Color', red);
plot([0 1], [0 1], 'k--');
xlabel('False positive rate'); ylabel('True positive rate');
title(sprintf('ROC curve, AUC = %.3f', auc));
legend({'linear probe','chance'}, 'Location','southeast'); axis square; grid on;

%% -------------- Bonus: test AUC for every layer (same split) ---------------
if ~isempty(H_all)
    nL     = size(H_all, 3);
    aucL   = zeros(nL, 1);   accL = zeros(nL, 1);
    for l = 1:nL
        Hl = H_all(:,:,l);
        [dl, taul] = fit_probe(Hl(idxTr,:), y(idxTr));   % train only
        sl = Hl(idxTe,:) * dl;
        [~,~,~,aucL(l)] = perfcurve(y_te, sl, 1);
        accL(l) = mean((sl > taul) == y_te);
    end
    layerIdx = 0:nL-1;                                   % 0 = embeddings
    [bestAUC, b] = max(aucL);

    figure('Name','AUC vs layer','Color','w'); hold on; box on;
    plot(layerIdx, aucL, '-o', 'LineWidth', 2, 'Color', red, 'MarkerFaceColor', red);
    plot(layerIdx(b), bestAUC, 'p', 'MarkerSize', 16, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k');
    xlabel('GPT-2 layer (0 = embeddings, 12 = last block)'); ylabel('test AUC');
    title(sprintf('Which layer separates Human vs AI best? (best: layer %d, AUC %.3f)', ...
          layerIdx(b), bestAUC));
    ylim([min(0.5, min(aucL)-0.02), 1.01]); grid on;
    fprintf('Best layer = %d (AUC %.4f, acc %.2f %%)\n', layerIdx(b), bestAUC, 100*accL(b));
else
    disp('No H_all in the .mat file -> skipping the layer sweep.');
end

%% ================================ functions ================================
function [d, tau] = fit_probe(Htr, ytr)
% FIT_PROBE  Steps 3 and 5 of the pipeline, using TRAINING data only.
    mu_h = mean(Htr(ytr==0, :), 1)';          % class mean, Human  (D x 1)
    mu_a = mean(Htr(ytr==1, :), 1)';          % class mean, AI     (D x 1)
    diffmu = mu_a - mu_h;
    d = diffmu / norm(diffmu);                % step 3: unit difference of means
    % step 5: midpoint between the mean projected scores of the two classes
    tau = 0.5 * (mu_h' * d + mu_a' * d);
end
