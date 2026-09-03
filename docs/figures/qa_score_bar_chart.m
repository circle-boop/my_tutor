% QA score grouped bar chart for thesis Chapter 4.2
% Run this script in MATLAB from the project root.

clear; clc;

methods = categorical({'Plain LLM', 'S RAG', 'MC RAG'});
methods = reordercats(methods, {'Plain LLM', 'S RAG', 'MC RAG'});

% Columns: relevance, accuracy, context match, completeness,
% hallucination control, total score.
scores = [
    3.46, 2.75, 2.08, 2.50, 3.04, 2.77;
    4.08, 3.67, 4.67, 3.46, 3.17, 3.81;
    4.58, 3.96, 4.38, 3.96, 3.54, 4.08
];

metricNames = {'相关性', '准确性', '上下文匹配度', '完整性', '幻觉控制', '综合得分'};
colors = [
    0.39, 0.52, 0.82;
    0.36, 0.67, 0.48;
    0.86, 0.57, 0.25
];

fig = figure('Color', 'w', 'Position', [100, 100, 1100, 620]);
b = bar(scores', 'grouped', 'BarWidth', 0.78);
for i = 1:numel(b)
    b(i).FaceColor = colors(i, :);
    b(i).EdgeColor = 'none';
end

set(gca, ...
    'XTickLabel', metricNames, ...
    'FontName', 'Microsoft YaHei', ...
    'FontSize', 12, ...
    'LineWidth', 1.0, ...
    'YGrid', 'on', ...
    'GridAlpha', 0.18, ...
    'Box', 'off');

ylim([0, 5]);
ylabel('评分（1-5分）', 'FontName', 'Microsoft YaHei', 'FontSize', 13);
title('QA Score Comparison', 'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold');
legend({'Plain LLM', 'S RAG', 'MC RAG'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'FontName', 'Arial', ...
    'FontSize', 12, ...
    'Box', 'off');

for i = 1:numel(b)
    x = b(i).XEndPoints;
    y = b(i).YEndPoints;
    labels = string(round(b(i).YData, 2));
    text(x, y + 0.06, labels, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontName', 'Arial', ...
        'FontSize', 10, ...
        'Color', [0.25, 0.29, 0.36]);
end

text(3.5, -0.62, '图4.x  不同问答方法评分结果对比', ...
    'HorizontalAlignment', 'center', ...
    'FontName', 'Microsoft YaHei', ...
    'FontSize', 12);

outDir = fullfile('docs', 'figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

exportgraphics(fig, fullfile(outDir, 'qa_score_bar_chart.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outDir, 'qa_score_bar_chart.svg'), 'ContentType', 'vector');
