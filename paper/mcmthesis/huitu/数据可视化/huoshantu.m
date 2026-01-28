

%% 5. 火山图（差异表达分析）
clc; clear; close all;

% 生成基因表达差异分析数据
rng(789);
n_genes = 1000;

% 生成log2 fold change数据
log2fc = randn(n_genes, 1) * 1.5;
% 添加一些显著差异的基因
significant_indices = randperm(n_genes, 50);
log2fc(significant_indices) = log2fc(significant_indices) + ...
                              sign(log2fc(significant_indices)) * 3;

% 生成p值数据（负log10转换）
p_values = rand(n_genes, 1);
p_values(significant_indices) = p_values(significant_indices) / 1000; % 使显著基因的p值更小
neg_log10_p = -log10(p_values);

% 创建图形
figure('Position', [200, 200, 700, 500]);

% 定义颜色
up_color = [0.8, 0.2, 0.2];      % 上调 - 红色
down_color = [0.2, 0.2, 0.8];    % 下调 - 蓝色
ns_color = [0.6, 0.6, 0.6];      % 不显著 - 灰色

% 定义显著性阈值
fc_threshold = 1.0;           % fold change阈值
p_threshold = 0.05;           % p值阈值

% 分类基因
is_up = (log2fc > fc_threshold) & (neg_log10_p > -log10(p_threshold));
is_down = (log2fc < -fc_threshold) & (neg_log10_p > -log10(p_threshold));
is_ns = ~(is_up | is_down);

% 绘制散点图
scatter(log2fc(is_ns), neg_log10_p(is_ns), 20, ns_color, 'filled', ...
        'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none');
hold on;
scatter(log2fc(is_up), neg_log10_p(is_up), 40, up_color, 'filled', ...
        'MarkerFaceAlpha', 0.8, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
scatter(log2fc(is_down), neg_log10_p(is_down), 40, down_color, 'filled', ...
        'MarkerFaceAlpha', 0.8, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
% 添加阈值线
h1 = plot([-fc_threshold, -fc_threshold], [0, max(neg_log10_p)*1.05], ...
     'k--', 'LineWidth', 1);
h1.Color = [0 0 0 0.5];  % 黑色，50%透明度

h2 = plot([fc_threshold, fc_threshold], [0, max(neg_log10_p)*1.05], ...
     'k--', 'LineWidth', 1);
h2.Color = [0 0 0 0.5];  % 黑色，50%透明度

h3 = plot([min(log2fc)*1.1, max(log2fc)*1.1], [-log10(p_threshold), -log10(p_threshold)], ...
     'k--', 'LineWidth', 1);
h3.Color = [0 0 0 0.5];  % 黑色，50%透明度

% 坐标轴设置
xlim([min(log2fc)*1.1, max(log2fc)*1.1]);
ylim([0, max(neg_log10_p)*1.05]);
set(gca, 'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
grid on;

% 标签和标题
xlabel('log_2(Fold Change)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('-log_{10}(p-value)', 'FontSize', 12, 'FontWeight', 'bold');
title('Volcano Plot: Differential Expression Analysis', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 添加图例
legend({'Not Significant', 'Up-regulated', 'Down-regulated'}, ...
       'Location', 'northeast', 'Box', 'off');

% 添加计数信息
n_up = sum(is_up);
n_down = sum(is_down);
text(max(log2fc)*0.9, max(neg_log10_p)*0.95, ...
     sprintf('Up: %d genes\nDown: %d genes', n_up, n_down), ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
     'FontSize', 10, 'BackgroundColor', [1 1 1 0.8]);

% 标注前几个最显著的基因
[~, top_indices] = sort(neg_log10_p, 'descend');
top_n = min(5, length(top_indices));
for i = 1:top_n
    idx = top_indices(i);
    if is_up(idx) || is_down(idx)
        gene_name = sprintf('Gene_%04d', idx);
        text(log2fc(idx), neg_log10_p(idx), gene_name, ...
             'FontSize', 8, 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom');
    end
end

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure5.png');

%% 6. 高级火山图（多组比较）
clc; clear; close all;

% 生成多条件比较数据
rng(321);
n_genes = 800;
n_conditions = 3;

% 创建多个条件的数据
data_multi = struct();
condition_names = {'Treatment A', 'Treatment B', 'Treatment C'};
colors_multi = [0.8, 0.3, 0.3;    % 红色
                0.3, 0.8, 0.3;    % 绿色
                0.3, 0.3, 0.8];   % 蓝色

figure('Position', [200, 200, 800, 550]);
hold on;

% 先收集所有数据用于计算全局最大值
all_neg_log10_p = [];

for cond = 1:n_conditions
    % 为每个条件生成数据
    log2fc = randn(n_genes, 1) * 1.2;
    p_values = rand(n_genes, 1);
    
    % 添加条件特异性信号
    signal_genes = randperm(n_genes, 30 + cond*10);
    effect_size = 1.5 + cond*0.5;
    log2fc(signal_genes) = log2fc(signal_genes) + ...
                          effect_size * sign(randn(length(signal_genes), 1));
    p_values(signal_genes) = p_values(signal_genes) / (100 * cond);
    
    neg_log10_p = -log10(p_values);
    
    % 保存数据
    data_multi(cond).log2fc = log2fc;
    data_multi(cond).neg_log10_p = neg_log10_p;
    data_multi(cond).p_values = p_values;
    
    % 收集所有neg_log10_p用于计算全局最大值
    all_neg_log10_p = [all_neg_log10_p; neg_log10_p];
    
    % 绘制条件特异性点
    fc_threshold = 1.0;
    p_threshold = 0.05;
    is_sig = (abs(log2fc) > fc_threshold) & (neg_log10_p > -log10(p_threshold));
    
    % 不显著的点（小尺寸）
    scatter(log2fc(~is_sig), neg_log10_p(~is_sig), 15, ...
            colors_multi(cond,:), 'filled', 'MarkerFaceAlpha', 0.3);
    
    % 显著的点（大尺寸）
    scatter(log2fc(is_sig), neg_log10_p(is_sig), 60, ...
            colors_multi(cond,:), 'filled', 'MarkerFaceAlpha', 0.8, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end

% 计算全局最大值
global_max_neg_log10_p = max(all_neg_log10_p);
y_max_value = global_max_neg_log10_p * 1.05;

% 添加阈值线
fc_threshold = 1.0;
p_threshold = 0.05;

% 垂直阈值线
h1 = plot([-fc_threshold, -fc_threshold], [0, y_max_value], ...
     'k--', 'LineWidth', 1.5);
h1.Color = [0 0 0 0.7];  % 设置透明度为0.7

h2 = plot([fc_threshold, fc_threshold], [0, y_max_value], ...
     'k--', 'LineWidth', 1.5);
h2.Color = [0 0 0 0.7];  % 设置透明度为0.7

% 水平阈值线
h3 = plot([-5, 5], [-log10(p_threshold), -log10(p_threshold)], ...
     'k--', 'LineWidth', 1.5);
h3.Color = [0 0 0 0.7];  % 设置透明度为0.7

% 坐标轴设置
xlim([-4, 4]);
ylim([0, y_max_value]);
set(gca, 'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
grid on;

% 标签和标题
xlabel('log_2(Fold Change)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('-log_{10}(Adjusted p-value)', 'FontSize', 12, 'FontWeight', 'bold');
title('Multi-Condition Volcano Plot Comparison', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 创建自定义图例
legend_elements = [];
for cond = 1:n_conditions
    h = scatter(NaN, NaN, 60, colors_multi(cond,:), 'filled', ...
                'MarkerFaceAlpha', 0.8, 'MarkerEdgeColor', 'k');
    legend_elements = [legend_elements, h];
end
legend(legend_elements, condition_names, ...
       'Location', 'northeast', 'Box', 'off');

% 添加象限标注
text(-2.5, y_max_value * 0.8, 'Down-regulated', ...
     'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', ...
     'Color', [0.2, 0.2, 0.8]);
text(2.5, y_max_value * 0.8, 'Up-regulated', ...
     'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', ...
     'Color', [0.8, 0.2, 0.2]);

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure6.png');



