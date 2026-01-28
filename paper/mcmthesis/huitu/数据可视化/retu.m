%% 1. 基本相关矩阵热图
clc; clear; close all;

% 生成数据
rng(123);
X = rand(13);
X = round(X, 2);
B = ones(1, size(X, 1));
X(logical(eye(size(X)))) = B;

% 创建对称相关矩阵
data = (X + X') / 2;
data(logical(eye(size(data)))) = 1;

% 创建自定义颜色映射
custom_cmap = [
    0.267, 0.004, 0.329;  % 深紫色
    0.282, 0.165, 0.459;  % 紫色
    0.231, 0.318, 0.545;  % 蓝紫色
    0.153, 0.533, 0.557;  % 蓝绿色
    0.122, 0.733, 0.471;  % 绿色
    0.678, 0.847, 0.902   % 浅蓝色
];

% 标签名称
label_names = {'Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E', 'Gene F', ...
               'Gene G', 'Gene H', 'Gene I', 'Gene J', 'Gene K', 'Gene L', 'Gene M'};

% 创建图形
figure('Position', [200, 200, 800, 600]);

% 绘制热图
hot_figure = heatmap(label_names, label_names, data, ...
                     'FontName', 'Arial', 'FontSize', 11);
hot_figure.GridVisible = 'off';
hot_figure.Title = 'Gene Expression Correlation Matrix';
hot_figure.XLabel = 'Genes';
hot_figure.YLabel = 'Genes';
colormap(gca, custom_cmap);

% 对于heatmap，colorbar不需要输出参数
colorbar;

% 方法1：使用heatmap的CellLabelFormat属性显示所有数值
hot_figure.CellLabelFormat = '%.2f';
hot_figure.FontColor = 'w';  % 设置字体颜色为白色，便于在深色背景上显示

% 或者方法2：使用imagesc代替heatmap来添加文本标注
% 删除上面的热图代码，使用以下替代方案：
% 
% figure('Position', [200, 200, 800, 600]);
% imagesc(data);
% colormap(custom_cmap);
% 
% % 设置坐标轴
% xticks(1:13);
% yticks(1:13);
% xticklabels(label_names);
% yticklabels(label_names);
% xtickangle(45);
% 
% % 添加颜色条
% colorbar;
% 
% % 添加数值标注
% [row, col] = find(abs(data) > 0.7 & ~eye(size(data)));
% for i = 1:length(row)
%     text(col(i), row(i), sprintf('%.2f', data(row(i), col(i))), ...
%          'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'w');
% end
% 
% title('Gene Expression Correlation Matrix', 'FontSize', 14, 'FontWeight', 'bold');
% xlabel('Genes');
% ylabel('Genes');

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure1.png');

%% 2. 分层聚类热图
clc; clear; close all;

% 生成基因表达数据
rng(456);
n_genes = 15;
n_samples = 10;

% 创建分组数据
data_matrix = zeros(n_genes, n_samples);
for i = 1:n_genes
    base_level = rand() * 5;
    group_effect = floor((i-1)/5) * 3;
    noise = randn(1, n_samples) * 0.5;
    data_matrix(i, :) = base_level + group_effect + noise;
end

% 添加一些缺失值
missing_indices = randperm(n_genes * n_samples, 5);
data_matrix(missing_indices) = NaN;

% 进行聚类
Z = linkage(data_matrix, 'average', 'euclidean');
T = cluster(Z, 'maxclust', 3);
[sorted_data, idx] = sortrows([T, data_matrix], 1);
sorted_data = sorted_data(:, 2:end);

% 创建图形
figure('Position', [200, 200, 900, 650]);

% 创建子图布局
subplot(1, 10, 1:8);
imagesc(sorted_data);
colormap(jet);
caxis([0, max(data_matrix(:))]);

% 设置坐标轴
xticks(1:n_samples);
xticklabels(arrayfun(@(x) sprintf('Sample %d', x), 1:n_samples, 'UniformOutput', false));
xtickangle(45);
yticks(1:n_genes);
yticklabels(arrayfun(@(x) sprintf('Gene %02d', idx(x)), 1:n_genes, 'UniformOutput', false));

% 添加颜色条
c = colorbar('Position', [0.82, 0.15, 0.02, 0.7]);
c.Label.String = 'Expression Level';
c.Label.FontSize = 11;
c.Label.FontWeight = 'bold';

% 添加聚类树状图
%subplot(1, 10, 9:10);
%dendrogram(Z, 0, 'Orientation', 'right');
%set(gca, 'XTick', [], 'YTick', [], 'Visible', 'off');

% 添加标题和标签
sgtitle('Hierarchical Clustered Heatmap of Gene Expression', 'FontSize', 14, 'FontWeight', 'bold');

% 添加分组标注
%group_boundaries = find(diff(sorted_data(:, 1)) > 0);
%for i = 1:length(group_boundaries)
 %   yline(group_boundaries(i) + 0.5, 'r-', 'LineWidth', 1.5);
%end

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure2.png');

%% 3. 时间序列热图
clc; clear; close all;

% 生成时间序列数据
rng(789);
n_timepoints = 24;
n_variables = 8;

% 创建周期性时间序列数据
time_data = zeros(n_variables, n_timepoints);
time_points = 1:n_timepoints;

for i = 1:n_variables
    % 基础信号
    base_signal = sin(2*pi*time_points/n_timepoints * (1 + rand())) * 2;
    
    % 趋势
    trend = (time_points - n_timepoints/2) * 0.1 * rand();
    
    % 噪声
    noise = randn(1, n_timepoints) * 0.3;
    
    % 组合
    time_data(i, :) = base_signal + trend + noise + 3;
end

% 创建图形
figure('Position', [200, 200, 850, 550]);

% 使用imagesc绘制热图
imagesc(time_points, 1:n_variables, time_data);
colormap(turbo);  % 使用turbo颜色映射

% 设置坐标轴
xticks(1:2:n_timepoints);
xlabel('Time (hours)', 'FontSize', 12, 'FontWeight', 'bold');
yticks(1:n_variables);
yticklabels(arrayfun(@(x) sprintf('Variable %d', x), 1:n_variables, 'UniformOutput', false));
ylabel('Variables', 'FontSize', 12, 'FontWeight', 'bold');

% 添加颜色条
c = colorbar;
c.Label.String = 'Signal Intensity';
c.Label.FontSize = 12;
c.Label.FontWeight = 'bold';

% 添加标题
title('Time Series Heatmap with Contour Lines', 'FontSize', 14, 'FontWeight', 'bold');


% 保存图像
print(gcf, '-dpng', '-r300', 'Figure3.png');




%% 4 分块差异热图（对照组 vs 实验组）
clc; clear; close all;

% 生成两组数据
rng(456);
n_features = 20;
n_samples_per_group = 15;

% 对照组数据
control_data = randn(n_samples_per_group, n_features);
control_data = control_data + repmat(linspace(0, 2, n_features), n_samples_per_group, 1);

% 实验组数据（有差异）
treatment_data = control_data + randn(n_samples_per_group, n_features) * 0.5;
% 在某些特征上添加系统性差异
diff_features = [3:5, 10:12, 17:19];
treatment_data(:, diff_features) = treatment_data(:, diff_features) + 1.5;

% 计算每组的均值
control_mean = mean(control_data, 1);
treatment_mean = mean(treatment_data, 1);

% 计算差异和p值
differences = treatment_mean - control_mean;
p_values = zeros(1, n_features);
for i = 1:n_features
    [~, p_values(i)] = ttest2(control_data(:, i), treatment_data(:, i));
end

% 创建图形
figure('Position', [200, 200, 900, 600]);

% 创建3个子图
% 子图1：差异热图
subplot(2, 3, [1, 2, 4, 5]);
imagesc(differences');
colormap(flipud(cool));  % 使用cool颜色映射
caxis([-3, 3]);

% 设置坐标轴
yticks(1:n_features);
yticklabels(arrayfun(@(x) sprintf('Feature %02d', x), 1:n_features, 'UniformOutput', false));
xticks([]);
ylabel('Features', 'FontSize', 12, 'FontWeight', 'bold');

% 添加显著性标记
sig_indices = find(p_values < 0.05);
for i = 1:length(sig_indices)
    idx = sig_indices(i);
    if p_values(idx) < 0.001
        marker = '***';
    elseif p_values(idx) < 0.01
        marker = '**';
    else
        marker = '*';
    end
    text(1.1, idx, marker, 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'Color', 'r');
end

% 子图2：差异条形图
subplot(2, 3, 3);
barh(differences, 'FaceColor', [0.3, 0.6, 0.8], 'EdgeColor', 'k');
ylim([0.5, n_features+0.5]);
set(gca, 'YTick', []);
xlabel('Difference (Treatment - Control)', 'FontSize', 10);
title('Effect Size', 'FontSize', 11);
grid on;

% 子图3：p值分布
subplot(2, 3, 6);
neg_log10_p = -log10(p_values);
scatter(neg_log10_p, 1:n_features, 50, neg_log10_p, 'filled');
colormap(autumn);
xlabel('-log_{10}(p-value)', 'FontSize', 10);
set(gca, 'YTick', []);
title('Statistical Significance', 'FontSize', 11);
grid on;

% 添加显著性阈值线
sig_threshold = -log10(0.05);
xline(sig_threshold, 'r--', 'LineWidth', 1.5, 'Alpha', 0.7);
text(sig_threshold+0.1, n_features*0.9, 'p=0.05', ...
     'FontSize', 9, 'Color', 'r', 'FontWeight', 'bold');

% 添加颜色条
%c = colorbar('Position', [0.45, 0.05, 0.02, 0.4]);
c.Label.String = 'Difference Magnitude';
c.Label.FontSize = 11;
c.Label.FontWeight = 'bold';

% 添加总标题
sgtitle('Differential Analysis Heatmap: Control vs Treatment', ...
        'FontSize', 14, 'FontWeight', 'bold');

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure4.png');
disp('高级热图3已保存为 Figure4.png');




%% 5 交互式小倍数热图（Small Multiples）
clc; clear; close all;

% 生成多个条件下的数据
rng(321);
n_conditions = 6;
n_timepoints = 20;
n_features = 8;

% 创建数据立方体
data_cube = zeros(n_features, n_timepoints, n_conditions);

for cond = 1:n_conditions
    for feat = 1:n_features
        % 基础趋势
        base_trend = linspace(0, 3, n_timepoints) * (0.5 + rand());
        
        % 条件特异性效应
        cond_effect = sin(2*pi*(1:n_timepoints)/n_timepoints * cond) * 1.5;
        
        % 特征特异性模式
        feat_pattern = cos(2*pi*(1:n_timepoints)/n_timepoints * feat) * 1.2;
        
        % 噪声
        noise = randn(1, n_timepoints) * 0.4;
        
        % 组合
        data_cube(feat, :, cond) = base_trend + cond_effect + feat_pattern + noise + 5;
    end
end

% 创建图形
figure('Position', [200, 200, 950, 750]);

% 创建小倍数热图网格
condition_names = {'Ctrl', 'Drug A', 'Drug B', 'Drug C', 'Combination', 'Vehicle'};
time_labels = arrayfun(@(x) sprintf('T%d', x), 1:n_timepoints, 'UniformOutput', false);
feature_labels = arrayfun(@(x) sprintf('F%d', x), 1:n_features, 'UniformOutput', false);

% 确定子图布局
n_rows = ceil(sqrt(n_conditions));
n_cols = ceil(n_conditions / n_rows);

% 计算全局颜色范围
global_min = min(data_cube(:));
global_max = max(data_cube(:));

for cond = 1:n_conditions
    subplot(n_rows, n_cols, cond);
    
    % 提取当前条件的数据
    current_data = squeeze(data_cube(:, :, cond));
    
    % 绘制热图
    imagesc(current_data);
    colormap(jet);
    caxis([global_min, global_max]);  % 使用统一的颜色范围
    
    % 设置坐标轴
    if cond > n_conditions - n_cols  % 最后一行显示x轴标签
        xticks(1:2:n_timepoints);
        xticklabels(time_labels(1:2:end));
        xtickangle(45);
        xlabel('Time', 'FontSize', 9);
    else
        xticks([]);
    end
    
    if mod(cond, n_cols) == 1  % 第一列显示y轴标签
        yticks(1:n_features);
        yticklabels(feature_labels);
        ylabel('Features', 'FontSize', 9);
    else
        yticks([]);
    end
    
    % 添加条件标题
    title(condition_names{cond}, 'FontSize', 11, 'FontWeight', 'bold');
    
    % 添加网格
    grid on;
    set(gca, 'GridColor', [0.5 0.5 0.5], 'GridAlpha', 0.3, ...
             'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5]);
    
    % 在每个子图中添加数值范围
    text(n_timepoints*0.05, n_features*0.95, ...
         sprintf('[%.1f,%.1f]', min(current_data(:)), max(current_data(:))), ...
         'FontSize', 8, 'BackgroundColor', [1 1 1 0.7], ...
         'HorizontalAlignment', 'left');
end

% 添加全局颜色条
c = colorbar('Position', [0.92, 0.15, 0.02, 0.7]);
c.Label.String = 'Signal Intensity';
c.Label.FontSize = 11;
c.Label.FontWeight = 'bold';

% 添加总标题
sgtitle('Small Multiples Heatmap: Multiple Experimental Conditions', ...
        'FontSize', 14, 'FontWeight', 'bold');

% 添加图例说明
annotation('textbox', [0.35, 0.01, 0.3, 0.05], 'String', ...
           'Each panel shows feature dynamics under different experimental conditions', ...
           'HorizontalAlignment', 'center', 'FontSize', 10, ...
           'EdgeColor', 'none', 'BackgroundColor', [1 1 1 0.8]);

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure5.png');
disp('高级热图5已保存为 Figure5.png');

%% 6. 热图
clc;
clear;
close all;

% X为0-1矩阵 
X = rand(13);
% 获取矩阵的列数
X = round(X, 2);
% 制造全是1的向量
B = ones(1, size(X, 1));
% 替换X矩阵对角元素，使其均为1
X(logical(eye(size(X)))) = B;
data = X;

% 如果color_cell1.mat文件存在则加载，否则使用默认颜色
if exist('color_cell1.mat', 'file')
    load('color_cell1.mat')
    mycolor1 = color_cell1{1, 1};
else
    % 使用parula颜色映射作为替代
    mycolor1 = parula(256);
    warning('color_cell1.mat文件不存在，使用parula颜色映射替代');
end

% 开始绘制热图
% 命名所有变量名字，这里大家可以替换成自己需要的变量名
label_name = {'N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11','N12','N13'};
xlabel_name = label_name;
ylabel_name = label_name;

% 创建图形
figure('Position', [200, 200, 700, 600]);

% 热图函数为heatmap;开始绘制
hot_figure = heatmap(xlabel_name, ylabel_name, data, 'FontName', 'Arial', 'FontSize', 11);
hot_figure.GridVisible = 'off';

% 设置坐标区名字与图的大标题
hot_figure.Title = 'Correlation Matrix Heatmap';
hot_figure.XLabel = 'Variables';
hot_figure.YLabel = 'Variables';
colormap(gca, mycolor1);

% 添加颜色条
colorbar;

% 保存图像为Figure1.png
print(gcf, '-dpng', '-r300', 'Figure6.png');

% 可选：显示保存成功的消息
disp('热图已保存为 Figure6.png');
