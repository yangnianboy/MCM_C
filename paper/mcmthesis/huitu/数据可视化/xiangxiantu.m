%% 箱线图
clc;clear;close all;
load data.mat;
% 坐标区域每组变量之间的标签
X ={' ','M1','M2','M3','M4'};
% 设置画图背景为白色
figure('color',[1 1 1],'Position',[200,200,600,400]);
% 设置配色
data1=[3.85785000000000	4.56787500000000	3.85785000000000	4.16950000000000
3.76887000000000	4.36350000000000	3.76887000000000	4.30075000000000
3.22700000000000	4.51150000000000	3.22700000000000	5.05450000000000];

mycolor1 = [220 211 30;180 68 108;242 166 31;244 146 121;59 125 183]./255;


% 开始绘图
box_figure = boxplot(data1,'color',[0 0 0],'Symbol','o');

set(box_figure,'Linewidth',1.2);
boxobj = findobj(gca,'Tag','Box');

for i = 1:size(data1,2)
    patch(get(boxobj(i),'XData'),get(boxobj(i),'YData'),mycolor1(i,:),'FaceAlpha',0.5,...
        'LineWidth',1.1);
end
hold on;

xlabel('x');
ylabel('y');
title('title')

% 设置坐标区域的参数
set(gca,"FontName","Times New Roman",'Fontsize',12,'Linewidth',1.1); %设置坐标区的线宽

% 对X轴刻度与显示范围调整
set(gca,'Xlim',[0.5 5], 'Xtick', [0:1:5],'Xticklabel',X);
% 对Y轴刻度与显示范围调整

% 对刻度长度与刻度显示位置调整
set(gca, 'TickDir', 'in', 'TickLength', [.008 .008]);

saveas(gcf, 'Figure1.png'); % 将当前图窗保存为PNG格式



%% 2. 带异常值标注和统计信息的箱线图
clc; clear; close all;

% 生成示例数据（模拟实验数据）
rng(42); % 设置随机种子确保可重复性
data_groups = cell(1, 4);
for i = 1:4
    % 每组数据包含不同分布
    n_samples = 40 + randi(20);
    base_value = 10 + i*5;
    data_groups{i} = base_value + randn(n_samples, 1)*3;
    
    % 添加一些异常值
    n_outliers = randi(5);
    outlier_indices = randperm(n_samples, n_outliers);
    data_groups{i}(outlier_indices) = data_groups{i}(outlier_indices) + 10 + randn(n_outliers, 1)*5;
end

% 转换为矩阵格式用于boxplot
max_len = max(cellfun(@length, data_groups));
data_matrix = nan(max_len, 4);
for i = 1:4
    data_matrix(1:length(data_groups{i}), i) = data_groups{i};
end

% 创建图形
figure('color', [1 1 1], 'Position', [200, 200, 700, 450]);

% 高级配色方案
mycolor = [0.2, 0.6, 0.8;    % 蓝色
           0.8, 0.4, 0.4;    % 红色
           0.4, 0.8, 0.6;    % 绿色
           0.8, 0.6, 0.2];   % 橙色

% 绘制箱线图
box_figure = boxplot(data_matrix, 'Colors', 'k', 'Symbol', 'k+', 'Widths', 0.7);
set(box_figure, 'LineWidth', 1.5);

% 填充箱体颜色
boxobj = findobj(gca, 'Tag', 'Box');
for i = 1:4
    patch(get(boxobj(i), 'XData'), get(boxobj(i), 'YData'), ...
          mycolor(i,:), 'FaceAlpha', 0.6, 'LineWidth', 1.2, 'EdgeColor', 'k');
end

hold on;

% 添加数据点（抖动散点）
for i = 1:4
    x_jitter = i + (rand(size(data_groups{i})) - 0.5) * 0.3;
    scatter(x_jitter, data_groups{i}, 30, 'k', 'filled', 'MarkerFaceAlpha', 0.4);
end

% 计算并显示统计信息
stats_text = cell(4, 1);
for i = 1:4
    group_data = data_groups{i};
    median_val = median(group_data);
    mean_val = mean(group_data);
    std_val = std(group_data);
    
    % 在箱体上方显示统计信息
    stats_text{i} = sprintf('N=%d\nM=%.2f\nSD=%.2f', ...
                           length(group_data), mean_val, std_val);
    
    text(i, max(group_data)*1.05, stats_text{i}, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontName', 'Arial', 'BackgroundColor', [1 1 1 0.8]);
end

% 坐标轴设置
xlabel('Experimental Groups', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial');
ylabel('Measurement Value (units)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial');
title('Box Plot with Outliers and Statistical Summary', ...
      'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Arial');

set(gca, 'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.2, ...
         'XTickLabel', {'Control', 'Treatment A', 'Treatment B', 'Treatment C'}, ...
         'Box', 'on', 'GridLineStyle', '--');
grid on;

% 添加显著性比较线（示例）
comparison_y = max(cellfun(@max, data_groups)) * 1.15;
comparisons = {[1, 2], [2, 3], [3, 4]};
sig_symbols = {'*', '**', 'ns'};

for idx = 1:length(comparisons)
    pair = comparisons{idx};
    x1 = pair(1);
    x2 = pair(2);
    
    % 绘制比较线
    plot([x1, x1, x2, x2], [comparison_y-idx*0.5, comparison_y-idx*0.5+0.2, ...
                            comparison_y-idx*0.5+0.2, comparison_y-idx*0.5], 'k-', 'LineWidth', 1);
    
    % 添加显著性标记
    text(mean([x1, x2]), comparison_y-idx*0.5+0.3, sig_symbols{idx}, ...
         'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

% 调整显示范围
ylim([min(cellfun(@min, data_groups))*0.9, comparison_y+1]);

% 添加图例
legend_elements = [patch(NaN, NaN, mycolor(1,:), 'FaceAlpha', 0.6), ...
                   scatter(NaN, NaN, 30, 'k', 'filled', 'MarkerFaceAlpha', 0.4)];
legend(legend_elements, {'Box (IQR)', 'Individual Data Points'}, ...
       'Location', 'northwest', 'Box', 'off', 'FontName', 'Arial');

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure2.png');
% close(gcf);

%% 3. 分组对比箱线图（子组分析）
clc; clear; close all;

% 生成复杂分组数据
% 结构：2个主要组 × 3个子组 × 每个子组有数据
rng(123);
main_groups = {'Male', 'Female'};
sub_groups = {'Young', 'Middle', 'Old'};
n_per_subgroup = 30;

% 创建数据结构
data_complex = cell(2, 3);
for main_idx = 1:2
    for sub_idx = 1:3
        % 生成基于组别的数据
        base_value = 50 + main_idx*10 + sub_idx*5;
        if main_idx == 1 && sub_idx == 1
            % 对照组
            data_complex{main_idx, sub_idx} = base_value + randn(n_per_subgroup, 1)*8;
        else
            % 处理组
            treatment_effect = 15 + randn(n_per_subgroup, 1)*3;
            data_complex{main_idx, sub_idx} = base_value + treatment_effect + randn(n_per_subgroup, 1)*8;
        end
    end
end

% 创建图形
figure('color', [1 1 1], 'Position', [200, 200, 900, 500]);

% 定义配色方案
main_colors = [0.3, 0.5, 0.8;   % 男性 - 蓝色系
               0.8, 0.3, 0.5];   % 女性 - 红色系

% 位置计算
group_spacing = 1;
positions = [];
group_labels = {};
color_matrix = [];

pos_counter = 1;
for main_idx = 1:2
    for sub_idx = 1:3
        positions = [positions, pos_counter];
        
        % 创建组标签
        group_labels{end+1} = sprintf('%s\n%s', main_groups{main_idx}, sub_groups{sub_idx});
        
        % 分配颜色（基于主要组别）
        color_matrix = [color_matrix; main_colors(main_idx, :)];
        
        pos_counter = pos_counter + 1;
    end
    % 添加主要组之间的间隔
    pos_counter = pos_counter + 0.5;
end

% 准备数据矩阵
max_len = max(cellfun(@length, data_complex(:)));
data_plot = nan(max_len, length(positions));
for idx = 1:length(positions)
    data_plot(1:length(data_complex{idx}), idx) = data_complex{idx};
end

% 绘制箱线图
box_figure = boxplot(data_plot, 'Positions', positions, 'Colors', 'k', ...
                     'Symbol', 'o', 'Widths', 0.5);
set(box_figure, 'LineWidth', 1.3);

% 填充颜色
boxobj = findobj(gca, 'Tag', 'Box');
for i = 1:length(boxobj)
    patch(get(boxobj(i), 'XData'), get(boxobj(i), 'YData'), ...
          color_matrix(i, :), 'FaceAlpha', 0.5, 'LineWidth', 1.2, 'EdgeColor', 'k');
end

hold on;

% 添加主要组分隔线
sep_positions = [3.75, 7.25];
for sep_pos = sep_positions
    plot([sep_pos, sep_pos], [min(data_plot(:))*0.9, max(data_plot(:))*1.1], ...
         'k--', 'LineWidth', 1, 'Color', [0.5 0.5 0.5 0.7]);
end

% 添加主要组标签
text(2, max(data_plot(:))*1.05, main_groups{1}, ...
     'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
     'BackgroundColor', [1 1 1 0.8]);
text(5.5, max(data_plot(:))*1.05, main_groups{2}, ...
     'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
     'BackgroundColor', [1 1 1 0.8]);

% 坐标轴设置
xlim([0.5, positions(end)+0.5]);
ylim([min(data_plot(:))*0.9, max(data_plot(:))*1.15]);

set(gca, 'XTick', positions, 'XTickLabel', group_labels, ...
         'FontName', 'Arial', 'FontSize', 10, 'LineWidth', 1.2, ...
         'XTickLabelRotation', 0);

ylabel('Biomarker Concentration (ng/mL)', 'FontSize', 12, 'FontWeight', 'bold');
title('Grouped Box Plot: Age and Gender Effects on Biomarker Levels', ...
      'FontSize', 14, 'FontWeight', 'bold');

grid on;
box on;

% 添加水平参考线（示例：临床阈值）
clinical_threshold = 80;
yline(clinical_threshold, 'r--', 'LineWidth', 1.5, 'Alpha', 0.7);
text(positions(end)+0.2, clinical_threshold, 'Clinical Threshold', ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
     'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');

% 计算并显示p值（示例）
p_values = [0.001, 0.045, 0.320];  % 示例p值
for sub_idx = 1:3
    % 计算子组间的差异
    x_pos = mean([positions(sub_idx), positions(sub_idx+3)]);
    
    if p_values(sub_idx) < 0.001
        sig_text = '***';
    elseif p_values(sub_idx) < 0.01
        sig_text = '**';
    elseif p_values(sub_idx) < 0.05
        sig_text = '*';
    else
        sig_text = 'ns';
    end
    
    % 绘制连接线
    plot([positions(sub_idx), positions(sub_idx+3)], ...
         [max(data_plot(:, [sub_idx, sub_idx+3]), [], 'all')*1.08, ...
          max(data_plot(:, [sub_idx, sub_idx+3]), [], 'all')*1.08], ...
         'k-', 'LineWidth', 1);
    
    % 添加p值标记
    text(x_pos, max(data_plot(:, [sub_idx, sub_idx+3]), [], 'all')*1.1, sig_text, ...
         'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure3.png');
% close(gcf);

%% 4. 小提琴图-箱线图组合（高级数据可视化）
clc; clear; close all;

% 生成多模态数据
rng(456);
n_samples = 100;
groups = {'Group A', 'Group B', 'Group C', 'Group D'};
n_groups = length(groups);

% 创建多模态分布数据
data_violin = cell(1, n_groups);

% 组A：正态分布
data_violin{1} = randn(n_samples, 1)*10 + 50;

% 组B：双峰分布
data_violin{2} = [randn(n_samples/2, 1)*8 + 40; 
                  randn(n_samples/2, 1)*6 + 65];

% 组C：偏态分布
data_violin{3} = exprnd(15, n_samples, 1) + 30;

% 组D：均匀分布
data_violin{4} = unifrnd(30, 70, n_samples, 1);

% 创建图形
figure('color', [1 1 1], 'Position', [200, 200, 850, 550]);

% 定义渐变色
violin_colors = [0.2, 0.4, 0.8;    % 蓝色
                 0.8, 0.3, 0.3;    % 红色
                 0.3, 0.7, 0.4;    % 绿色
                 0.7, 0.5, 0.2];   % 橙色

% 绘制小提琴图（核密度估计）
violin_width = 0.4;
positions = 1:n_groups;

for i = 1:n_groups
    % 核密度估计
    [density, value] = ksdensity(data_violin{i});
    density = density / max(density) * violin_width;  % 归一化宽度
    
    % 绘制小提琴图的左右边界
    fill([positions(i)-density, fliplr(positions(i)+density)], ...
         [value, fliplr(value)], violin_colors(i,:), ...
         'FaceAlpha', 0.3, 'EdgeColor', violin_colors(i,:), 'LineWidth', 1);
    hold on;
    
    % 绘制中位数线
    median_val = median(data_violin{i});
    plot([positions(i)-violin_width/2, positions(i)+violin_width/2], ...
         [median_val, median_val], 'k-', 'LineWidth', 2);
end

% 准备数据矩阵用于箱线图
% 将cell数组转换为矩阵，缺失值用NaN填充
max_len = max(cellfun(@length, data_violin));
box_data = nan(max_len, n_groups);
for i = 1:n_groups
    box_data(1:length(data_violin{i}), i) = data_violin{i};
end

% 在每个小提琴图上叠加箱线图
box_handle = boxplot(box_data, 'Positions', positions, ...
                     'Colors', 'k', 'Symbol', 'k.', 'Widths', 0.25);
set(box_handle, 'LineWidth', 1.5);

% 添加数据点（抖动散点）
point_alpha = 0.4;
point_size = 25;
for i = 1:n_groups
    % 添加随机抖动
    x_jitter = positions(i) + (rand(size(data_violin{i})) - 0.5) * violin_width * 0.8;
    scatter(x_jitter, data_violin{i}, point_size, ...
            violin_colors(i,:), 'filled', 'MarkerFaceAlpha', point_alpha);
end

% 计算并显示描述性统计
stats_table = cell(n_groups, 4);
for i = 1:n_groups
    group_data = data_violin{i};
    stats_table{i, 1} = sprintf('N = %d', length(group_data));
    stats_table{i, 2} = sprintf('Mean = %.2f', mean(group_data));
    stats_table{i, 3} = sprintf('Median = %.2f', median(group_data));
    stats_table{i, 4} = sprintf('SD = %.2f', std(group_data));
    
    % 在图上显示关键统计量
    text(positions(i), min(group_data)*0.95, stats_table{i, 1}, ...
         'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

% 坐标轴设置
xlim([0.5, n_groups+0.5]);

% 计算y轴范围
min_vals = zeros(1, n_groups);
max_vals = zeros(1, n_groups);
for i = 1:n_groups
    min_vals(i) = min(data_violin{i});
    max_vals(i) = max(data_violin{i});
end

ylim([min(min_vals)*0.9, max(max_vals)*1.1]);

set(gca, 'XTick', positions, 'XTickLabel', groups, ...
         'FontName', 'Arial', 'FontSize', 12, 'LineWidth', 1.5, ...
         'Box', 'on');

ylabel('Measurement Value (units)', 'FontSize', 13, 'FontWeight', 'bold');
title('Violin-Box Plot Combination Showing Data Distribution Characteristics', ...
      'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

grid on;

% 添加分布类型标注
dist_types = {'Normal', 'Bimodal', 'Right-Skewed', 'Uniform'};
for i = 1:n_groups
    % 获取当前组的最大值
    current_max = max(data_violin{i});
    % 计算标注位置
    y_pos = current_max * 1.05;
    
    text(positions(i), y_pos, dist_types{i}, ...
         'HorizontalAlignment', 'center', 'FontSize', 10, 'FontStyle', 'italic');
end

% 添加图例
legend_elements = [
    fill(NaN, NaN, violin_colors(1,:), 'FaceAlpha', 0.3, 'EdgeColor', violin_colors(1,:)),
    plot(NaN, NaN, 'k-', 'LineWidth', 2),
    scatter(NaN, NaN, point_size, violin_colors(1,:), 'filled', 'MarkerFaceAlpha', point_alpha)
];

legend(legend_elements, {'Kernel Density', 'Median Line', 'Individual Points'}, ...
       'Location', 'northwest', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 10);

% 添加直方图子图（展示分布详情）
inset_pos = [0.65, 0.15, 0.25, 0.25];
axes('Position', inset_pos);
hold on;

% 绘制所有组的直方图（重叠）
% 计算所有数据的范围用于直方图分箱
all_data = [];
for i = 1:n_groups
    all_data = [all_data; data_violin{i}];
end
bin_edges = linspace(min(all_data), max(all_data), 30);
for i = 1:n_groups
    histogram(data_violin{i}, bin_edges, ...
              'FaceColor', violin_colors(i,:), 'FaceAlpha', 0.4, ...
              'EdgeColor', violin_colors(i,:), 'LineWidth', 0.5);
end

set(gca, 'FontName', 'Arial', 'FontSize', 8, 'Box', 'on');
xlabel('Value', 'FontSize', 9);
ylabel('Frequency', 'FontSize', 9);
title('Combined Histogram', 'FontSize', 10);

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure4.png');

