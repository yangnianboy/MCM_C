%% 7. 气泡图（多变量可视化）
clc; clear; close all;

% 生成气泡图数据
rng(654);
n_points = 50;

% 生成基础变量
x_data = randn(n_points, 1) * 10 + 50;      % X轴：变量1
y_data = randn(n_points, 1) * 8 + 60;       % Y轴：变量2
z_data = rand(n_points, 1) * 30 + 10;       % 气泡大小：变量3
color_data = rand(n_points, 1);             % 气泡颜色：变量4

% 创建分组
groups = randi([1, 4], n_points, 1);
group_names = {'Group A', 'Group B', 'Group C', 'Group D'};
group_colors = [0.2, 0.4, 0.8;    % 蓝色
                0.8, 0.4, 0.2;    % 橙色
                0.2, 0.8, 0.4;    % 绿色
                0.8, 0.2, 0.8];   % 紫色

% 创建图形
figure('Position', [200, 200, 750, 500]);

% 按组绘制气泡
for g = 1:4
    idx = groups == g;
    
    % 计算气泡大小（与z_data成比例）
    bubble_sizes = z_data(idx) * 10;
    
    % 使用散点图绘制气泡
    scatter(x_data(idx), y_data(idx), bubble_sizes, ...
            group_colors(g,:), 'filled', 'MarkerFaceAlpha', 0.7, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    hold on;
end

% 坐标轴设置
xlim([min(x_data)*0.9, max(x_data)*1.1]);
ylim([min(y_data)*0.9, max(y_data)*1.1]);
set(gca, 'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
grid on;

% 标签和标题
xlabel('Variable X (units)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Variable Y (units)', 'FontSize', 12, 'FontWeight', 'bold');
title('Bubble Chart: Multi-Variable Relationship Visualization', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 添加图例
legend_handles = gobjects(4, 1);
for g = 1:4
    legend_handles(g) = scatter(NaN, NaN, 100, group_colors(g,:), ...
                                'filled', 'MarkerFaceAlpha', 0.7, ...
                                'MarkerEdgeColor', 'k');
end
legend(legend_handles, group_names, 'Location', 'northeastoutside', 'Box', 'off');

% 添加大小图例
size_values = [10, 20, 30];
size_positions = [max(x_data)*0.8, max(y_data)*0.7];
for i = 1:length(size_values)
    scatter(size_positions(1) + i*5, size_positions(2), size_values(i)*10, ...
            [0.5, 0.5, 0.5], 'filled', 'MarkerFaceAlpha', 0.5, ...
            'MarkerEdgeColor', 'k');
    text(size_positions(1) + i*5, size_positions(2) - 8, ...
         sprintf('Z=%.0f', size_values(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9);
end
text(size_positions(1), size_positions(2) + 15, 'Bubble Size:', ...
     'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');

% 添加相关系数
corr_coef = corr(x_data, y_data);
text(min(x_data)*1.05, max(y_data)*0.95, ...
     sprintf('Correlation: r = %.3f', corr_coef), ...
     'HorizontalAlignment', 'left', 'FontSize', 10, ...
     'BackgroundColor', [1 1 1 0.8]);

% 添加回归线
p = polyfit(x_data, y_data, 1);
x_fit = linspace(min(x_data), max(x_data), 100);
y_fit = polyval(p, x_fit);
h = plot(x_fit, y_fit, 'k-', 'LineWidth', 2);
h.Color(4) = 0.5;  % 设置透明度为0.5

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure7.png');

%% 8. 高级气泡图（时间序列气泡）
clc; clear; close all;

% 生成时间序列气泡图数据
rng(987);
n_timepoints = 8;
n_categories = 6;

% 时间点
years = 2015:2022;

% 创建数据矩阵
bubble_matrix = zeros(n_timepoints, n_categories);
for i = 1:n_categories
    base = 20 + rand() * 30;
    growth = rand() * 5;
    for j = 1:n_timepoints
        noise = randn() * 3;
        bubble_matrix(j, i) = max(base + growth * (j-1) + noise, 5);
    end
end

% 创建图形
figure('Position', [200, 200, 850, 550]);

% 定义颜色（替代turbo）
% category_colors = turbo(n_categories);  % 原代码，需要替换

% 使用自定义彩虹色
category_colors = [
    0.8, 0.2, 0.2;    % 红色
    0.8, 0.5, 0.2;    % 橙色
    0.8, 0.8, 0.2;    % 黄色
    0.4, 0.8, 0.2;    % 绿色
    0.2, 0.6, 0.8;    % 蓝色
    0.5, 0.2, 0.8;    % 紫色
];

% 或者使用MATLAB内置颜色映射
% category_colors = jet(n_categories);  % 也可以使用这个

% 绘制气泡时间序列
for i = 1:n_categories
    for j = 1:n_timepoints
        % 气泡大小与值成比例
        bubble_size = bubble_matrix(j, i) * 2;
        
        % 绘制气泡
        scatter(years(j), i, bubble_size, category_colors(i,:), ...
                'filled', 'MarkerFaceAlpha', 0.7, ...
                'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        hold on;
        
        % 添加数值标签（仅显示较大气泡）
        if bubble_size > 80
            text(years(j), i, sprintf('%.1f', bubble_matrix(j, i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'w');
        end
    end
    
    % 添加连接线显示趋势
    h = plot(years, ones(size(years)) * i, 'k-', 'LineWidth', 0.5);
    h.Color(4) = 0.3;  % 设置透明度为0.3
end

% 坐标轴设置
xlim([years(1)-0.5, years(end)+0.5]);
ylim([0.5, n_categories+0.5]);
set(gca, 'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.2, ...
         'YTick', 1:n_categories, 'YTickLabel', ...
         {'Tech', 'Health', 'Finance', 'Energy', 'Retail', 'Auto'});
grid on;

% 标签和标题
xlabel('Year', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Industry Sector', 'FontSize', 12, 'FontWeight', 'bold');
title('Bubble Time Series: Market Share Trends by Industry', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 添加颜色条
colormap(category_colors);
c = colorbar;
c.Label.String = 'Industry Sector';
c.Label.FontSize = 11;
c.Label.FontWeight = 'bold';
c.Ticks = linspace(0, 1, n_categories);
c.TickLabels = {'Tech', 'Health', 'Finance', 'Energy', 'Retail', 'Auto'};

% 添加大小图例
legend_sizes = [20, 40, 60];
legend_x = years(end) + 0.8;
legend_y = n_categories - 1;
for i = 1:length(legend_sizes)
    scatter(legend_x, legend_y - i*0.5, legend_sizes(i)*2, ...
            [0.5, 0.5, 0.5], 'filled', 'MarkerFaceAlpha', 0.7, ...
            'MarkerEdgeColor', 'k');
    text(legend_x + 1, legend_y - i*0.5, sprintf('Value = %.0f', legend_sizes(i)), ...
         'VerticalAlignment', 'middle', 'FontSize', 9);
end

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure8.png');