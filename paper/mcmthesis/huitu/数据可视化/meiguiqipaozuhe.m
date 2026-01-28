%% 12. 多变量气泡玫瑰组合图
clc; clear; close all;

% 生成组合图数据
rng(444);
n_categories = 6;
n_variables = 4;

% 类别名称
categories = {'Group A', 'Group B', 'Group C', 'Group D', 'Group E', 'Group F'};

% 生成多变量数据
data_complex = zeros(n_categories, n_variables);
for i = 1:n_categories
    for j = 1:n_variables
        base = 30 + rand() * 40;
        category_effect = (i-1) * 5;
        variable_effect = (j-1) * 8;
        noise = randn() * 5;
        data_complex(i, j) = base + category_effect + variable_effect + noise;
    end
end

% 创建图形
figure('Position', [200, 200, 900, 600]);

% 使用subplot创建组合视图

% 子图1：玫瑰图视图（不使用polaraxes，手动绘制）
subplot(1, 2, 1);
hold on;

% 计算角度
theta = linspace(0, 2*pi, n_categories+1);
theta = theta(1:end-1);

% 绘制每个变量的玫瑰图
variable_colors = lines(n_variables);
for var = 1:n_variables
    variable_data = data_complex(:, var)';
    
    % 归一化数据
    normalized_data = (variable_data - min(variable_data)) / ...
                      (max(variable_data) - min(variable_data)) * 80 + 20;
    
    % 转换为笛卡尔坐标
    theta_closed = [theta, theta(1)];
    data_closed = [normalized_data, normalized_data(1)];
    [x_poly, y_poly] = pol2cart(theta_closed, data_closed);
    
    % 使用patch绘制填充区域
    patch(x_poly, y_poly, variable_colors(var,:), ...
          'FaceAlpha', 0.3, 'EdgeColor', variable_colors(var,:), ...
          'LineWidth', 2);
end

% 添加网格线
angles = linspace(0, 2*pi, 100);
for r = [20, 50, 80, 100]
    [x_grid, y_grid] = pol2cart(angles, r);
    plot(x_grid, y_grid, 'k:', 'LineWidth', 0.5, 'Color', [0.7 0.7 0.7]);
end

% 添加径向线
for i = 1:n_categories
    [x_line, y_line] = pol2cart(theta(i), 100);
    plot([0, x_line], [0, y_line], 'k:', 'LineWidth', 0.5, 'Color', [0.7 0.7 0.7]);
    
    % 添加角度标签
    [x_label, y_label] = pol2cart(theta(i), 105);
    text(x_label, y_label, categories{i}, ...
         'HorizontalAlignment', 'center', 'FontSize', 9, 'Rotation', rad2deg(theta(i))-90);
end

% 添加径向刻度标签
for r = [20, 50, 80]
    text(0, r, sprintf('%d', r), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 8);
end

% 设置坐标轴
axis equal;
axis([-110, 110, -110, 110]);
axis off;
box on;
title('Radar View: Multi-Variable Profiles', 'FontSize', 12, 'FontWeight', 'bold');

% 子图2：气泡图视图
subplot(1, 2, 2);
hold on;

% 选择两个主要变量进行气泡图展示
x_var = 1;  % 变量1作为X轴
y_var = 2;  % 变量2作为Y轴
size_var = 3;  % 变量3作为气泡大小
color_var = 4;  % 变量4作为气泡颜色

% 提取数据
x_data = data_complex(:, x_var);
y_data = data_complex(:, y_var);
size_data = data_complex(:, size_var);
color_data = data_complex(:, color_var);

% 归一化气泡大小和颜色
size_normalized = (size_data - min(size_data)) / ...
                  (max(size_data) - min(size_data)) * 500 + 100;
color_normalized = (color_data - min(color_data)) / ...
                   (max(color_data) - min(color_data));

% 创建颜色映射
cmap = jet(256);
bubble_colors = cmap(round(color_normalized * 255) + 1, :);

% 绘制气泡图
for i = 1:n_categories
    scatter(x_data(i), y_data(i), size_normalized(i), ...
            bubble_colors(i,:), 'filled', 'MarkerFaceAlpha', 0.7, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1);
    
    % 添加类别标签
    text(x_data(i), y_data(i), categories{i}, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 10, 'FontWeight', 'bold');
end

% 坐标轴设置
xlabel(sprintf('Variable %d', x_var), 'FontSize', 11, 'FontWeight', 'bold');
ylabel(sprintf('Variable %d', y_var), 'FontSize', 11, 'FontWeight', 'bold');
title('Bubble View: Multi-Dimensional Relationships', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% 添加颜色条
colormap(jet);
c = colorbar;
c.Label.String = sprintf('Variable %d', color_var);
c.Label.FontSize = 10;
c.Label.FontWeight = 'bold';

% 添加大小图例
legend_sizes = [min(size_data), median(size_data), max(size_data)];
legend_x = max(x_data) * 1.1;
legend_y = linspace(min(y_data), max(y_data), 3);
for i = 1:3
    size_norm = (legend_sizes(i) - min(size_data)) / ...
                (max(size_data) - min(size_data)) * 500 + 100;
    scatter(legend_x, legend_y(i), size_norm, [0.5, 0.5, 0.5], ...
            'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'k');
    text(legend_x + (max(x_data)-min(x_data))*0.05, legend_y(i), sprintf('%.1f', legend_sizes(i)), ...
         'VerticalAlignment', 'middle', 'FontSize', 9);
end
text(legend_x, max(y_data)*1.05, 'Bubble Size:', ...
     'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');

% 总标题
sgtitle('Multi-Variable Visualization: Combined Radar and Bubble Charts', ...
        'FontSize', 14, 'FontWeight', 'bold');

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure12.png');