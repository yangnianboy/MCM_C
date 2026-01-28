%% 9. 玫瑰图（雷达图变体）
clc; clear; close all;

% 生成玫瑰图数据
rng(111);
n_categories = 8;

% 创建类别名称
categories = {'Speed', 'Accuracy', 'Precision', 'Recall', ...
              'F1-Score', 'Robustness', 'Scalability', 'Efficiency'};

% 生成性能数据（两个条件对比）
performance_A = rand(1, n_categories) * 60 + 40;  % 方法A
performance_B = rand(1, n_categories) * 60 + 40;  % 方法B
performance_B = performance_B .* (0.8 + rand(1, n_categories)*0.4); % 添加一些差异

% 确保最大值不超过100
performance_A = min(performance_A, 100);
performance_B = min(performance_B, 100);

% 创建图形
figure('Position', [200, 200, 700, 600]);

% 方法1：创建笛卡尔坐标系，手动绘制极坐标图（推荐）
% 计算角度
theta = linspace(0, 2*pi, n_categories+1);
theta = theta(1:end-1);  % 移除重复的2π点

% 转换为笛卡尔坐标
[x_a, y_a] = pol2cart([theta, theta(1)], [performance_A, performance_A(1)]);
[x_b, y_b] = pol2cart([theta, theta(1)], [performance_B, performance_B(1)]);

% 在笛卡尔坐标系中绘制雷达图
subplot(1, 1, 1);
hold on;

% 绘制方法A的玫瑰图
h1 = fill(x_a, y_a, 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 2);
plot(x_a, y_a, 'b-', 'LineWidth', 3);

% 绘制方法B的玫瑰图
h2 = fill(x_b, y_b, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r', 'LineWidth', 2);
plot(x_b, y_b, 'r-', 'LineWidth', 3);

% 添加网格线
angles = linspace(0, 2*pi, 100);
for r = [20, 40, 60, 80, 100]
    [x_grid, y_grid] = pol2cart(angles, r * ones(size(angles)));
    plot(x_grid, y_grid, 'k:', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5]);
end

% 添加径向线
for i = 1:n_categories
    [x_line, y_line] = pol2cart(theta(i), 100);
    plot([0, x_line], [0, y_line], 'k:', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5]);
    
    % 添加角度标签
    [x_label, y_label] = pol2cart(theta(i), 110);
    text(x_label, y_label, categories{i}, ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end

% 设置坐标轴
axis equal;
axis([-120, 120, -120, 120]);
axis off;

% 添加径向刻度标签
for r = [20, 40, 60, 80, 100]
    text(0, r, sprintf('%d', r), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9);
end

% 标题
title('Radar Chart: Performance Comparison of Two Methods', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 添加图例
legend([h1, h2], {'Method A', 'Method B'}, ...
       'Location', 'southoutside', 'Box', 'off', 'Orientation', 'horizontal');

% 添加性能差值标注
for i = 1:n_categories
    diff_val = performance_B(i) - performance_A(i);
    if abs(diff_val) > 5
        angle = theta(i);
        r_pos = max(performance_A(i), performance_B(i)) + 5;
        [x, y] = pol2cart(angle, r_pos);
        
        if diff_val > 0
            color = 'r';
            symbol = '↑';
        else
            color = 'b';
            symbol = '↓';
        end
        
        text(x, y, sprintf('%s%.1f', symbol, abs(diff_val)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'FontSize', 9, 'FontWeight', 'bold', 'Color', color);
    end
end

% 添加总体得分
text(0, -90, sprintf('Mean Score\nMethod A: %.1f\nMethod B: %.1f', ...
                    mean(performance_A), mean(performance_B)), ...
     'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure9.png');

%% 10. 高级玫瑰图（多组对比）
clc; clear; close all;

% 生成多组对比玫瑰图数据
rng(222);
n_metrics = 10;
n_groups = 4;

% 指标名称
metrics = {'Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5', ...
           'Metric 6', 'Metric 7', 'Metric 8', 'Metric 9', 'Metric 10'};

% 组名和颜色
group_names = {'Control', 'Treatment A', 'Treatment B', 'Treatment C'};
group_colors = [0.2, 0.2, 0.8;    % 蓝色
                0.8, 0.2, 0.2;    % 红色
                0.2, 0.8, 0.2;    % 绿色
                0.8, 0.5, 0.2];   % 橙色

% 生成数据
data_groups = zeros(n_groups, n_metrics);
for g = 1:n_groups
    base = 40 + rand() * 30;
    pattern = sin(linspace(0, 2*pi, n_metrics)) * 15;
    noise = randn(1, n_metrics) * 5;
    data_groups(g, :) = base + pattern * g/2 + noise;
    data_groups(g, :) = min(max(data_groups(g, :), 10), 90);
end

% 创建图形
figure('Position', [200, 200, 800, 650]);

% 创建子图网格
for g = 1:n_groups
    subplot(2, 2, g);
    hold on;
    
    % 计算角度
    theta = linspace(0, 2*pi, n_metrics+1);
    theta = theta(1:end-1);
    
    % 当前组的数据
    current_data = data_groups(g, :);
    theta_closed = [theta, theta(1)];
    data_closed = [current_data, current_data(1)];
    
    % 转换为笛卡尔坐标
    [x_poly, y_poly] = pol2cart(theta_closed, data_closed);
    
    % 使用patch在笛卡尔坐标系中绘制填充区域
    h_fill = patch(x_poly, y_poly, group_colors(g,:), ...
                   'FaceAlpha', 0.4, 'EdgeColor', group_colors(g,:), ...
                   'LineWidth', 1.5);
    
    % 绘制边界线
    plot(x_poly, y_poly, 'Color', group_colors(g,:), 'LineWidth', 2.5);
    
    % 添加网格线
    angles = linspace(0, 2*pi, 100);
    for r = [30, 60, 90]
        [x_grid, y_grid] = pol2cart(angles, r);
        plot(x_grid, y_grid, 'k:', 'LineWidth', 0.5, 'Color', [0.7 0.7 0.7]);
    end
    
    % 添加径向线
    for i = 1:n_metrics
        [x_line, y_line] = pol2cart(theta(i), 100);
        plot([0, x_line], [0, y_line], 'k:', 'LineWidth', 0.5, 'Color', [0.7 0.7 0.7]);
    end
    
    % 设置坐标轴
    axis equal;
    axis([-110, 110, -110, 110]);
    axis off;
    box on;
    
    % 子图标题
    title(group_names{g}, 'FontSize', 12, 'FontWeight', 'bold', ...
          'Color', group_colors(g,:));
    
    % 在中心显示平均得分
    mean_score = mean(current_data);
    text(0, 0, sprintf('Avg:\n%.1f', mean_score), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 10, 'FontWeight', 'bold');
    
    % 标注最大值指标
    [max_val, max_idx] = max(current_data);
    max_angle = theta(max_idx);
    [x_star, y_star] = pol2cart(max_angle, max_val + 8);
    text(x_star, y_star, '★', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 14, 'Color', 'r', 'FontWeight', 'bold');
end

% 添加共享的指标标签（外部添加）
annotation('textbox', [0.4, 0.02, 0.2, 0.05], 'String', 'Metrics (clockwise from top)', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
           'FontSize', 11, 'FontWeight', 'bold', 'EdgeColor', 'none');

% 添加总标题
sgtitle('Multi-Group Radar Chart Comparison', ...
        'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Arial');

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure10.png');

%% 11. 3D玫瑰图（极坐标堆叠）
clc; clear; close all;

% 生成3D玫瑰图数据
rng(333);
n_segments = 12;      % 径向分段数
n_layers = 5;         % 堆叠层数

% 生成堆叠数据
data_3d = zeros(n_segments, n_layers);
for layer = 1:n_layers
    base = 10 * layer;
    pattern = sin(linspace(0, 4*pi, n_segments)) * (3 + rand()*2);
    noise = randn(1, n_segments) * 2;
    data_3d(:, layer) = base + pattern + noise;
    data_3d(:, layer) = max(data_3d(:, layer), 1);
end

% 创建图形
figure('Position', [200, 200, 750, 600]);

% 计算累积高度
cumulative_data = cumsum(data_3d, 2);

% 定义颜色方案（从底到顶渐变）
layer_colors = parula(n_layers);

% 转换为笛卡尔坐标并绘制
theta = linspace(0, 2*pi, n_segments+1);
theta = theta(1:end-1);

% 绘制每一层
for layer = n_layers:-1:1  % 从顶层开始绘制
    if layer == 1
        bottom = zeros(1, n_segments);
    else
        bottom = cumulative_data(:, layer-1)';
    end
    top = cumulative_data(:, layer)';
    
    % 扩展数据以闭合图形
    theta_ext = [theta, theta(1)];
    bottom_ext = [bottom, bottom(1)];
    top_ext = [top, top(1)];
    
    % 转换为笛卡尔坐标
    [x_bottom, y_bottom] = pol2cart(theta_ext, bottom_ext);
    [x_top, y_top] = pol2cart(theta_ext, top_ext);
    
    % 绘制3D效果（使用补片）
    fill3([x_bottom, fliplr(x_top)], [y_bottom, fliplr(y_top)], ...
          layer*ones(1, 2*length(theta_ext)), ...
          layer_colors(layer,:), 'FaceAlpha', 0.7, 'EdgeColor', 'k', ...
          'LineWidth', 0.5);
    hold on;
end

% 设置3D视图
view(45, 30);
axis equal;
axis off;
grid on;

% 设置坐标轴属性
set(gca, 'FontName', 'Arial', 'FontSize', 10, 'Box', 'on');
xlim([-max(cumulative_data(:))*1.2, max(cumulative_data(:))*1.2]);
ylim([-max(cumulative_data(:))*1.2, max(cumulative_data(:))*1.2]);
zlim([0, n_layers+1]);

% 标题
title('3D Polar Stacked Area Chart', 'FontSize', 14, 'FontWeight', 'bold');

% 添加颜色条
colormap(layer_colors);
c = colorbar;
c.Label.String = 'Layer Index';
c.Label.FontSize = 11;
c.Label.FontWeight = 'bold';
c.Ticks = 1:n_layers;
c.TickLabels = arrayfun(@(x) sprintf('Layer %d', x), 1:n_layers, 'UniformOutput', false);

% 添加角度标注
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330];
for i = 1:length(angles)
    angle_rad = deg2rad(angles(i));
    r = max(cumulative_data(:)) * 1.1;
    [x, y] = pol2cart(angle_rad, r);
    text(x, y, 0, sprintf('%d°', angles(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 9, 'FontWeight', 'bold');
    
    % 添加径向线
    h = plot3([0, x], [0, y], [0, 0], 'k-', 'LineWidth', 0.5);
    h.Color(4) = 0.3;  % 设置透明度为0.3
end

% 添加同心圆标注
for r = [20, 40, 60]
    theta_circle = linspace(0, 2*pi, 100);
    [x_circle, y_circle] = pol2cart(theta_circle, r);
    h = plot3(x_circle, y_circle, zeros(size(x_circle)), 'k--', 'LineWidth', 0.5);
    h.Color(4) = 0.3;  % 设置透明度为0.3
    text(r, 0, 0, sprintf('R=%.0f', r), 'FontSize', 9, 'HorizontalAlignment', 'left');
end

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure11.png');