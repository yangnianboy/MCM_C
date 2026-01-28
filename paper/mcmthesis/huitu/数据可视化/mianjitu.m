%% 1. 基本面积图（已修改）
clc; clear; close all;

figure('Position', [200, 200, 600, 400])
color = [0.9255, 0.7059, 0.6902;
         0.8627, 0.4510, 0.4667;
         0.8235, 0.6353, 0.6196;
         0.7020, 0.3608, 0.3686;
         0.6118, 0.6980, 0.7882;
         0.2510, 0.5020, 0.6392;
         0.3686, 0.4627, 0.5529;
         0.4118, 0.5255, 0.6157];

x = 1:8;
y1 = [0, 2, 2.5, 2.7, 3, 2.4, 1.9, 1.6];
y2 = [0, 1.2, 1.4, 1.5, 2, 1.6, 1.3, 0.7];
y3 = [0, 1.2, 1.4, 1.3, 1.5, 1.3, 1.0, 0.6];

area(x, y1, 'FaceAlpha', .7, 'FaceColor', color(1,:), 'EdgeColor', color(2,:), 'LineWidth', 2)
hold on
area(x+1, y1, 'FaceAlpha', .7, 'FaceColor', color(3,:), 'EdgeColor', color(4,:), 'LineWidth', 2)
hold on
area(x+5, y2, 'FaceAlpha', .6, 'FaceColor', color(5,:), 'EdgeColor', color(6,:), 'LineWidth', 2)
hold on
area(x+7, y3, 'FaceAlpha', .6, 'FaceColor', color(7,:), 'EdgeColor', color(8,:), 'LineWidth', 2)

ax = gca;
ax.YLim = [0, 4];
set(gca, "FontName", "Arial", "FontSize", 12, "LineWidth", 1.5)
box off
xlabel('Time (months)', 'FontSize', 12, 'FontWeight', 'bold')
ylabel('Expression Level', 'FontSize', 12, 'FontWeight', 'bold')
title('Gene Expression Over Time', 'FontSize', 14, 'FontWeight', 'bold')
legend({'Gene A', 'Gene B', 'Gene C', 'Gene D'}, 'Location', 'northeast', 'Box', 'off')

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure1.png');

%% 2. 堆叠面积图（多类别）
clc; clear; close all;

% 生成示例数据
rng(42);
months = 1:12;
n_categories = 5;

% 创建随时间变化的堆叠数据
data_stacked = zeros(length(months), n_categories);
for i = 1:n_categories
    base = rand() * 10;
    seasonal = 2 * sin(2*pi*(months-3)/12 + rand()*2*pi);
    trend = rand() * 0.3 * months;
    noise = randn(size(months)) * 0.5;
    data_stacked(:, i) = base + seasonal + trend + noise;
    data_stacked(:, i) = max(data_stacked(:, i), 0); % 确保非负
end

% 排序使最稳定的类别在底部
data_stacked = sort(data_stacked, 2, 'ascend');

% 创建图形
figure('Position', [200, 200, 700, 450]);

% 使用自定义配色方案（替代viridis）
% colors = viridis(n_categories);  % 原代码，需要替换
colors = [
    0.267, 0.004, 0.329;  % 深紫色
    0.282, 0.165, 0.459;  % 紫蓝色
    0.231, 0.318, 0.545;  % 蓝色
    0.153, 0.533, 0.557;  % 蓝绿色
    0.122, 0.733, 0.471;  % 绿色
];

% 或者使用MATLAB内置颜色映射
% colors = parula(n_categories);  % 也可以使用这个

% 绘制堆叠面积图
area(months, data_stacked, 'LineStyle', 'none');
hold on;

% 设置颜色
for i = 1:n_categories
    h(i) = area(months, data_stacked(:, i), 'FaceColor', colors(i,:), ...
                'FaceAlpha', 0.7, 'EdgeColor', 'none');
end

% 添加边界线
for i = 1:n_categories
    if i == n_categories
        plot(months, sum(data_stacked(:, 1:i), 2), 'k-', 'LineWidth', 1.2);
    end
end

% 坐标轴设置
xlim([1, 12]);
ylim([0, max(sum(data_stacked, 2)) * 1.1]);
set(gca, 'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
grid on;

% 标签和标题
xlabel('Time (Months)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Cumulative Value', 'FontSize', 12, 'FontWeight', 'bold');
title('Stacked Area Chart: Component Contributions Over Time', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 添加图例
category_names = {'Component A', 'Component B', 'Component C', ...
                  'Component D', 'Component E'};
legend(h, category_names, 'Location', 'northeastoutside', 'Box', 'off');

% 添加百分比标注
percentages = data_stacked(end, :) ./ sum(data_stacked(end, :)) * 100;
for i = 1:n_categories
    y_pos = sum(data_stacked(end, 1:i)) - data_stacked(end, i)/2;
    text(months(end), y_pos, sprintf('%.1f%%', percentages(i)), ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
         'FontSize', 9, 'BackgroundColor', [1 1 1 0.7]);
end

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure2.png');

%% 3. 流式面积图（Streamgraph）
clc; clear; close all;

% 生成流式图数据
rng(123);
time_points = 100;
n_streams = 6;

% 创建正弦波叠加的数据
t = linspace(0, 10*pi, time_points);
data_streams = zeros(time_points, n_streams);

for i = 1:n_streams
    % 每个流有不同的频率和相位
    freq = 0.5 + rand() * 2;
    phase = rand() * 2*pi;
    amplitude = 5 + rand() * 10;
    
    % 创建基础波形
    base_wave = amplitude * sin(freq * t + phase);
    
    % 添加一些随机扰动
    noise = randn(size(t)) * 2;
    
    % 确保非负值
    data_streams(:, i) = max(base_wave + noise + 10, 0);
end

% 对称化数据以创建流式图
data_streams_centered = data_streams - mean(data_streams, 2);

% 创建图形
figure('Position', [200, 200, 800, 500]);

% 使用自定义配色方案
colors = [
    0.941, 0.902, 0.549;  % 浅黄色
    0.941, 0.737, 0.431;  % 橙色
    0.859, 0.427, 0.384;  % 红色
    0.690, 0.208, 0.427;  % 深红色
    0.408, 0.114, 0.408;  % 紫色
    0.157, 0.039, 0.294;  % 深紫色
];

% 绘制流式面积图
bottom = zeros(time_points, 1);
handles = gobjects(n_streams, 1);

for i = 1:n_streams
    % 计算上下边界
    top = bottom + data_streams_centered(:, i);
    
    % 填充面积
    x_fill = [t, fliplr(t)];
    y_fill = [bottom', fliplr(top')];
    
    handles(i) = fill(x_fill, y_fill, colors(i,:), ...
                      'FaceAlpha', 0.7, 'EdgeColor', colors(i,:)*0.7, ...
                      'LineWidth', 0.5);
    hold on;
    
    % 更新底部边界
    bottom = top;
end

% 坐标轴设置
xlim([t(1), t(end)]);
set(gca, 'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.2, ...
         'Box', 'off', 'YTick', []);
grid on;

% 标签和标题
xlabel('Time Progression', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Flow Intensity', 'FontSize', 12, 'FontWeight', 'bold');
title('Streamgraph: Multi-Dimensional Flow Visualization', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 添加图例
stream_names = {'Stream A', 'Stream B', 'Stream C', ...
                'Stream D', 'Stream E', 'Stream F'};
legend(handles, stream_names, 'Location', 'northeastoutside', 'Box', 'off');

% 修复：移除了Alpha参数，使用颜色RGBA格式
center_line = plot([t(1), t(end)], [0, 0], 'k--', 'LineWidth', 1);
set(center_line, 'Color', [0, 0, 0, 0.5]); % 使用RGBA格式设置透明度

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure3.png');

%% 4. 带置信区间的面积图
clc; clear; close all;

% 生成时间序列数据与置信区间
rng(456);
n_timepoints = 50;
time = linspace(0, 20, n_timepoints);

% 创建基础信号
signal = 10 * sin(time * 0.5) + 2 * cos(time * 1.2) + 15;
noise_level = 2;

% 生成带噪声的重复测量
n_replicates = 10;
measurements = zeros(n_timepoints, n_replicates);
for i = 1:n_replicates
    noise = randn(size(time)) * noise_level;
    measurements(:, i) = signal + noise + rand() * 3;
end

% 计算统计量
mean_signal = mean(measurements, 2);
std_signal = std(measurements, 0, 2);
ci_upper = mean_signal + 1.96 * std_signal / sqrt(n_replicates);
ci_lower = mean_signal - 1.96 * std_signal / sqrt(n_replicates);

% 创建图形
figure('Position', [200, 200, 750, 450]);

% 绘制置信区间区域
fill([time, fliplr(time)], [ci_upper', fliplr(ci_lower')], ...
     [0.3, 0.6, 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on;

% 绘制均值线
plot(time, mean_signal, 'b-', 'LineWidth', 3, 'Color', [0.2, 0.4, 0.8]);

% 绘制原始数据点（部分显示）
plot_indices = 1:3:n_timepoints;
scatter(time(plot_indices), mean_signal(plot_indices), 40, ...
        [0.2, 0.4, 0.8], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);

% 坐标轴设置
xlim([time(1), time(end)]);
ylim([min(ci_lower)*0.9, max(ci_upper)*1.1]);
set(gca, 'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
grid on;

% 标签和标题
xlabel('Experimental Time (hours)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Response Amplitude (μV)', 'FontSize', 12, 'FontWeight', 'bold');
title('Time Series with 95% Confidence Interval', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 添加图例
legend({'95% CI', 'Mean Signal', 'Data Points'}, ...
       'Location', 'northeast', 'Box', 'off');

% 添加统计信息文本
text(time(end)*0.05, max(ci_upper)*0.95, ...
     sprintf('N = %d replicates\nMean ± 95%% CI', n_replicates), ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
     'FontSize', 10, 'BackgroundColor', [1 1 1 0.8]);

% 保存图像
print(gcf, '-dpng', '-r300', 'Figure4.png');
