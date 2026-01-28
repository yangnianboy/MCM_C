%% 柱状图（竖向）
clc; clear; close all;
color = [0.7176,0.6824,0.7412; ...
         0.8078,0.7922,0.8902; ...
         0.5804,0.5255,0.7294; ...
         0.3922,0.3451,0.4706; ...
         0.5608,0.5333,0.7412; ...
         0.3725,0.2824,0.6000; ...
         0.0157,0.0196,0.0157];

y = [28, 34, 18, 13];
x = 1:4;

fig = figure('Units', 'pixels', 'Position', [100, 100, 460, 275]);
width = 0.7;
for i = 1:length(y)
    b = bar(i, y(i), width);
    set(b, 'FaceColor', color(i,:), 'EdgeColor', color(i+3,:), 'LineWidth', 2);
    hold on;
end

ylabel('Time on Warm Floor (%)');
xticks(x);
xticklabels({'Object1','Object2','Object3','Object4'});
box off;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 2);
title('TPT Timecourse', 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 2);

% Save
saveas(fig, 'Figure1.png');
close(fig);

%% 柱状图（横向）
clc; clear; close all;
color = [0.1804,0.7804,0.7882; ...
         0.7137,0.6353,0.8706; ...
         0.3529,0.6941,0.9373; ...
         1.0000,0.7255,0.5020];

y = [28, 34, 18, 13];
x = 1:4;

fig = figure('Units', 'pixels', 'Position', [100, 100, 500, 305]);
width = 0.7;
for i = 1:length(y)
    b = barh(i, y(i), width);
    set(b, 'FaceColor', color(i,:), 'EdgeColor', color(i,:), 'LineWidth', 2);
    hold on;
end

xlabel('Time on Warm Floor (%)');
yticks(x);
yticklabels({'Object1','Object2','Object3','Object4'});
box off;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 2);
title('TPT Timecourse', 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 2);
set(gca, 'XGrid', 'on');

% Save
saveas(fig, 'Figure2.png');
close(fig);

%% 分组柱状图（带误差线）
clc; clear; close all;
color = [0.7412,0.7294,0.7255; ...
         0.5255,0.6235,0.7529; ...
         0.6314,0.8039,0.8353; ...
         0.5882,0.5765,0.5765; ...
         0.0745,0.4078,0.6078; ...
         0.4549,0.7373,0.7765; ...
         0.0157,0.0196,0.0157];

data = [1.5, 4, 5; 18, 24, 25; 6, 7, 8];
erro_data = [1,1,3; 6,0.5,1; 0.2,0.1,2];

fig = figure;
b = bar(data);
hold on;
ax = gca;
x_data = zeros(size(data));
for i = 1:size(data,2)
    x_data(:,i) = b(i).XEndPoints';
    errorbar(x_data(:,i), data(:,i), erro_data(:,i), ...
        'LineStyle','none','Color',color(i+3,:),'LineWidth',2,'CapSize',10);
end

ax.YLim = [0, 30];
for i = 1:3
    b(i).FaceColor = color(i,:);
    b(i).EdgeColor = color(i+3,:);
    b(i).LineWidth = 1.5;
end
ax.XTickLabels = {'CT', 'WT', 'RWD'};
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
legend('Object1','Object2','Object3', 'Box', 'off');

% Save
saveas(fig, 'Figure3.png');
close(fig);

%% 分组柱状图（深色配色 + 黑边）
clc; clear; close all;
color = [0.6353,0.1686,0.1686; ...
         0.4627,0.1216,0.1176; ...
         0,0,0];

data = [1.5, 4, 5; 18, 24, 25; 6, 7, 8];
erro_data = [1,1,3; 6,0.5,1; 0.2,0.1,2];

fig = figure;
b = bar(data);
hold on;
ax = gca;
x_data = zeros(size(data));
for i = 1:size(data,2)
    x_data(:,i) = b(i).XEndPoints';
    errorbar(x_data(:,i), data(:,i), erro_data(:,i), ...
        'LineStyle','none','Color','k','LineWidth',2,'CapSize',10);
end

ax.YLim = [0, 30];
for i = 1:3
    b(i).FaceColor = color(i,:);
    b(i).EdgeColor = 'k';
    b(i).LineWidth = 1.5;
end
ax.XTickLabels = {'CT', 'WT', 'RWD'};
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 2);
box off;
legend('Data1','Data2','Data3', 'Box', 'off');

% Save
saveas(fig, 'Figure4.png');
close(fig);

%% 堆叠柱状图（三类）
clc; clear; close all;

color = [0.5020, 0.6706, 0.7373; ...
         0.9961, 0.8275, 0.4824; ...
         0.7373, 0.4902, 0.7098];

data = [1.5, 2.5, 1; ...
        18, 6, 1; ...
        6, 1, 1; ...
        6, 1, 1];

% 先创建 figure（可指定位置/大小，也可省略）
fig = figure('Units', 'pixels', 'Position', [100, 100, 500, 400]);

% 绘制堆叠柱状图
b = bar(data, 0.6, 'stacked');

% 获取当前坐标轴（即刚画图的 axes）
ax = gca;

% 设置颜色和线宽
for i = 1:3
    b(i).FaceColor = color(i,:);
    b(i).LineWidth = 1;
end

% 坐标轴设置
ax.YLim = [0, 30];
ax.XTickLabels = {'CT', 'WT', 'RWD', 'RA'};
ax.YTick = 0:10:30;
ax.YTickLabels = string(ax.YTick);
box off;

% 标签和标题
ylabel('Index');
title({'My'; 'Picture'}, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 2);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 2);

% 图例
legend('Ana', 'Oth', 'Met', 'Box', 'off');

% 保存为高分辨率 PNG（推荐使用 print 而非 saveas）
print(fig, '-dpng', '-r300', 'Figure5.png');

% 可选：关闭 figure（如果不需要显示）
% close(fig);

%% 堆叠柱状图（四类，百分比数据）
clc; clear; close all;
color = [0.2471,0.3059,0.5608; ...
         0.4667,0.4706,0.6824; ...
         0.6863,0.7373,0.8745; ...
         0.8588,0.9020,0.9569];

data = [19.3,31.6,34.4,14.6; ...
        43.4,32.7,16.4,7.5; ...
        31.5,30.2,25,13.3; ...
        37.1,26.2,28,8.3];

fig = figure;
b = bar(data, 0.6, 'stacked');
ax = gca;
for i = 1:4
    b(i).FaceColor = color(i,:);
    b(i).EdgeColor = color(i,:);
    b(i).LineWidth = 1;
end
ax.YLim = [0, 100];
ax.XTickLabels = {'White', 'Black', 'Hispanic', 'American'};
ax.YTick = 0:20:100;
ax.YTickLabels = string(ax.YTick);
box off;
ylabel('Percentage (%)');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 2);
title('Population Distribution', 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 2);
legend('Group A','Group B','Group C','Group D', 'Box', 'off');

% Save
saveas(fig, 'Figure6.png');
close(fig);

%% 三维柱状图
clc; clear; close all;
color = [0.5020,0.6706,0.7373; ...
         0.9961,0.8275,0.4824; ...
         0.7373,0.4902,0.7098];

data = [19.3,31.6,34.4,14.6; ...
        43.4,32.7,16.4,7.5; ...
        31.5,30.2,25,13.3]';

fig = figure;
h = bar3(data, 0.7);
for i = 1:3
    h(1,i).FaceColor = color(i,:);
end
xticklabels({'Auto','B','C','D'});
yticklabels({'Exp1','Exp2','Exp3'});
zlabel('Value');
title('3D Bar Chart');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.5);
box off;

% Save
saveas(fig, 'Figure7.png');
close(fig);

%% 8. 带统计显著性的柱状图（星号标注）
clc; clear; close all;
% 创建颜色方案
color_scheme = [0.2, 0.4, 0.6;    % 深蓝色
                0.6, 0.2, 0.2;    % 深红色
                0.3, 0.6, 0.3;    % 深绿色
                0.7, 0.5, 0.1];   % 金色

% 示例数据
group_data = [15.2, 22.4, 18.6;    % 组1: 控制, 处理1, 处理2
              23.8, 31.2, 26.7;    % 组2
              18.9, 27.3, 22.1;    % 组3
              12.5, 19.8, 15.9];   % 组4

% 标准差
error_data = [1.2, 1.8, 1.5;
              2.1, 1.5, 1.8;
              1.5, 1.2, 1.3;
              1.0, 1.6, 1.2];

fig = figure('Units', 'pixels', 'Position', [100, 100, 600, 400]);
b = bar(group_data);
hold on;

% 获取柱状图位置用于误差线
num_groups = size(group_data, 1);
num_bars = size(group_data, 2);
group_width = 0.8;
bar_width = group_width/num_bars;
x_positions = zeros(num_groups, num_bars);

for k = 1:num_bars
    x_positions(:, k) = b(k).XEndPoints;
    % 添加误差线
    errorbar(x_positions(:, k), group_data(:, k), error_data(:, k), ...
             'k.', 'LineWidth', 1.5, 'CapSize', 8);
    % 设置颜色
    b(k).FaceColor = color_scheme(k, :);
    b(k).EdgeColor = 'k';
    b(k).LineWidth = 1.5;
end

% 添加显著性标记（示例）
% ** p < 0.01, * p < 0.05
sig_y_pos = max(group_data(:)) + max(error_data(:)) + 2;

% 组1内的比较（控制 vs 处理1）
plot([x_positions(1,1), x_positions(1,2)], [sig_y_pos, sig_y_pos], 'k-', 'LineWidth', 1.2);
text(mean([x_positions(1,1), x_positions(1,2)]), sig_y_pos+0.5, '*', ...
     'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

% 组3内的比较（控制 vs 处理2）
plot([x_positions(3,1), x_positions(3,3)], [sig_y_pos-3, sig_y_pos-3], 'k-', 'LineWidth', 1.2);
text(mean([x_positions(3,1), x_positions(3,3)]), sig_y_pos-2.5, '**', ...
     'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

% 坐标轴设置
ylim([0, max(group_data(:)) + max(error_data(:)) + 5]);
xlabel('Experimental Groups', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Response Amplitude (mV)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', {'Group A', 'Group B', 'Group C', 'Group D'}, ...
         'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.5);
title('Treatment Effects with Statistical Significance', ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on; box on;

% 图例
legend({'Control', 'Treatment 1', 'Treatment 2'}, ...
       'Location', 'northwest', 'Box', 'off');

% 保存图像
print(fig, '-dpng', '-r300', 'Figure8.png');
% close(fig);

%% 9. 分组柱状图带折线（双Y轴）
clc; clear; close all;
% 定义颜色
bar_colors = [0.4, 0.6, 0.8;    % 浅蓝色
              0.8, 0.5, 0.4;    % 浅橙色
              0.5, 0.7, 0.5];   % 浅绿色

line_color = [0.8, 0.2, 0.2];   % 红色折线

% 数据
bar_data = [45, 38, 52;    % 组1
            62, 55, 48;    % 组2
            38, 42, 35;    % 组3
            51, 58, 46];   % 组4

line_data = [75, 82, 68, 79];  % 折线数据（百分比）

fig = figure('Units', 'pixels', 'Position', [100, 100, 650, 400]);

% 绘制分组柱状图（左Y轴）
yyaxis left;
b = bar(bar_data, 0.8);
hold on;

% 设置柱状图颜色
for k = 1:3
    b(k).FaceColor = bar_colors(k, :);
    b(k).EdgeColor = 'k';
    b(k).LineWidth = 1.2;
end

ylabel('Absolute Value (units)', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0, max(bar_data(:))*1.2]);

% 绘制折线（右Y轴）
yyaxis right;
x_pos = mean([b(1).XEndPoints; b(2).XEndPoints; b(3).XEndPoints]);
plot(x_pos, line_data, 'o-', 'Color', line_color, ...
     'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', line_color);

ylabel('Relative Percentage (%)', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0, 100]);

% 统一设置
set(gca, 'XTick', 1:4, ...
         'XTickLabel', {'Condition 1', 'Condition 2', 'Condition 3', 'Condition 4'}, ...
         'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.5, ...
         'Box', 'off');

% 标题
title('Dual-Axis Bar Chart with Overlaid Line Plot', ...
      'FontSize', 13, 'FontWeight', 'bold');

% 图例
legend([b(1), b(2), b(3)], {'Treatment A', 'Treatment B', 'Treatment C'}, ...
       'Location', 'northwest', 'Box', 'off');

% 添加网格
grid on;

% 保存图像
print(fig, '-dpng', '-r300', 'Figure9.png');
% close(fig);



%% 10. 堆叠分组柱状图（复杂数据结构）
clc; clear; close all;
% 定义颜色方案
stack_colors = [0.7, 0.85, 0.95;   % 浅蓝
                0.95, 0.85, 0.7;   % 浅橙
                0.85, 0.95, 0.7;   % 浅绿
                0.95, 0.7, 0.85];  % 浅紫

% 复杂堆叠数据（4组 × 3类别 × 4子类别）
complex_data = zeros(4, 3, 4);
% 组1
complex_data(1, :, :) = [15, 8, 5, 3;    % 类别1
                         12, 6, 4, 2;    % 类别2
                         10, 5, 3, 2];   % 类别3
% 组2
complex_data(2, :, :) = [20, 12, 8, 5;
                         18, 10, 7, 4;
                         15, 9, 6, 3];
% 组3
complex_data(3, :, :) = [12, 7, 4, 2;
                         10, 6, 3, 2;
                         8, 4, 3, 1];
% 组4
complex_data(4, :, :) = [22, 15, 10, 7;
                         20, 12, 8, 5;
                         18, 10, 7, 4];

fig = figure('Units', 'pixels', 'Position', [100, 100, 800, 450]);

% 计算每个组的x位置
group_spacing = 1;
bar_width = 0.2;
group_centers = 1:group_spacing:4*group_spacing;

% 为每个类别绘制堆叠柱状图
hold on;
for cat = 1:3
    % 计算当前类别柱子的x位置
    x_pos = group_centers + (cat-2)*bar_width;
    
    % 提取当前类别的数据
    cat_data = squeeze(complex_data(:, cat, :));
    
    % 绘制堆叠柱状图
    bottom = zeros(4, 1);
    for stack = 1:4
        bar_data = cat_data(:, stack);
        b = bar(x_pos, bar_data, bar_width, 'stacked');
        b.FaceColor = 'flat';
        b.CData = repmat(stack_colors(stack, :), 4, 1);
        b.EdgeColor = 'k';
        b.LineWidth = 1;
        
        % 添加数据标签（只显示最上层）
        if stack == 4
            for i = 1:4
                total_height = sum(cat_data(i, :));
                text(x_pos(i), total_height+0.5, sprintf('%d', total_height), ...
                     'HorizontalAlignment', 'center', 'FontSize', 8);
            end
        end
        
        bottom = bottom + bar_data;
    end
end

% 坐标轴设置
xlim([0.5, 4.5]);
ylim([0, max(sum(complex_data, 3), [], 'all') * 1.15]);
set(gca, 'XTick', group_centers, ...
         'XTickLabel', {'Group 1', 'Group 2', 'Group 3', 'Group 4'}, ...
         'FontName', 'Arial', 'FontSize', 11, 'LineWidth', 1.5);
ylabel('Cumulative Score', 'FontSize', 12, 'FontWeight', 'bold');
title('Complex Stacked Grouped Bar Chart', 'FontSize', 13, 'FontWeight', 'bold');
box on; grid on;

% 创建自定义图例
legend_labels = {'Component A', 'Component B', 'Component C', 'Component D'};
h = zeros(4, 1);
for i = 1:4
    h(i) = patch(NaN, NaN, stack_colors(i, :));
end
legend(h, legend_labels, 'Location', 'northeast', 'Box', 'off');

% 保存图像
print(fig, '-dpng', '-r300', 'Figure10.png');
% close(fig);



