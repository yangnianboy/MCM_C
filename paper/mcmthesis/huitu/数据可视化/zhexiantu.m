% MATLAB 官方整理
%https://ww2.mathworks.cn/products/matlab/plot-gallery.html?s_tid=srchtitle_gallery_1
% 绘图案例
%% 折线图
%% 图框的基本设置
clc;clear;close all;
load('color_list.mat')

load('折线图测试数据.mat')

figure('Position',[100,100,600,400])  %整个框大小设置

x1=[1:100];
y1=2*x1+10*rand(1,length(x1));
y2=0.01*x1.^2+0.5*x1+10*rand(1,length(x1));

color_index=[1,3,4,7,8,9];  %颜色选取

fu={'-*','-^','-o','-d','-p'};   %符号

% %单根折线图  
% plot(x1,y1,'--','LineWidth',2,'Color',color_list(color_index(1),:))
% hold on      %保持在一个图窗内
% plot(x1,y2,'--','LineWidth',2,'Color',color_list(color_index(2),:))
% hold on 

color_set=[0.1451    0.6118    0.1451;1.0000    0.4980    0.0549];
% 单根折线图  
plot(x1,y1,'--','LineWidth',2,'Color',color_set(1,:))
hold on      %保持在一个图窗内
plot(x1,y2,'-','LineWidth',1.5,'Color',color_set(2,:))
hold on 

xlabel('time')
ylabel('value')
title('title')

xticks([10:20:100]);
% xticks([10,30,50,70,90]);

xticklabels({'1-10','1-20','2-10','2-20','3-10'})

%图框全框或者半框的设置
box off
%box on

% x轴和y轴的范围设置

% xlim([0,100]) 
% ylim([0,100])

%直线的设置
xline(30,'--','LineWidth',1.2)
yline(30)

%网格的设置
% grid("on")
% set(gca,'XGrid','on',GridLineStyle',':','GridColor','k','GridAlpha',1);

% set(gca,'xgrid','on');
% set(gca,'ygrid','on');

legend('sybol1','sybol2')
legend('Box','off')

set(gca,'FontName','Times New Roman',"FontSize",12,"LineWidth",1.2)

% set(gca,'looseInset',[0 0 0 0]); %去除图片的白边


%%
% 导入颜色矩阵
clc;clear;close all;
load('color_list.mat')
load('折线图测试数据.mat')
%
color_index=[1,3,4,7,8,9];  %颜色选取
fu={'-*','-^','-o','-d','-p'};   %符号
%单根折线图  
figure
plot(data_test4(1,:),fu{1,5},'LineWidth',2,'Color',color_list(color_index(1),:))
hold on 
plot(data_test4(2,:),fu{1,4},'LineWidth',2,'Color',color_list(color_index(2),:))
set(gca,"FontSize",11,"LineWidth",1.2)
xlabel('时间')
ylabel('数值')
title('标题自定义')
box off
legend('对象一','对象二')

%%
%多根折线图

figure('Position',[100,200,650,350])
str=[];

color_index=[3,6,10,50,60];
color_set=color_list(color_index,:);

for i=1:5
   plot(data_test4(i,:),fu{1,i},'LineWidth',2,'Color',color_set(i,:))
   hold on  
   str{1,i}=['object',num2str(i)];
end
set(gca,'FontName','Times New Roman',"FontSize",12,"LineWidth",1.2)

% box off

xlabel('x')
ylabel('y')
title('title')
% str={'A','B','C','D','E'};
lgb=legend(str);

lgb.NumColumns=3;
legend('Box','on','EdgeColor',[0.8,0.8,0.8])
% legend('Box','on','Color',[0.95,0.95,0.95])

%%  带误差棒的折线图
clc;clear;close all;
load('color_list.mat')
color=[0.611764705882353,0.207843137254902,0.560784313725490;0.301960784313725,...
    0.164705882352941,0.458823529411765;0.556862745098039,0.549019607843137,0.549019607843137];
% index=[1,30,40];
% color=color_list(index,:);
% data_test=[1,2,3;1.1,1.9,2.9;1.2,2.1,3.1];
% dmean=mean(data_test);
% dstd=std(data_test);
% 
% x=[0.1,2.1,3.1]; %x轴数据
% 
% y1=dmean;  %y轴数据
% 
% low1=dstd;  %数据上限
% high1=dstd;  %数据下限
% 
% errorbar(x,y1,low1,high1,'-s','Color',color(1,:),'MarkerSize',10, 'MarkerEdgeColor',color(1,:),'MarkerFaceColor',[1,1,1],...
%     'LineWidth', 2,'CapSize',10)
% hold on
x=[0.1,2.1,3.1,4.1,5.1]; %x轴数据

y1=[1.1,0.7,0.75,0.9,0.95];  %y轴数据

low1=0.2*rand(1,length(y1));  %数据上限
high1=low1;  %数据下限

y2=[1.3,0.32,0.35,0.3,0.25];
low2=0.1*rand(1,length(y2));
high2=low2;

y3=[1.5,1.2,1.15,1.6,1.2];
low3=0.3*rand(1,length(y3));
high3=low3;

% 画图
%调整整个图的范围
figure('Units', 'pixels', ...
    'Position', [100 100 600 375]);
%'Color'整个连接线的颜色，'MarkerSize',标记的大小，'MarkerEdgeColor'，标记边缘颜色，'MarkerFaceColor',标记填充颜色
%'LineWidth', 线宽,'CapSize',误差图标帽的大小

errorbar(x,y1,low1,high1,'-s','Color',color(1,:),'MarkerSize',10, 'MarkerEdgeColor',color(1,:),'MarkerFaceColor',[1,1,1],...
    'LineWidth', 2,'CapSize',10)
hold on

errorbar(x,y2,low2,high2,'-o','Color',color(2,:),'MarkerSize',8, 'MarkerEdgeColor',color(2,:),'MarkerFaceColor',color(2,:),...
    'LineWidth', 2,'CapSize',10)
hold on
errorbar(x,y3,low3,high3,'-^','Color',color(3,:),'MarkerSize',8, 'MarkerEdgeColor',color(3,:),'MarkerFaceColor',color(3,:),...
    'LineWidth', 2,'CapSize',10)
hold on
ylabel('Mechanical Threshold')
% 调坐标的范围
ax = gca;
%x轴范围
ax.XTick = [0.1,2.1,3.1,4.1,5.1];
%x轴标签
ax.XTickLabels ={'Baseline', '0', '4h','24h','48h'};
ax.YTick= [1,2];
%y轴范围
ax.YLim=[0,2];
% 图例
% 不同图例不同颜色标记，'LineWidth',字体粗细,'FontSize',字体大小 ,'Orientation' 图例位置
% legend('第一次实验','第二次','第三次')
legend(['\color[rgb]{',num2str(color(1,:)) ,'}','Control'],['\color[rgb]{',num2str(color(2,:)) ,'}',' CFA'] ,...
    ['\color[rgb]{',num2str(color(3,:)) ,'}','BY'],'LineWidth',2,'FontSize',14,'Location','best');
legend('boxoff') %图例框消失
%设置字体
% set(gca,"FontSize",14,"LineWidth",2)
set(gca,"FontName","Times New Roman","FontSize",14,"LineWidth",2)
title("MY picture","FontName","Times New Roman","FontSize",14,"LineWidth",2);
box off

%%  单个误差带的折线图

data_test=[1,2,3,4,5;1.5,1.8,2.3,3.2,4.5;0.2,2.2,3.1,3.3,4.6];
dmean=mean(data_test);
dstd=std(data_test);

x=[0.1,2.1,3.1,4.1,5.1]; %x轴数据

y1=dmean;  %y轴数据

low1=dmean-dstd;  %数据上限
high1=dmean+dstd;  %数据下限


figure('Position',[200,200,600,350])



color_set=[1.0000    0.6863    0.3490];

plot(x,y1,'-s','Color',color_set(1,:),'LineWidth',1.5)
hold on 
h1=fill([x,fliplr(x)],[low1,fliplr(high1)],'r');

hold on

h1.FaceColor = color_set(1,:);%定义区间的填充颜色   

h1.EdgeColor =[1,1,1];%边界颜色设置为白色

alpha (0.3)   %设置透明色

scatter(x,data_test,20,'o','filled','MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5])
hold on

set(gca,'FontName','Times New Roman',"FontSize",12,"LineWidth",1.1)

% box off
legend('mean','std','data')

xlabel('x')
ylabel('y')
title('title')

%%  多个带误差带的折线图
% clc;clear;close all;
color=[0.611764705882353,0.207843137254902,0.560784313725490;0.301960784313725,...
    0.164705882352941,0.458823529411765;0.556862745098039,0.549019607843137,0.549019607843137];
x=[0.1,2.1,3.1,4.1,5.1]; %x轴数据
y1=[1.1,0.7,0.75,0.9,0.95];  %y轴数据
low1=0.2*rand(1,length(y1));  %数据上限
high1=low1;  %数据下限
y2=[1.3,0.32,0.35,0.3,0.25];
low2=0.1*rand(1,length(y2));
high2=low2;
y3=[1.5,1.2,1.15,1.6,1.2];
low3=0.3*rand(1,length(y3));
high3=low3;
% 画图
%调整整个图的范围
figure('Units', 'pixels', ...
    'Position', [100 100 600 375]);

plot(x,y1,'-s','Color',color(1,:),'LineWidth',1.5)
hold on 
h1=fill([x,fliplr(x)],[y1-low1,fliplr(y1+high1)],'r');

hold on
h1.FaceColor = color(1,:);%定义区间的填充颜色      
h1.EdgeColor =[1,1,1];%边界颜色设置为白色
alpha (0.1)   %设置透明色


plot(x,y2,'-p','Color',color(2,:),'LineWidth',1.5)
hold on 
h1=fill([x,fliplr(x)],[y2-low2,fliplr(y2+high2)],'r');
hold on
h1.FaceColor = color(2,:);%定义区间的填充颜色      
h1.EdgeColor =[1,1,1];%边界颜色设置为白色
alpha (0.2)   %设置透明色

plot(x,y3,'-d','Color',color(3,:),'LineWidth',1.5)
hold on 
h1=fill([x,fliplr(x)],[y3-low1,fliplr(y3+high3)],'r');
hold on
h1.FaceColor = color(3,:);%定义区间的填充颜色      
h1.EdgeColor =[1,1,1];%边界颜色设置为白色
alpha (0.2)   %设置透明色
ylabel('Mechanical Threshold')
% 调坐标的范围
ax = gca;
%x轴范围
ax.XTick = [0.1,2.1,3.1,4.1,5.1];
%x轴标签
ax.XTickLabels ={'Baseline', '0', '4h','24h','48h'};
ax.YTick= [1,2];
%y轴范围
ax.YLim=[0,2];
% 图例
% 不同图例不同颜色标记，'LineWidth',字体粗细,'FontSize',字体大小 ,'Orientation' 图例位置
% legend(['\color[rgb]{',num2str(color(1,:)) ,'}','Control'],['\color[rgb]{',num2str(color(2,:)) ,'}',' CFA'] ,...
%     ['\color[rgb]{',num2str(color(3,:)) ,'}','BY'],'LineWidth',2,'FontSize',14,'Location','best');
% legend('boxoff') %图例框消失
%设置字体
set(gca,"FontName","Times New Roman","FontSize",14,"LineWidth",2)
title("MY picture","FontName","Times New Roman","FontSize",14,"LineWidth",2);
box off


%% ================= 补充1：双Y轴折线图 =================
% 适用于比较两个量纲不同但存在关联的数据

figure('Position', [100, 100, 600, 400]);

% 基于已有数据创建示例
x_double = 1:50;
y_left = sin(x_double/5) + 0.1*randn(1,50);
y_right = 50*cos(x_double/10) + 5*randn(1,50);

% 左侧Y轴
yyaxis left;
plot(x_double, y_left, '-', 'LineWidth', 2, 'Color', color_list(1,:));
ylabel('Left Y-axis (V)', 'FontSize', 12);
ylim([-1.5, 1.5]);

% 右侧Y轴
yyaxis right;
plot(x_double, y_right, '--', 'LineWidth', 2, 'Color', color_list(30,:));
ylabel('Right Y-axis (°C)', 'FontSize', 12);
ylim([-80, 80]);

% 公共设置
xlabel('Time (s)', 'FontSize', 12);
title('Dual Y-axis Line Plot Example', 'FontSize', 14);
legend('Signal Amplitude', 'Temperature', 'Location', 'best');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);
box off;

%% ================= 补充2：阶梯图（离散数据展示） =================
% 适用于强调数据突变点或离散采样

figure('Position', [150, 150, 600, 400]);

% 创建示例数据
x_stem = 1:2:20;
y_stem = exp(-0.2*x_stem) .* sin(2*x_stem/5) + 0.1*randn(size(x_stem));

% 绘制阶梯图
stem(x_stem, y_stem, 'LineWidth', 1.5, 'Marker', 's', 'MarkerSize', 8, ...
    'MarkerFaceColor', color_list(15,:), ...
    'Color', color_list(15,:)*0.7, ...
    'MarkerEdgeColor', 'k');

hold on;

% 叠加趋势线
x_fine = 1:0.1:20;
y_fine = exp(-0.2*x_fine) .* sin(2*x_fine/5);
plot(x_fine, y_fine, '--', 'LineWidth', 1.5, 'Color', color_list(45,:));

% 图形设置
xlabel('Discrete Sampling Points', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Discrete Signal Display', 'FontSize', 14);
legend('Sampled Data', 'Fitted Curve', 'Location', 'best');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);
box off;

%% ================= 补充3：堆叠面积图 =================
% 用于展示多个数据序列的累积贡献

figure('Position', [200, 200, 650, 400]);

% 创建示例数据
x_stack = 1:10;
stack_data = [
    cumsum(rand(1,10)) * 10;
    cumsum(rand(1,10)) * 8;
    cumsum(rand(1,10)) * 6;
    cumsum(rand(1,10)) * 4;
];

% 颜色设置
stack_colors = color_list([1, 20, 40, 60], :);
categories = {'Category A', 'Category B', 'Category C', 'Category D'};

% 绘制堆叠面积图
area_handles = area(x_stack, stack_data');
for i = 1:length(area_handles)
    area_handles(i).FaceColor = stack_colors(i, :);
    area_handles(i).FaceAlpha = 0.7;
    area_handles(i).EdgeColor = 'none';
end

% 添加数据点标记
hold on;
for i = 1:size(stack_data, 1)
    plot(x_stack, stack_data(i,:), 'o-', 'Color', stack_colors(i,:)*0.7, ...
        'LineWidth', 1.2, 'MarkerFaceColor', stack_colors(i,:), 'MarkerSize', 5);
end

% 图形设置
xlabel('Time Phase', 'FontSize', 12);
ylabel('Cumulative Contribution', 'FontSize', 12);
title('Stacked Area Plot', 'FontSize', 14);
legend(categories, 'Location', 'best', 'Box', 'off');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);
grid on;
box off;

%% ================= 补充4：交互式折线图 =================
% 带数据光标功能，点击显示数据点信息

figure('Position', [250, 250, 600, 400]);

% 使用已有数据
x_interactive = 1:50;
y_interactive = sin(x_interactive/5) + 0.1*randn(1,50);

% 绘制折线图和散点
h_plot = plot(x_interactive, y_interactive, '-', 'LineWidth', 1.5, 'Color', color_list(25,:));
hold on;
scatter_idx = 1:5:length(x_interactive);
h_scatter = scatter(x_interactive(scatter_idx), y_interactive(scatter_idx), 40, 'filled', ...
    'MarkerFaceColor', color_list(50,:), 'MarkerEdgeColor', 'k');

% 基本设置
xlabel('Data Point Index', 'FontSize', 12);
ylabel('Signal Value', 'FontSize', 12);
title('Interactive Line Plot (Click Data Points)', 'FontSize', 14);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);
box off;

% 启用数据光标
dcm = datacursormode(gcf);
set(dcm, 'Enable', 'on', 'SnapToDataVertex', 'on', ...
    'DisplayStyle', 'datatip');

%% ================= 补充5：子图组合展示 =================
% 在一个图窗中展示多种图形

figure('Position', [100, 100, 900, 600], 'Name', 'Multi-plot Combination');

% 准备数据
x_combo = 1:50;
y1_combo = sin(x_combo/8);
y2_combo = cos(x_combo/6);

% 子图1：基本折线图
subplot(2, 3, 1);
plot(x_combo, y1_combo, '-', 'LineWidth', 1.5, 'Color', color_list(1,:));
title('(a) Basic Line Plot', 'FontSize', 11);
xlabel('X'); ylabel('Y'); grid on; box off;

% 子图2：带误差棒
subplot(2, 3, 2);
x_err = 1:5:50;
y_mean = sin(x_err/8);
y_err = 0.2*ones(size(x_err));
errorbar(x_err, y_mean, y_err, 's-', 'LineWidth', 1.5, ...
    'Color', color_list(20,:), 'CapSize', 8);
title('(b) With Error Bars', 'FontSize', 11);
xlabel('X'); ylabel('Y'); grid on; box off;

% 子图3：阶梯图
subplot(2, 3, 3);
stem(x_err, y_mean, 'LineWidth', 1.5, ...
    'MarkerFaceColor', color_list(15,:), 'Color', color_list(15,:));
title('(c) Stem Plot', 'FontSize', 11);
xlabel('X'); ylabel('Y'); grid on; box off;

% 子图4：双Y轴
subplot(2, 3, 4);
yyaxis left;
plot(x_combo, y1_combo, '-', 'Color', color_list(1,:));
ylabel('Left Axis');
yyaxis right;
plot(x_combo, y2_combo, '--', 'Color', color_list(30,:));
ylabel('Right Axis');
title('(d) Dual Y-axis Plot', 'FontSize', 11);
xlabel('X'); box off; grid on;

% 子图5：散点+趋势线
subplot(2, 3, 5);
scatter(x_combo(1:20), y1_combo(1:20), 40, color_list(50,:), 'filled');
hold on;
p = polyfit(x_combo(1:20), y1_combo(1:20), 2);
y_fit = polyval(p, x_combo(1:20));
plot(x_combo(1:20), y_fit, 'r-', 'LineWidth', 1.5);
title('(e) Scatter + Trend Line', 'FontSize', 11);
xlabel('X'); ylabel('Y'); grid on; box off;

% 子图6：简单面积图
subplot(2, 3, 6);
area(x_combo(1:20), y1_combo(1:20), 'FaceAlpha', 0.6, 'FaceColor', color_list(10,:));
title('(f) Area Plot', 'FontSize', 11);
xlabel('X'); ylabel('Y'); grid on; box off;

% 整体标题
sgtitle('Multi-plot Combination Display', 'FontSize', 14, 'FontWeight', 'bold');

%% ================= 补充6：极坐标图 =================
% 适用于周期性或方向性数据

figure('Position', [280, 280, 500, 500]);

% 创建极坐标数据
theta = linspace(0, 2*pi, 100);
r1 = 1 + 0.3*cos(5*theta);
r2 = 0.8*abs(sin(3*theta));

% 切换为极坐标
polaraxes;
hold on;

% 绘制曲线
polarplot(theta, r1, 'LineWidth', 2.5, 'Color', color_list(1,:));
polarplot(theta, r2, 'LineWidth', 2.5, 'Color', color_list(35,:));

% 图形设置
title('Polar Plot Example', 'FontSize', 14);
legend('Series 1', 'Series 2', 'Location', 'best');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);




%% ================= 补充7：多子图时间序列对比 =================
% 多个相关时间序列的对比展示

figure('Position', [400, 400, 700, 400]);

% 创建相关时间序列
t = 1:100;
signals = zeros(4, 100);
for i = 1:4
    signals(i, :) = sin(t/(10+i)) + 0.3*randn(1,100) + i*0.2;
end

signal_names = {'Signal A', 'Signal B', 'Signal C', 'Signal D'};
colors = color_list([1, 25, 50, 75], :);

% 绘制4个子图
for i = 1:4
    subplot(2, 2, i);
    plot(t, signals(i, :), '-', 'LineWidth', 1.5, 'Color', colors(i, :));
    hold on;
    
    % 添加移动平均
    window_size = 10;
    moving_avg = movmean(signals(i, :), window_size);
    plot(t, moving_avg, '--', 'LineWidth', 2, 'Color', 'k');
    
    title(signal_names{i}, 'FontSize', 12);
    xlabel('Time Point');
    ylabel('Amplitude');
    grid on;
    box off;
    legend('Original Signal', 'Moving Average', 'Location', 'best');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, 'LineWidth', 1);
end

sgtitle('Multi-time Series Comparative Analysis', 'FontSize', 14, 'FontWeight', 'bold');
box off;
% ========== 保存为 PNG 文件 ==========
saveas(gcf, 'Figure14.png');  % 保存到当前工作目录
