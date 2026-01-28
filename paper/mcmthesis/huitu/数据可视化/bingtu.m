%% 图1：使用color_cell1.mat中的颜色设置

% 加载自定义颜色设置
load('color_cell1.mat')
mycolor1=color_cell1{1,1};

% 数据定义
data=[1,2,3,6,7];
figure(1)
pie3(data,ones(1,length(data))) % 绘制三维饼图
colormap(mycolor1) % 应用颜色映射
set(gca,'looseInset',[0 0 0 0]); % 去除图片的白边
saveas(gcf, 'Figure1.png') % 保存当前图形

%% 图2：带有标签的饼状图

figure(2)
labels={'species1 48%', 'species2 35%', 'species3 17%'}; % 定义标签（英文）
pie3([48,35,17], labels) % 使用数据和标签绘制三维饼图
color1_set=[0.0039, 0.4471, 0.7373; 
            0.4706, 0.6706, 0.1882; 
            0.8471, 0.3255, 0.0980]; % 自定义颜色集
colormap(color1_set) % 应用颜色映射
set(gca,'looseInset',[0 0 0 0]); % 去除图片的白边
saveas(gcf, 'Figure2.png') % 保存当前图形

%% 图3：二维饼状图（圆形图）

figure(3)
data_example=[10, 20, 30, 40]; % 示例数据
labels_example={'Group A', 'Group B', 'Group C', 'Group D'}; % 示例标签
pie(data_example, labels_example) % 绘制二维饼图
colormap(mycolor1) % 使用第一个图的颜色映射
saveas(gcf, 'Figure3.png') % 保存当前图形


%% 图4：部分扇区分离的饼状图

figure(4)
mycolor2=color_cell1{1,2};
data_separated=[10, 20, 30, 40]; % 示例数据
explode = [0 1 0 0]; % 分离第二个扇区
pie(data_separated, explode, labels_example) % 绘制并分离指定扇区
colormap(mycolor2) % 应用颜色映射
saveas(gcf, 'Figure4.png') % 保存当前图形

