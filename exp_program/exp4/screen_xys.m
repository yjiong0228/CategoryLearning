function [feature_rects, dots_xys] = screen_xys(info,center_x,center_y,img_ratio,transform_to_pix_rate)

% Step 1. 转换长度
length_fields = {'body_length', 'green_length', 'yellow_length', 'pink_length', 'blue_length'};
for i = 1:length(length_fields)
    info.(length_fields{i}) = info.(length_fields{i}) * transform_to_pix_rate;
end

img_width = info.body_length * img_ratio;

% Step 2. 计算坐标
x11 = center_x + cos(info.angle) .* (info.body_length / 2 + info.green_length / 2);
y11 = center_y - sin(info.angle) .* (info.body_length / 2 + info.green_length / 2); %green右上

x21 = center_x - cos(info.angle) .* (info.body_length / 2 + info.yellow_length / 2);
y21 = center_y - sin(info.angle) .* (info.body_length / 2 + info.yellow_length / 2); %yellow左上

x31 = center_x - cos(info.angle) .* (info.body_length / 2 + info.pink_length / 2);
y31 = center_y + sin(info.angle) .* (info.body_length / 2 + info.pink_length / 2); %pink左下

x41 = center_x + cos(info.angle) .* (info.body_length / 2 + info.blue_length / 2);
y41 = center_y + sin(info.angle) .* (info.body_length / 2 + info.blue_length / 2); %blue右下

f1_rect_xmin = x11 - img_width / 2;
f1_rect_ymin = y11 - info.green_length / 2;

f1_rect_xmax = x11 + img_width / 2;
f1_rect_ymax = y11 + info.green_length / 2;

f2_rect_xmin = x21 - img_width / 2;
f2_rect_ymin = y21 - info.yellow_length / 2;

f2_rect_xmax = x21 + img_width / 2;
f2_rect_ymax = y21 + info.yellow_length / 2;

f3_rect_xmin = x31 - img_width / 2;
f3_rect_ymin = y31 - info.pink_length / 2;

f3_rect_xmax = x31 + img_width / 2;
f3_rect_ymax = y31 + info.pink_length / 2;

f4_rect_xmin = x41 - img_width / 2;
f4_rect_ymin = y41 - info.blue_length / 2;

f4_rect_xmax = x41 + img_width / 2;
f4_rect_ymax = y41 + info.blue_length / 2;


% Step 3. 生成图片框顶点
feature_rects = [
    f1_rect_xmin, f1_rect_ymin, f1_rect_xmax, f1_rect_ymax;  % f1行
    f2_rect_xmin, f2_rect_ymin, f2_rect_xmax, f2_rect_ymax;  % f2行
    f3_rect_xmin, f3_rect_ymin, f3_rect_xmax, f3_rect_ymax;  % f3行
    f4_rect_xmin, f4_rect_ymin, f4_rect_xmax, f4_rect_ymax   % f4行
];


% dot
x1_dot = center_x + cos(info.angle) .* (info.body_length / 2 + info.green_length);
y1_dot = center_y - sin(info.angle) .* (info.body_length / 2 + info.green_length);

x2_dot = center_x - cos(info.angle) .* (info.body_length / 2 + info.yellow_length);
y2_dot = center_y - sin(info.angle) .* (info.body_length / 2 + info.yellow_length);

x3_dot = center_x - cos(info.angle) .* (info.body_length / 2 + info.pink_length);
y3_dot = center_y + sin(info.angle) .* (info.body_length / 2 + info.pink_length);

x4_dot = center_x + cos(info.angle) .* (info.body_length / 2 + info.blue_length);
y4_dot = center_y + sin(info.angle) .* (info.body_length / 2 + info.blue_length);


dots_xys = [x1_dot,x2_dot,x3_dot,x4_dot;y1_dot,y2_dot,y3_dot,y4_dot];
