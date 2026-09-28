function [line_xys, dot_xys] = screen_xys(info,center_x,center_y,transform_to_pix_rate)

% Step 1. 计算坐标
x3 = info.body_ori .* info.body_length / 2;
y3 = 0;

x4 = - info.body_ori .* info.body_length / 2;
y4 = y3;

x2 = x3 + info.body_ori .* info.neck_length .* cos(info.neck_angle);
y2 = y3 - info.neck_length .* sin(info.neck_angle);

x1 = x2 + info.body_ori .* info.head_length .* cos(info.head_angle);
y1 = y2 - info.head_length .* sin(info.head_angle);

x5 = x4 - info.body_ori .* info.tail_length .* cos(info.tail_angle);
y5 = y4 - info.tail_length .* sin(info.tail_angle);

x6 = x3 + info.body_ori .* info.leg_length .* cos(info.leg_angle);
y6 = y3 - info.leg_length .* sin(info.leg_angle);
x7 = x3 - info.body_ori .* info.leg_length .* cos(info.leg_angle);
y7 = y3 - info.leg_length .* sin(info.leg_angle);

x8 = x4 + info.body_ori .* info.leg_length .* cos(info.leg_angle);
y8 = y4 - info.leg_length .* sin(info.leg_angle);
x9 = x4 - info.body_ori .* info.leg_length .* cos(info.leg_angle);
y9 = y4 - info.leg_length .* sin(info.leg_angle);

% Step 2. 生成所有需要画的线段和点
xstart = [x3,x3,x2,x3,x3,x4,x4,x4];
xend   = [x4,x2,x1,x6,x7,x8,x9,x5];
ystart = [y3,y3,y2,y3,y3,y4,y4,y4];
yend   = [y4,y2,y1,y6,y7,y8,y9,y5];

x_dots = [x1,x2,x3,x4,x5,x6,x7,x8,x9];
y_dots = [y1,y2,y3,y4,y5,y6,y7,y8,y9];


% 坐标变换: 原点 -> 屏幕中心
% 注意：PTB的y轴向下是正，所以我们要反转y方向
toScreenX = @(x) x * transform_to_pix_rate + center_x;
toScreenY = @(y) -y * transform_to_pix_rate + center_y;

line_xys = zeros(2, 2*length(xstart));
for i = 1:length(xstart)
    xs = toScreenX(xstart(i));
    ys = toScreenY(ystart(i));
    xe = toScreenX(xend(i));
    ye = toScreenY(yend(i));
    line_xys(:,2*i-1) = [xs; ys];
    line_xys(:,2*i) = [xe; ye];
end

dot_xys = zeros(2, length(x_dots));
for i = 1:length(x_dots)
    x = toScreenX(x_dots(i));
    y = toScreenY(y_dots(i));
    dot_xys(:,i) = [x; y];
end
