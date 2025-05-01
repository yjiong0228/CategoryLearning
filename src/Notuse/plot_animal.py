
class Fig1A:

    @staticmethod
    def _compute_keypoints(info):
        """
        计算并返回 9 个关节点的坐标，按顺序：
          [head_tip, head_base, body_right, body_left, tail_tip,
           leg1, leg2, leg3, leg4]
        返回 shape = (9,2) 的 numpy array
        """
        ori = info['body_ori']
        Lb = info['body_length']
        # 身体两端
        x3, y3 =  ori * Lb/2, 0
        x4, y4 = -ori * Lb/2, 0
        # 颈 & 头
        x2 = x3 + ori * info['neck_length'] * np.cos(info['neck_angle'])
        y2 = y3 -       info['neck_length'] * np.sin(info['neck_angle'])
        x1 = x2 + ori * info['head_length'] * np.cos(info['head_angle'])
        y1 = y2 -       info['head_length'] * np.sin(info['head_angle'])
        # 尾
        x5 = x4 - ori * info['tail_length'] * np.cos(info['tail_angle'])
        y5 = y4 -       info['tail_length'] * np.sin(info['tail_angle'])
        # 四条腿
        x6 = x3 + ori * info['leg_length'] * np.cos(info['leg_angle'])
        y6 = y3 -       info['leg_length'] * np.sin(info['leg_angle'])
        x7 = x3 - ori * info['leg_length'] * np.cos(info['leg_angle'])
        y7 = y3 -       info['leg_length'] * np.sin(info['leg_angle'])
        x8 = x4 + ori * info['leg_length'] * np.cos(info['leg_angle'])
        y8 = y4 -       info['leg_length'] * np.sin(info['leg_angle'])
        x9 = x4 - ori * info['leg_length'] * np.cos(info['leg_angle'])
        y9 = y4 -       info['leg_length'] * np.sin(info['leg_angle'])
        return np.array([
            [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5],
            [x6, y6], [x7, y7], [x8, y8], [x9, y9]
        ])
        
    def draw_animal(self, info, ax, *, lw=3.5, point_size=50, alpha=1.0):
        """
        在现有 ax 上画一只动物轮廓 & 关节点。
        """
        pts = self._compute_keypoints(info)
        # 8 条骨骼的节点索引对 (与原代码对应)
        bones = [
            (2,3),  # body
            (2,1),  # neck
            (1,0),  # head
            (3,4),  # tail
            (2,5),  # leg1
            (2,6),  # leg2
            (3,7),  # leg3
            (3,8),  # leg4
        ]
        for i,j in bones:
            ax.plot(
                [pts[i,0], pts[j,0]],
                [pts[i,1], pts[j,1]],
                color='k', linewidth=lw, alpha=alpha
            )
        ax.scatter(
            pts[:,0], pts[:,1],
            s=point_size, c='k', alpha=alpha, zorder=5
        )

    def draw_animal_with_gradient(
        self,
        base_info: dict,
        L_short: float,
        L_long:  float,
        save_path=None,
        segments: int    = 100,
        alpha_start: float = 1.0,
        alpha_end:   float = 0.1,
        figsize       = (4,3),
        dpi           = 300):

        # ------ 0. 生成短 / 长动物关键点 ------
        info_s = base_info.copy()
        info_l = base_info.copy()
        for k in ('neck_length','head_length','leg_length','tail_length'):
            info_s[k] = L_short
            info_l[k] = L_long
        pts_s = self._compute_keypoints(info_s)
        pts_l = self._compute_keypoints(info_l)

        # 对应骨骼 (i, j)
        bones = [
            (2,3),  # body
            (2,1),  # neck      ← 绿色圈出，要反转 alpha
            (1,0),  # head tip  ← 红色圈出，要完全省略渐变
            (3,4),  # tail
            (2,5), (2,6), (3,7), (3,8)  # four legs
        ]

        # ------ 1. 画底层“短”动物 ------
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_aspect('equal'); ax.axis('off')
        self.draw_animal(info_s, ax, lw=3.5, point_size=20, alpha=alpha_start)

        # ------ 2. 依骨骼绘制渐变延伸 ------
        for _, j in bones:

            # (1) 跳过红框里的 head-tip 渐变
            if j == 0:          # j=0 → point[0] = head_tip
                continue

            p0, p1 = pts_s[j], pts_l[j]      # 短 / 长动物同一端点
            xs = np.linspace(p0[0], p1[0], segments + 1)
            ys = np.linspace(p0[1], p1[1], segments + 1)
            segs = [ [(xs[k], ys[k]), (xs[k+1], ys[k+1])]
                     for k in range(segments) ]

            # (2) 透明度规则
            alphas = np.linspace(alpha_start, alpha_end, segments)

            colors = [(0, 0, 0, a) for a in alphas]
            ax.add_collection(LineCollection(segs,
                                             colors=colors,
                                             linewidths=3.5))

        # ------ 3. 画“长”动物其余骨骼轮廓 ------
        for i, j in bones:
            ax.plot([pts_l[i,0], pts_l[j,0]],
                    [pts_l[i,1], pts_l[j,1]],
                    color='k', linewidth=3.5, alpha=alpha_end)
        ax.scatter(pts_l[:,0], pts_l[:,1],
                   s=20, c='k', alpha=alpha_end, zorder=5)

        # 保存
        plt.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved to {save_path}")
