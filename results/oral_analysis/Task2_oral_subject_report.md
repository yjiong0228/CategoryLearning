# Task2 口头汇报分析报告

## 生成说明

- `fidelity` 是文本语义与当前 trial 的 `feature1~feature4` 是否一致的自动评分，范围为 0 到 1。
- 有明确可解析断言时，按断言通过比例计算；没有新语义断言但已有旧版 region 编码时，用旧版 region 约束的通过比例兜底。
- `一样长/差不多/均匀` 按相对接近处理，默认容差为 0.10；严格的 `等于` 默认容差不超过 0.06。
- `躯干/身体` 在 fidelity 中按 0.5 处理；`3/4躯干` 按 0.375 处理；单独的 `一半` 按 0.5 处理。
- `两长两短/三个部位很长` 等未点名部位的计数抽象目前只做标记，不用 feature 自动反推部位。

## 总览

| 被试数 | trial数 | 非空文本 | fidelity可评分率 | 平均fidelity | 完全忠实率 |
| --- | --- | --- | --- | --- | --- |
| 96.000 | 62720.000 | 62056.000 | 0.970 | 0.897 | 0.770 |

## 被试摘要表

| iSub | n_trials | n_text | fidelity_parseable_rate | fidelity_mean | fidelity_full_rate | fidelity_low_rate | legacy_region_encoded_rate | legacy_region_unparsed_rate | dominant_style_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 320 | 320 | 1.000 | 0.885 | 0.841 | 0.078 | 1.000 | 0.000 | direct_absolute:320, comparison:23, equality:3, body_ref:2 |
| 102 | 448 | 448 | 0.993 | 0.950 | 0.882 | 0.018 | 0.993 | 0.109 | direct_absolute:413, comparison:274, ranking:26, superlative:24, equality:6 |
| 103 | 256 | 256 | 1.000 | 0.953 | 0.828 | 0.000 | 1.000 | 0.047 | direct_absolute:256, comparison:149, body_ref:70 |
| 104 | 192 | 192 | 0.995 | 0.840 | 0.812 | 0.135 | 0.995 | 0.010 | direct_absolute:177, comparison:22, equality:18 |
| 105 | 64 | 64 | 0.984 | 0.952 | 0.938 | 0.047 | 0.984 | 0.016 | direct_absolute:64, comparison:24, body_ref:1 |
| 106 | 192 | 191 | 0.854 | 0.918 | 0.766 | 0.057 | 0.854 | 0.146 | direct_absolute:167, comparison:121, body_ref:20, superlative:19, equality:9 |
| 107 | 192 | 190 | 0.990 | 0.881 | 0.724 | 0.052 | 0.990 | 0.026 | direct_absolute:190, comparison:65, ranking:25, negation:20, superlative:5 |
| 108 | 640 | 638 | 0.975 | 0.928 | 0.889 | 0.061 | 0.975 | 0.033 | direct_absolute:636, comparison:506, body_ref:158, superlative:94, negation:9 |
| 109 | 448 | 445 | 0.993 | 0.926 | 0.781 | 0.031 | 0.993 | 0.004 | direct_absolute:445, superlative:57, comparison:45, body_ref:7, ranking:5 |
| 110 | 448 | 447 | 0.998 | 0.906 | 0.732 | 0.022 | 0.998 | 0.002 | direct_absolute:447, comparison:2, empty:1 |
| 111 | 320 | 320 | 1.000 | 0.937 | 0.884 | 0.022 | 1.000 | 0.003 | direct_absolute:320, comparison:78, superlative:3, equality:3 |
| 112 | 384 | 379 | 0.974 | 0.903 | 0.839 | 0.073 | 0.974 | 0.016 | direct_absolute:374, comparison:302, empty:5, equality:5, other:4 |
| 113 | 384 | 383 | 0.987 | 0.925 | 0.857 | 0.047 | 0.987 | 0.068 | direct_absolute:375, comparison:216, superlative:41, equality:33, ranking:5 |
| 114 | 512 | 512 | 0.994 | 0.918 | 0.875 | 0.066 | 0.994 | 0.006 | direct_absolute:512, comparison:203, equality:35, superlative:8, negation:1 |
| 115 | 128 | 128 | 1.000 | 0.916 | 0.883 | 0.062 | 1.000 | 0.000 | direct_absolute:128, comparison:34, count_abstract:2 |
| 116 | 128 | 128 | 0.984 | 0.894 | 0.750 | 0.031 | 0.984 | 0.039 | direct_absolute:128, comparison:27 |
| 117 | 128 | 128 | 1.000 | 0.988 | 0.977 | 0.008 | 1.000 | 0.000 | direct_absolute:128, count_abstract:8 |
| 118 | 256 | 256 | 0.801 | 0.915 | 0.613 | 0.031 | 0.801 | 0.254 | direct_absolute:254, comparison:123, count_abstract:50, superlative:35, equality:15 |
| 119 | 768 | 762 | 0.725 | 0.925 | 0.661 | 0.048 | 0.725 | 0.267 | direct_absolute:755, comparison:556, superlative:80, body_ref:55, count_abstract:49 |
| 120 | 512 | 511 | 0.955 | 0.883 | 0.756 | 0.041 | 0.955 | 0.057 | direct_absolute:504, comparison:474, body_ref:344, equality:28, superlative:15 |
| 121 | 448 | 446 | 0.971 | 0.910 | 0.799 | 0.040 | 0.971 | 0.060 | direct_absolute:439, comparison:306, equality:42, group_sum:36, superlative:19 |
| 122 | 256 | 256 | 1.000 | 0.766 | 0.559 | 0.148 | 1.000 | 0.000 | direct_absolute:256, comparison:75, group_sum:75, body_ref:1 |
| 123 | 512 | 510 | 0.994 | 0.908 | 0.781 | 0.029 | 0.994 | 0.014 | direct_absolute:509, superlative:75, comparison:71, equality:15, ranking:3 |
| 124 | 128 | 128 | 1.000 | 0.943 | 0.859 | 0.023 | 1.000 | 0.000 | direct_absolute:128 |
| 125 | 128 | 128 | 1.000 | 0.976 | 0.906 | 0.000 | 1.000 | 0.125 | direct_absolute:128, comparison:62, superlative:7, equality:6, group_sum:1 |
| 126 | 128 | 128 | 1.000 | 0.901 | 0.688 | 0.008 | 1.000 | 0.000 | direct_absolute:128 |
| 127 | 256 | 256 | 0.996 | 0.952 | 0.906 | 0.023 | 0.996 | 0.008 | direct_absolute:256, superlative:64, comparison:59, equality:9, ranking:1 |
| 128 | 192 | 192 | 1.000 | 0.926 | 0.688 | 0.010 | 1.000 | 0.000 | direct_absolute:192 |
| 129 | 256 | 256 | 0.762 | 0.829 | 0.621 | 0.125 | 0.762 | 0.242 | direct_absolute:209, body_ref:101, comparison:94, equality:44, count_abstract:21 |
| 130 | 448 | 419 | 0.850 | 0.841 | 0.656 | 0.121 | 0.850 | 0.085 | direct_absolute:397, comparison:249, group_sum:54, count_abstract:41, empty:29 |
| 131 | 192 | 192 | 0.896 | 0.833 | 0.667 | 0.089 | 0.896 | 0.203 | direct_absolute:171, comparison:38, body_ref:22, equality:18, other:9 |
| 132 | 384 | 361 | 0.904 | 0.968 | 0.875 | 0.029 | 0.904 | 0.036 | direct_absolute:361, comparison:259, empty:23, superlative:23, negation:11 |
| 201 | 640 | 640 | 1.000 | 0.871 | 0.759 | 0.077 | 1.000 | 0.000 | direct_absolute:640, comparison:190, equality:30 |
| 202 | 256 | 255 | 0.992 | 0.949 | 0.895 | 0.008 | 0.992 | 0.004 | direct_absolute:254, comparison:38, empty:1, meta:1 |
| 203 | 832 | 807 | 0.968 | 0.937 | 0.778 | 0.008 | 0.968 | 0.012 | direct_absolute:805, superlative:86, comparison:60, empty:25, ranking:15 |
| 204 | 512 | 512 | 0.996 | 0.914 | 0.834 | 0.033 | 0.996 | 0.027 | direct_absolute:511, comparison:104, negation:39, equality:29, superlative:11 |
| 205 | 704 | 704 | 0.999 | 0.839 | 0.716 | 0.114 | 0.999 | 0.006 | direct_absolute:703, superlative:522, equality:129, body_ref:30, comparison:3 |
| 206 | 1408 | 1408 | 0.994 | 0.892 | 0.789 | 0.035 | 0.994 | 0.016 | direct_absolute:1403, comparison:573, superlative:72, body_ref:31, equality:11 |
| 207 | 960 | 959 | 0.996 | 0.891 | 0.775 | 0.056 | 0.996 | 0.028 | direct_absolute:959, superlative:474, comparison:131, equality:23, ranking:14 |
| 208 | 576 | 571 | 0.964 | 0.933 | 0.811 | 0.023 | 0.964 | 0.038 | direct_absolute:570, comparison:101, superlative:77, equality:27, empty:5 |
| 209 | 512 | 512 | 1.000 | 0.969 | 0.885 | 0.006 | 1.000 | 0.055 | direct_absolute:512, superlative:87, comparison:77, equality:32 |
| 210 | 1728 | 1703 | 0.922 | 0.921 | 0.740 | 0.035 | 0.922 | 0.069 | direct_absolute:1693, superlative:677, comparison:420, equality:309, count_abstract:89 |
| 211 | 960 | 959 | 0.998 | 0.897 | 0.818 | 0.064 | 0.998 | 0.006 | direct_absolute:959, superlative:287, equality:127, comparison:38, ranking:8 |
| 212 | 1792 | 1782 | 0.975 | 0.912 | 0.816 | 0.028 | 0.975 | 0.076 | direct_absolute:1748, comparison:751, body_ref:211, superlative:188, equality:35 |
| 213 | 1536 | 1535 | 0.986 | 0.911 | 0.783 | 0.028 | 0.986 | 0.020 | direct_absolute:1534, comparison:956, body_ref:36, equality:31, count_abstract:24 |
| 214 | 1792 | 1724 | 0.935 | 0.895 | 0.769 | 0.052 | 0.935 | 0.030 | direct_absolute:1624, equality:76, empty:68, comparison:41, meta:39 |
| 215 | 768 | 763 | 0.947 | 0.910 | 0.773 | 0.016 | 0.947 | 0.051 | direct_absolute:763, comparison:52, superlative:45, count_abstract:34, equality:19 |
| 216 | 768 | 765 | 0.988 | 0.901 | 0.732 | 0.025 | 0.988 | 0.023 | direct_absolute:764, comparison:130, superlative:104, equality:45, negation:4 |
| 217 | 1408 | 1407 | 0.998 | 0.853 | 0.712 | 0.060 | 0.998 | 0.004 | direct_absolute:1407, comparison:158, superlative:58, body_ref:34, equality:5 |
| 218 | 1664 | 1658 | 0.978 | 0.923 | 0.772 | 0.019 | 0.978 | 0.065 | direct_absolute:1656, superlative:570, comparison:502, equality:39, body_ref:7 |
| 219 | 1024 | 1021 | 0.959 | 0.904 | 0.789 | 0.044 | 0.959 | 0.050 | direct_absolute:980, comparison:196, equality:49, superlative:16, group_sum:11 |
| 220 | 1408 | 1406 | 0.984 | 0.885 | 0.770 | 0.067 | 0.984 | 0.058 | direct_absolute:1397, comparison:487, equality:249, superlative:245, count_abstract:85 |
| 221 | 832 | 830 | 0.993 | 0.864 | 0.769 | 0.069 | 0.993 | 0.005 | direct_absolute:823, comparison:299, equality:39, superlative:21, group_sum:3 |
| 222 | 832 | 665 | 0.770 | 0.851 | 0.588 | 0.079 | 0.770 | 0.132 | direct_absolute:627, comparison:188, empty:167, superlative:129, equality:117 |
| 223 | 1280 | 1278 | 0.998 | 0.910 | 0.673 | 0.009 | 0.998 | 0.002 | direct_absolute:1278, superlative:125, comparison:39, empty:2, equality:1 |
| 224 | 1664 | 1660 | 0.995 | 0.917 | 0.785 | 0.028 | 0.995 | 0.017 | direct_absolute:1657, superlative:319, comparison:290, count_abstract:104, equality:76 |
| 225 | 704 | 701 | 0.993 | 0.908 | 0.795 | 0.028 | 0.993 | 0.003 | direct_absolute:701, superlative:59, comparison:42, body_ref:24, equality:8 |
| 226 | 640 | 636 | 0.906 | 0.867 | 0.709 | 0.072 | 0.906 | 0.169 | direct_absolute:612, comparison:88, equality:61, count_abstract:28, superlative:13 |
| 227 | 448 | 448 | 0.960 | 0.929 | 0.824 | 0.036 | 0.960 | 0.047 | direct_absolute:448, equality:53, superlative:11, comparison:9, count_abstract:2 |
| 228 | 640 | 609 | 0.920 | 0.864 | 0.680 | 0.056 | 0.920 | 0.031 | direct_absolute:587, comparison:117, superlative:35, empty:31, equality:24 |
| 229 | 1088 | 1086 | 0.998 | 0.932 | 0.903 | 0.040 | 0.998 | 0.000 | direct_absolute:1086, equality:22, superlative:22, empty:2 |
| 230 | 640 | 638 | 0.997 | 0.894 | 0.816 | 0.045 | 0.997 | 0.000 | direct_absolute:630, superlative:74, equality:15, comparison:7, empty:2 |
| 231 | 1344 | 1342 | 0.995 | 0.957 | 0.902 | 0.012 | 0.995 | 0.017 | direct_absolute:1340, comparison:300, superlative:238, equality:96, ranking:6 |
| 232 | 512 | 504 | 0.984 | 0.931 | 0.758 | 0.004 | 0.984 | 0.000 | direct_absolute:504, comparison:222, empty:8, equality:1 |
| 301 | 640 | 640 | 1.000 | 0.895 | 0.759 | 0.022 | 1.000 | 0.052 | direct_absolute:640, comparison:156, equality:37, count_abstract:13, superlative:6 |
| 302 | 448 | 446 | 0.993 | 0.940 | 0.846 | 0.007 | 0.993 | 0.029 | direct_absolute:446, comparison:41, negation:9, equality:2, empty:2 |
| 303 | 192 | 191 | 0.995 | 0.932 | 0.755 | 0.000 | 0.995 | 0.005 | direct_absolute:191, comparison:6, superlative:1, empty:1 |
| 304 | 320 | 319 | 0.988 | 0.931 | 0.903 | 0.059 | 0.988 | 0.009 | direct_absolute:318, equality:16, body_ref:14, empty:1 |
| 305 | 192 | 192 | 0.875 | 0.908 | 0.667 | 0.010 | 0.875 | 0.172 | direct_absolute:168, other:22, comparison:2, negation:1, count_abstract:1 |
| 306 | 768 | 768 | 1.000 | 0.916 | 0.809 | 0.016 | 1.000 | 0.047 | direct_absolute:768, comparison:197, equality:81, superlative:19, body_ref:14 |
| 307 | 704 | 702 | 0.996 | 0.928 | 0.795 | 0.011 | 0.996 | 0.013 | direct_absolute:700, comparison:416, body_ref:173, equality:42, empty:2 |
| 308 | 896 | 889 | 0.978 | 0.904 | 0.767 | 0.028 | 0.978 | 0.030 | direct_absolute:889, comparison:560, superlative:292, body_ref:160, equality:156 |
| 309 | 768 | 768 | 0.990 | 0.927 | 0.833 | 0.009 | 0.990 | 0.013 | direct_absolute:755, comparison:84, superlative:38, equality:18, meta:1 |
| 310 | 1088 | 1085 | 0.996 | 0.913 | 0.756 | 0.023 | 0.996 | 0.034 | direct_absolute:1081, superlative:544, equality:133, comparison:125, body_ref:100 |
| 311 | 1024 | 1023 | 0.998 | 0.923 | 0.848 | 0.032 | 0.998 | 0.043 | direct_absolute:1021, comparison:369, superlative:306, ranking:99, equality:16 |
| 312 | 1344 | 1336 | 0.991 | 0.895 | 0.818 | 0.062 | 0.991 | 0.007 | direct_absolute:1335, comparison:102, equality:88, superlative:18, empty:8 |
| 313 | 960 | 959 | 0.998 | 0.862 | 0.626 | 0.027 | 0.998 | 0.007 | direct_absolute:928, comparison:192, body_ref:142, equality:76, superlative:3 |
| 314 | 832 | 828 | 0.981 | 0.372 | 0.149 | 0.410 | 0.981 | 0.026 | direct_absolute:822, comparison:818, body_ref:722, other:6, empty:4 |
| 315 | 704 | 703 | 0.997 | 0.893 | 0.797 | 0.037 | 0.997 | 0.001 | direct_absolute:701, empty:1, equality:1, other:1 |
| 316 | 384 | 383 | 0.995 | 0.943 | 0.854 | 0.010 | 0.995 | 0.135 | direct_absolute:383, comparison:111, superlative:67, negation:33, equality:10 |
| 317 | 1024 | 1024 | 0.940 | 0.847 | 0.725 | 0.081 | 0.940 | 0.061 | comparison:785, direct_absolute:619, body_ref:400, equality:163, superlative:90 |
| 318 | 576 | 575 | 0.997 | 0.889 | 0.747 | 0.024 | 0.997 | 0.030 | direct_absolute:574, comparison:264, body_ref:122, superlative:30, equality:18 |
| 319 | 1472 | 1441 | 0.933 | 0.883 | 0.737 | 0.058 | 0.933 | 0.050 | direct_absolute:1369, comparison:518, body_ref:309, superlative:276, equality:66 |
| 320 | 768 | 768 | 1.000 | 0.905 | 0.806 | 0.034 | 1.000 | 0.000 | direct_absolute:768, superlative:5 |
| 321 | 768 | 709 | 0.923 | 0.844 | 0.667 | 0.057 | 0.923 | 0.026 | direct_absolute:709, comparison:243, superlative:91, empty:59, count_abstract:14 |
| 322 | 896 | 859 | 0.934 | 0.898 | 0.750 | 0.046 | 0.934 | 0.062 | direct_absolute:859, comparison:285, equality:258, superlative:194, body_ref:150 |
| 323 | 256 | 254 | 0.980 | 0.931 | 0.820 | 0.023 | 0.980 | 0.012 | direct_absolute:251, comparison:70, meta:3, empty:2, count_abstract:1 |
| 324 | 512 | 512 | 1.000 | 0.941 | 0.869 | 0.008 | 1.000 | 0.008 | direct_absolute:512, comparison:16 |
| 325 | 512 | 512 | 0.992 | 0.925 | 0.775 | 0.014 | 0.992 | 0.008 | direct_absolute:512 |
| 326 | 512 | 503 | 0.951 | 0.864 | 0.721 | 0.043 | 0.951 | 0.100 | direct_absolute:483, comparison:68, body_ref:55, equality:21, superlative:21 |
| 327 | 128 | 128 | 1.000 | 0.951 | 0.883 | 0.000 | 1.000 | 0.008 | direct_absolute:128, comparison:8 |
| 328 | 704 | 702 | 0.997 | 0.915 | 0.780 | 0.009 | 0.997 | 0.009 | direct_absolute:702, comparison:440, equality:35, group_sum:11, negation:4 |
| 329 | 384 | 384 | 0.992 | 0.921 | 0.836 | 0.023 | 0.992 | 0.010 | direct_absolute:382, ranking:10, comparison:8, superlative:5, other:2 |
| 330 | 256 | 255 | 0.996 | 0.956 | 0.910 | 0.008 | 0.996 | 0.000 | direct_absolute:255, superlative:28, comparison:11, body_ref:2, empty:1 |
| 331 | 640 | 612 | 0.955 | 0.890 | 0.787 | 0.053 | 0.955 | 0.005 | direct_absolute:604, empty:28, superlative:19, equality:14, body_ref:1 |
| 332 | 128 | 128 | 1.000 | 0.975 | 0.945 | 0.000 | 1.000 | 0.000 | direct_absolute:128, comparison:1 |

## 逐被试报告

### S101

- trial 数: 320; 非空文本: 320; fidelity 可评分率: 1.000; 平均 fidelity: 0.885; 完全忠实率: 0.841; 低 fidelity 率: 0.078.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 320 | 1.000 |
| comparison | 23 | 0.072 |
| equality | 3 | 0.009 |
| body_ref | 2 | 0.006 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子很长。 | 56 |
| 脖子较短。 | 36 |
| 脖子较长。 | 33 |
| 脖子长。 | 23 |
| 脖子很短。 | 20 |
| 脖子短。 | 18 |
| 脖子短，头长。 | 8 |
| 脖子较短，头很长。 | 8 |
| 脖子短，头很长。 | 6 |
| 头长，脖子长。 | 4 |
| 头长，尾巴长。 | 3 |
| 脖子较短，尾巴很长。 | 3 |
| 脖子较长，头比脖子长。 | 3 |
| 脖子长，尾巴长。 | 3 |
| 尾巴短。 | 2 |
| 脖子较短，头很短。 | 2 |
| 脖子长，尾巴短。 | 2 |
| 脖子很短，头很长。 | 2 |
| 脖子较短，头比较长。 | 2 |
| 头很长，脖子很长。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子比躯干长。 | 2 | body_ref | S1T54, S1T80 |
| 四个部位长度比较均匀。 | 1 | equality | S1T231 |
| 尾巴和腿差不多长。 | 1 | equality | S1T21 |
| 脖子和头差不多长。 | 1 | equality | S1T217 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子较长。 | 9 | 0.000 | absolute_long:脖子 > 0.50 | S1T53, S1T84, S1T126, S1T163, S1T165, S1T189, S1T236, S1T280 |
| 脖子较短。 | 6 | 0.000 | absolute_short:脖子 < 0.50 | S1T75, S1T92, S1T261, S1T292, S1T294, S1T319 |
| 脖子长。 | 4 | 0.000 | absolute_long:脖子 > 0.50 | S1T65, S1T78, S1T98, S1T216 |
| 四个部位长度比较均匀。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T231 |
| 脖子比躯干长。 | 1 | 0.000 | body_ref:脖子 > 0.50 | S1T80 |
| 脖子短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S1T96 |
| 脖子较长，头也较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T100 |
| 尾巴短，头长，脖子长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T207 |
| 腿短，脖子长，头长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T40 |

### S102

- trial 数: 448; 非空文本: 448; fidelity 可评分率: 0.993; 平均 fidelity: 0.950; 完全忠实率: 0.882; 低 fidelity 率: 0.018.
- 旧版 region 覆盖率: 0.993; 旧版 region 有未处理片段率: 0.109.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 413 | 0.922 |
| comparison | 274 | 0.612 |
| ranking | 26 | 0.058 |
| superlative | 24 | 0.054 |
| equality | 6 | 0.013 |
| body_ref | 3 | 0.007 |
| group_sum | 3 | 0.007 |
| negation | 2 | 0.004 |
| count_abstract | 2 | 0.004 |
| meta | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子比头长。 | 43 |
| 脖子长。 | 40 |
| 脖子短。 | 40 |
| 脖子比头短。 | 37 |
| 脖子长于头。 | 13 |
| 脖子小于头。 | 12 |
| 尾巴短，腿长，脖子短，头长。 | 9 |
| 尾巴短，腿长，脖子长，头短。 | 8 |
| 脖子大于头。 | 7 |
| 脖子短于头。 | 7 |
| 腿长，尾巴长，脖子短于头。 | 6 |
| 脖子比头短，腿长，尾巴短。 | 5 |
| 从小到大是脖子、尾巴、头。 | 5 |
| 腿长，尾巴长，脖子长于头。 | 4 |
| 脖子比头长，但是是最短的两个。 | 4 |
| 尾巴长，腿短，脖子长，头短。 | 4 |
| 从小到大是头、脖子、尾巴。 | 4 |
| 脖子比头短，腿和尾巴适中。 | 4 |
| 四个部位都偏长。 | 3 |
| 脖子比头短，都短于腿。 | 3 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 从小到大是脖子、尾巴、头。 | 5 | ranking | S1T257, S1T258, S1T259, S1T262, S1T264 |
| 从小到大是头、脖子、尾巴。 | 4 | ranking | S1T261, S1T263, S1T265, S1T266 |
| 脖子长于头，腿和尾巴长度差不多。 | 3 | equality | S1T98, S1T99, S1T100 |
| 长度从小到大排序为脖子、尾巴、头、腿。 | 2 | ranking | S1T42, S1T51 |
| 长度从小到大排序为腿、头、尾巴、脖子。 | 2 | ranking | S1T46, S1T47 |
| 长度从小到大排序为腿、尾巴、脖子、头。 | 2 | ranking | S1T48, S1T49 |
| 三长一中。 | 1 | count_abstract | S1T286 |
| 两长两短。 | 1 | count_abstract | S1T275 |
| 从小到大是头、尾巴、脖子。 | 1 | ranking | S1T260 |
| 从小到大是头、脖子、腿。 | 1 | ranking | S1T293 |
| 从小到大是尾巴、腿、脖子、头。 | 1 | ranking | S1T267 |
| 从小到大是腿、脖子、头。 | 1 | ranking | S1T294 |
| 脖子和头和腿差不多长。 | 1 | equality | S1T302 |
| 脖子和头的组合使其的中段高于身体。 | 1 | body_ref, group_sum | S1T8 |
| 脖子和头的组合使其的终端高于身体。 | 1 | body_ref, group_sum | S1T9 |
| 脖子和头的长度组合使其的终端高于身体。 | 1 | body_ref, group_sum | S1T7 |
| 脖子比头短，而且不是最长的两个。 | 1 | negation | S1T184 |
| 脖子比头短，腿和尾巴差不多，都适中。 | 1 | equality | S1T138 |
| 脖子比头长，不确定是不是最短的两个。 | 1 | meta, negation | S1T157 |
| 脖子短于头，但是它们的平均长度长于尾巴。 | 1 | equality | S1T209 |
| 长度从小到大排序为头、腿、尾巴、脖子。 | 1 | ranking | S1T54 |
| 长度从小到大排序为尾巴、脖子、头、腿。 | 1 | ranking | S1T45 |
| 长度从小到大排序为尾巴、腿、头、脖子。 | 1 | ranking | S1T53 |
| 长度从小到大排序为脖子、头、腿、尾巴。 | 1 | ranking | S1T43 |
| 长度从小到大排序为脖子、尾巴、腿、头。 | 1 | ranking | S1T44 |
| 长度从小到大排序为脖子、腿、头、尾巴。 | 1 | ranking | S1T52 |
| 长度从小到大排序为腿、尾巴、头、脖子。 | 1 | ranking | S1T50 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子长。 | 3 | 0.000 | absolute_long:脖子 > 0.50 | S2T86, S2T102, S2T121 |
| 脖子偏长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T41 |
| 脖子和头和腿差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿 = | S1T302 |
| 脖子小于头。 | 1 | 0.000 | comparison:脖子 < 头 | S1T220 |
| 脖子比头长。 | 1 | 0.000 | comparison:脖子 > 头 | S1T295 |
| 腿短，脖子长，头短。 | 1 | 0.333 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S1T268 |

### S103

- trial 数: 256; 非空文本: 256; fidelity 可评分率: 1.000; 平均 fidelity: 0.953; 完全忠实率: 0.828; 低 fidelity 率: 0.000.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.047.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 256 | 1.000 |
| comparison | 149 | 0.582 |
| body_ref | 70 | 0.273 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 四个均长于一半。 | 9 |
| 只有脖子短于一半。 | 7 |
| 脖子和腿长于一半，其余短于一半。 | 6 |
| 尾巴和腿长于一半，其余短于一半。 | 4 |
| 头和脖子长于一半，其余短于一半。 | 4 |
| 只有头短于一半。 | 4 |
| 只有腿短。 | 3 |
| 只有腿短于一半。 | 3 |
| 尾巴和头长于一半，其余短于一半。 | 3 |
| 头和尾巴长于一半，其余短于一半。 | 2 |
| 四个部位均长于一半。 | 2 |
| 头和腿长于一半，其余短于一半。 | 2 |
| 尾巴和脖子长于一半，其余短于一半。 | 2 |
| 四个都很短。 | 2 |
| 只有头短。 | 2 |
| 只有头较短，其余都长于一半。 | 2 |
| 只有脖子长。 | 2 |
| 尾巴和头较长，其余较短。 | 2 |
| 头和腿长，其余短。 | 2 |
| 只有脖子较长，其余都很短。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个均长于一半。 | 9 | body_ref | S1T148, S1T149, S1T167, S1T197, S1T208, S1T211, S1T212, S1T213 |
| 只有脖子短于一半。 | 7 | body_ref | S1T175, S1T183, S1T186, S1T190, S1T221, S1T224, S1T228 |
| 脖子和腿长于一半，其余短于一半。 | 6 | body_ref | S1T172, S1T174, S1T182, S1T189, S1T194, S1T209 |
| 只有头短于一半。 | 4 | body_ref | S1T164, S1T195, S1T205, S1T222 |
| 头和脖子长于一半，其余短于一半。 | 4 | body_ref | S1T168, S1T196, S1T198, S1T227 |
| 尾巴和腿长于一半，其余短于一半。 | 4 | body_ref | S1T173, S1T203, S1T220, S1T226 |
| 只有腿短于一半。 | 3 | body_ref | S1T169, S1T179, S1T223 |
| 尾巴和头长于一半，其余短于一半。 | 3 | body_ref | S1T163, S1T225, S1T249 |
| 只有头较短，其余都长于一半。 | 2 | body_ref | S1T150, S1T154 |
| 只有尾巴短于一半。 | 2 | body_ref | S1T165, S1T184 |
| 四个部位均长于一半。 | 2 | body_ref | S1T177, S1T180 |
| 头和尾巴长于一半，其余短于一半。 | 2 | body_ref | S1T187, S1T199 |
| 头和腿长于一半，其余短于一半。 | 2 | body_ref | S1T207, S1T210 |
| 尾巴和脖子长于一半，其余短于一半。 | 2 | body_ref | S1T171, S1T178 |
| 只有头很长，长于一半。 | 1 | body_ref | S1T218 |
| 只有头长于一半。 | 1 | body_ref | S1T166 |
| 只有尾巴极短，短于一半。 | 1 | body_ref | S1T201 |
| 只有尾巴长于一半。 | 1 | body_ref | S1T176 |
| 只有腿中等偏短，其余都大于一半，尾巴、头、脖子都较长。 | 1 | body_ref | S1T62 |
| 只有腿很短，短于一半。 | 1 | body_ref | S1T215 |
| 只有腿略短于一半。 | 1 | body_ref | S1T188 |
| 只有腿长于一半。 | 1 | body_ref | S1T191 |
| 四个均短于一半。 | 1 | body_ref | S1T170 |
| 四个均长约一半 | 1 | body_ref | S1T193 |
| 四个均长约一半。 | 1 | body_ref | S1T192 |
| 四个都短于一半。 | 1 | body_ref | S1T160 |
| 头和尾巴很长，其余短于一半。 | 1 | body_ref | S1T219 |
| 头和尾巴很长，长于一半，其余短于一半。 | 1 | body_ref | S1T216 |
| 头和脖子短于一半，其余长于一半。 | 1 | body_ref | S1T181 |
| 尾巴很长，尾巴和脖子长于一半，其余短于一半。 | 1 | body_ref | S1T204 |
| 脖子、腿都极短，尾巴和头不到一半。 | 1 | body_ref | S1T86 |
| 脖子和腿长于一半。 | 1 | body_ref | S1T233 |

低忠实率对应试次（fidelity < 0.5）：
无。

### S104

- trial 数: 192; 非空文本: 192; fidelity 可评分率: 0.995; 平均 fidelity: 0.840; 完全忠实率: 0.812; 低 fidelity 率: 0.135.
- 旧版 region 覆盖率: 0.995; 旧版 region 有未处理片段率: 0.010.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 177 | 0.922 |
| comparison | 22 | 0.115 |
| equality | 18 | 0.094 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿短。 | 23 |
| 头长。 | 23 |
| 腿长。 | 20 |
| 四个部位比较均匀。 | 14 |
| 头短。 | 13 |
| 尾巴短。 | 12 |
| 脖子长。 | 9 |
| 头长，脖子长。 | 9 |
| 脖子长，腿长。 | 9 |
| 头长，腿长。 | 7 |
| 头长，尾巴长。 | 5 |
| 脖子短。 | 5 |
| 腿长，尾巴长。 | 4 |
| 尾巴比较短。 | 3 |
| 头和尾巴长。 | 3 |
| 头和脖子长。 | 2 |
| 头和腿长。 | 2 |
| 腿短，尾巴短。 | 2 |
| 头短，腿长。 | 2 |
| 脖子长，尾巴长。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位比较均匀。 | 14 | equality | S1T3, S1T6, S1T10, S1T16, S1T19, S1T33, S1T39, S1T40 |
| 头长，四个部位比较均匀。 | 1 | equality | S1T138 |
| 头长，较均匀。 | 1 | equality | S1T109 |
| 比较均匀。 | 1 | equality | S1T139 |
| 腿短，四个部位比较均匀。 | 1 | equality | S1T182 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位比较均匀。 | 14 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T3, S1T6, S1T10, S1T16, S1T19, S1T33, S1T39, S1T40 |
| 头长。 | 4 | 0.000 | absolute_long:头 > 0.50 | S1T71, S1T73, S1T106, S1T156 |
| 尾巴比较短。 | 2 | 0.000 | absolute_short:尾巴 < 0.50 | S1T2, S1T4 |
| 脖子长。 | 2 | 0.000 | absolute_long:脖子 > 0.50 | S1T166, S1T167 |
| 尾巴比较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T1 |
| 脖子长，尾巴长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T59 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T154 |
| 腿长，脖子长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S1T118 |

### S105

- trial 数: 64; 非空文本: 64; fidelity 可评分率: 0.984; 平均 fidelity: 0.952; 完全忠实率: 0.938; 低 fidelity 率: 0.047.
- 旧版 region 覆盖率: 0.984; 旧版 region 有未处理片段率: 0.016.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 64 | 1.000 |
| comparison | 24 | 0.375 |
| body_ref | 1 | 0.016 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿很长。 | 17 |
| 腿很短。 | 14 |
| 腿比较长。 | 11 |
| 腿比较短。 | 11 |
| 腿长。 | 2 |
| 腿非常短。 | 2 |
| 脖子很短，腿、尾巴和头都很长。 | 1 |
| 脖子很短。 | 1 |
| 腿长于躯干的一半。 | 1 |
| 腿略短。 | 1 |
| 腿短于阈值。 | 1 |
| 腿略长。 | 1 |
| 腿非常长。 | 1 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿长于躯干的一半。 | 1 | body_ref | S1T13 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿比较长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S1T37, S1T44 |
| 腿比较短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S1T52 |

### S106

- trial 数: 192; 非空文本: 191; fidelity 可评分率: 0.854; 平均 fidelity: 0.918; 完全忠实率: 0.766; 低 fidelity 率: 0.057.
- 旧版 region 覆盖率: 0.854; 旧版 region 有未处理片段率: 0.146.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 167 | 0.870 |
| comparison | 121 | 0.630 |
| body_ref | 20 | 0.104 |
| superlative | 19 | 0.099 |
| equality | 9 | 0.047 |
| negation | 9 | 0.047 |
| other | 6 | 0.031 |
| group_sum | 6 | 0.031 |
| empty | 1 | 0.005 |
| count_abstract | 1 | 0.005 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿比较短。 | 26 |
| 腿比较长。 | 19 |
| 腿足够长。 | 13 |
| 腿比躯干短。 | 9 |
| 腿比一半短。 | 7 |
| 躯体下方比上面高。 | 7 |
| 腿是最长的。 | 7 |
| 头和脖子加起来比腿长。 | 5 |
| 腿不够长。 | 3 |
| 腿是最短的。 | 3 |
| 脖子比腿长。 | 3 |
| 腿很长。 | 3 |
| 腿短。 | 3 |
| 头比脖子长。 | 2 |
| 躯体下方没有上面高。 | 2 |
| 腿比脖子长。 | 2 |
| 脖子比头长。 | 2 |
| 头离地面近。 | 2 |
| 头比脖子短，腿长。 | 2 |
| 头比脖子短，腿短。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿比躯干短。 | 9 | body_ref | S1T129, S1T131, S1T132, S1T134, S1T135, S1T136, S1T137, S1T138 |
| 腿比一半短。 | 7 | body_ref | S1T90, S1T95, S1T97, S1T99, S1T101, S1T102, S1T107 |
| 头和脖子加起来比腿长。 | 5 | group_sum | S1T38, S1T39, S1T40, S1T41, S1T42 |
| 腿不够长。 | 3 | negation | S1T91, S1T103, S1T115 |
| 下面的部分更高一些。 | 2 | other | S1T174, S1T175 |
| 各部位长度相近。 | 2 | equality | S1T2, S1T17 |
| 头离地面近。 | 2 | other | S1T32, S1T33 |
| 腿不是最长的。 | 2 | negation | S1T55, S1T58 |
| 躯体下方没有上面高。 | 2 | negation | S1T180, S1T182 |
| 头和尾一样长。 | 1 | equality | S1T16 |
| 头和脖子加起来跟腿差不多长。 | 1 | equality, group_sum | S1T44 |
| 头离地面远。 | 1 | other | S1T34 |
| 有两个部位一样长。 | 1 | equality, count_abstract | S1T15 |
| 点错了。 | 1 | other | S1T105 |
| 脖子与尾巴差不多长。 | 1 | equality | S1T73 |
| 脖子比头长，脖子和腿差不多长。 | 1 | equality | S1T78 |
| 腿和脖子差不多长，脖子比头长。 | 1 | equality | S1T80 |
| 腿比尾巴长，腿没有头长。 | 1 | negation | S1T74 |
| 腿比躯干略短。 | 1 | body_ref | S1T133 |
| 腿比躯干短一些。 | 1 | body_ref | S1T141 |
| 腿比躯干长。 | 1 | body_ref | S1T130 |
| 腿跟躯干差不多长。 | 1 | equality, body_ref | S1T140 |
| 躯体下方没有上面高，腿太短。 | 1 | negation | S1T188 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿足够长。 | 3 | 0.000 | absolute_long:腿 > 0.50 | S1T113, S1T123, S1T124 |
| 各部位长度相近。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T2, S1T17 |
| 头和脖子加起来跟腿差不多长。 | 1 | 0.000 | equality_range:头+脖子+腿 = | S1T44 |
| 尾巴相对较短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T6 |
| 腿比躯干略短。 | 1 | 0.000 | body_ref:腿 < 0.50 | S1T133 |
| 腿比躯干短一些。 | 1 | 0.000 | body_ref:腿 < 0.50 | S1T141 |
| 腿比较长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T153 |
| 腿跟躯干差不多长。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S1T140 |

### S107

- trial 数: 192; 非空文本: 190; fidelity 可评分率: 0.990; 平均 fidelity: 0.881; 完全忠实率: 0.724; 低 fidelity 率: 0.052.
- 旧版 region 覆盖率: 0.990; 旧版 region 有未处理片段率: 0.026.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 190 | 0.990 |
| comparison | 65 | 0.339 |
| ranking | 25 | 0.130 |
| negation | 20 | 0.104 |
| superlative | 5 | 0.026 |
| body_ref | 5 | 0.026 |
| equality | 4 | 0.021 |
| empty | 2 | 0.010 |
| group_sum | 1 | 0.005 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿和脖子都很短。 | 3 |
| 脖子尤其的长。 | 3 |
| 从长到短是脖子、头、尾巴、腿。 | 3 |
| 脖子和头都比较短。 | 3 |
| 脖子和头都非常长，腿也不算短。 | 2 |
| 腿和脖子都很长。 | 2 |
| 腿比较长，脖子非常长。 | 2 |
| 腿、脖子、头都挺长的。 | 2 |
| 腿比较短，但脖子很长。 | 2 |
| 脖子和尾巴长，头和腿短。 | 2 |
| 脖子和头比腿长。 | 2 |
| 头、脖子、腿、尾巴都比较短。 | 2 |
| 脖子和头都很长。 | 2 |
| 脖子和腿相对长，头和尾巴相对短。 | 2 |
| 腿短，但脖子长，头也长。 | 2 |
| 腿、脖子和头都很长。 | 2 |
| 从长到短是脖子、腿、尾巴、头。 | 2 |
| 从长到短是头、脖子、尾巴、腿。 | 2 |
| 从长到短是尾巴、脖子、头、腿。 | 2 |
| 从长到短是脖子、尾巴、头、腿。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 从长到短是脖子、头、尾巴、腿。 | 3 | ranking | S1T9, S1T11, S1T23 |
| 从长到短是头、脖子、尾巴、腿。 | 2 | ranking | S1T20, S1T28 |
| 从长到短是头、脖子、腿、尾巴。 | 2 | ranking | S1T18, S1T29 |
| 从长到短是尾巴、脖子、头、腿。 | 2 | ranking | S1T13, S1T16 |
| 从长到短是脖子、尾巴、头、腿。 | 2 | ranking | S1T10, S1T12 |
| 从长到短是脖子、腿、尾巴、头。 | 2 | ranking | S1T17, S1T19 |
| 脖子和头都非常长，腿也不算短。 | 2 | negation | S1T101, S1T108 |
| 从长到短是头、腿、脖子、尾巴。 | 1 | ranking | S1T21 |
| 从长到短是头最长，尾巴也比较长，腿比较短，脖子最短。 | 1 | ranking | S1T33 |
| 从长到短是尾巴、头、躯干、腿、脖子。 | 1 | ranking, body_ref | S1T36 |
| 从长到短是尾巴、腿、躯干、脖子、头。 | 1 | ranking, body_ref | S1T38 |
| 从长到短是尾巴、躯干、腿、头、脖子。 | 1 | ranking, body_ref | S1T37 |
| 从长到短是脖子、头、腿、尾巴，但是四个部位都比躯干短。 | 1 | ranking, body_ref | S1T35 |
| 从长到短是腿、头、脖子、尾巴。 | 1 | ranking | S1T14 |
| 从长到短是腿、尾巴、脖子、头。 | 1 | ranking | S1T24 |
| 从长到短是腿、脖子、头、尾巴。 | 1 | ranking | S1T22 |
| 从长到短是腿、脖子、尾巴、头。 | 1 | ranking | S1T15 |
| 头、脖子、尾巴和腿都不长，头相对长一些。 | 1 | negation | S1T54 |
| 头、脖子和尾巴都比较长，比躯干长，腿很短。 | 1 | body_ref | S1T39 |
| 头最长，其次是尾巴，脖子和腿相对较短。 | 1 | ranking | S1T43 |
| 尾巴、腿、脖子和头长度差不多，都还挺长的。 | 1 | equality | S1T164 |
| 尾巴最长，其次是腿，然后是脖子和头。 | 1 | ranking | S1T31 |
| 尾巴最长，腿、脖子、头差不多长。 | 1 | equality | S1T4 |
| 尾巴长，脖子短，腿和头不算长。 | 1 | negation | S1T123 |
| 脖子很短，头和腿也不算长。 | 1 | negation | S1T113 |
| 腿、脖子、头和尾巴都不算长。 | 1 | negation | S1T103 |
| 腿、脖子和头不算特别长，且长度相当。 | 1 | equality, negation | S1T191 |
| 腿、脖子和头比例比较协调，尾巴短。 | 1 | group_sum | S1T95 |
| 腿、脖子和头都不算很长，脖子和头尤其相对短一些，尾巴也很短。 | 1 | negation | S1T119 |
| 腿、脖子和头长度差不多，都不算很长。 | 1 | equality, negation | S1T111 |
| 腿不算很长，脖子和头都很长。 | 1 | negation | S1T142 |
| 腿不算长，脖子很短，头也不长。 | 1 | negation | S1T105 |
| 腿和头不算很长，脖子比较短。 | 1 | negation | S1T112 |
| 腿和脖子都不长，头长一些。 | 1 | negation | S1T147 |
| 腿很短，头、脖子也不算很长。 | 1 | negation | S1T81 |
| 腿很短，脖子也不长。 | 1 | negation | S1T132 |
| 腿很短，脖子和头不算很长，尾巴特别长。 | 1 | negation | S1T176 |
| 腿比较短，腿很短，脖子也不长。 | 1 | negation | S1T192 |
| 腿比较长，脖子和头也不算很短。 | 1 | negation | S1T117 |
| 腿短，尾巴长一些，脖子和头不算很长。 | 1 | negation | S1T78 |
| 腿长，脖子和头都不长。 | 1 | negation | S1T135 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴、腿、脖子和头长度差不多，都还挺长的。 | 1 | 0.000 | equality_range:尾巴+腿+脖子+头 = | S1T164 |
| 腿、脖子和头不算特别长，且长度相当。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S1T191 |
| 腿、脖子和头比例比较协调，尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T95 |
| 腿、脖子和头长度差不多，都不算很长。 | 1 | 0.000 | equality_range:腿+脖子+头 = | S1T111 |
| 头、脖子、腿、尾巴都比较短。 | 1 | 0.250 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T58 |
| 头、脖子、腿和尾巴都挺短。 | 1 | 0.250 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T67 |
| 腿和头不算很长，脖子比较短。 | 1 | 0.333 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S1T112 |
| 腿很短，头、脖子也不算很长。 | 1 | 0.333 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50 | S1T81 |
| 腿比较长，脖子和头也不算很短。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T117 |
| 从长到短是尾巴、头、躯干、腿、脖子。 | 1 | 0.429 | body_ref:尾巴 = 0.50; body_ref:头 = 0.50; body_ref:腿 = 0.50; body_ref:脖子 = 0.50 | S1T36 |

### S108

- trial 数: 640; 非空文本: 638; fidelity 可评分率: 0.975; 平均 fidelity: 0.928; 完全忠实率: 0.889; 低 fidelity 率: 0.061.
- 旧版 region 覆盖率: 0.975; 旧版 region 有未处理片段率: 0.033.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 636 | 0.994 |
| comparison | 506 | 0.791 |
| body_ref | 158 | 0.247 |
| superlative | 94 | 0.147 |
| negation | 9 | 0.014 |
| ranking | 4 | 0.006 |
| empty | 2 | 0.003 |
| meta | 2 | 0.003 |
| count_abstract | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿比脖子长。 | 130 |
| 腿比脖子短。 | 126 |
| 脖子比躯干短。 | 88 |
| 脖子比躯干长。 | 68 |
| 脖子是最长的。 | 19 |
| 脖子最长。 | 14 |
| 腿最长。 | 9 |
| 头是最长的。 | 9 |
| 脖子不是最长的。 | 8 |
| 脖子最短。 | 7 |
| 腿比脖子更长。 | 6 |
| 脖子短。 | 6 |
| 腿是最长的。 | 6 |
| 尾巴是最长的。 | 6 |
| 脖子比头短。 | 6 |
| 脖子比腿长。 | 5 |
| 脖子比头和腿长。 | 4 |
| 腿比脖子短，头很长。 | 4 |
| 头最长。 | 4 |
| 腿比脖子更短。 | 3 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子比躯干短。 | 88 | body_ref | S2T122, S2T123, S2T169, S2T172, S2T173, S2T174, S2T175, S2T176 |
| 脖子比躯干长。 | 68 | body_ref | S2T119, S2T120, S2T121, S2T167, S2T168, S2T170, S2T171, S2T177 |
| 脖子不是最长的。 | 8 | negation | S2T34, S2T35, S2T36, S2T37, S2T38, S2T39, S2T43, S2T44 |
| 腿第二长。 | 2 | ranking | S2T135, S2T137 |
| 选错了。 | 2 | meta | S1T225, S2T69 |
| 头是最长的，脖子是第二长的。 | 1 | ranking | S2T47 |
| 所有部位都比躯干长。 | 1 | body_ref | S2T164 |
| 脖子不是最长的，腿是最长的。 | 1 | negation | S2T48 |
| 脖子是第三长的。 | 1 | ranking, count_abstract | S2T78 |
| 腿和脖子比躯干长。 | 1 | body_ref | S2T145 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子比躯干短。 | 16 | 0.000 | body_ref:脖子 < 0.50 | S2T123, S2T176, S2T178, S2T188, S2T192, S2T198, S2T200, S2T201 |
| 腿比脖子短。 | 12 | 0.000 | comparison:腿 < 脖子 | S1T28, S1T130, S1T133, S1T135, S1T154, S1T176, S1T178, S1T214 |
| 腿比脖子长。 | 6 | 0.000 | comparison:腿 > 脖子 | S1T128, S1T239, S1T241, S1T242, S1T252, S1T285 |
| 脖子是第三长的。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T78 |
| 脖子比腿短。 | 1 | 0.000 | comparison:脖子 < 腿 | S2T128 |
| 脖子短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S2T68 |
| 腿比脖子更短。 | 1 | 0.000 | comparison:腿 < 脖子 | S1T40 |
| 腿比较短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S2T136 |

### S109

- trial 数: 448; 非空文本: 445; fidelity 可评分率: 0.993; 平均 fidelity: 0.926; 完全忠实率: 0.781; 低 fidelity 率: 0.031.
- 旧版 region 覆盖率: 0.993; 旧版 region 有未处理片段率: 0.004.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 445 | 0.993 |
| superlative | 57 | 0.127 |
| comparison | 45 | 0.100 |
| body_ref | 7 | 0.016 |
| ranking | 5 | 0.011 |
| empty | 3 | 0.007 |
| equality | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头较短。 | 36 |
| 头较长。 | 28 |
| 腿最长。 | 17 |
| 脖子最长。 | 15 |
| 头最长。 | 14 |
| 腿比头长。 | 10 |
| 腿较长，脖子一般，头一般。 | 10 |
| 头较长，脖子一般，腿一般。 | 9 |
| 头一般。 | 8 |
| 脖子比头长。 | 7 |
| 头很长，脖子一般，腿一般。 | 6 |
| 尾巴比头短。 | 6 |
| 脖子较长，腿一般，头一般。 | 6 |
| 脖子很长。 | 5 |
| 腿较长，脖子、头一般。 | 5 |
| 脖子较长，头一般，腿较短。 | 5 |
| 脖子较长，腿一般，头较短。 | 4 |
| 头比腿长。 | 4 |
| 尾巴较短。 | 4 |
| 腿很长，尾巴一般，头一般，脖子较短。 | 3 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿比躯干短。 | 3 | body_ref | S2T62, S2T63, S2T64 |
| 腿比躯干长。 | 2 | body_ref | S2T60, S2T61 |
| 尾巴比躯干长。 | 1 | body_ref | S2T59 |
| 脖子和头一样长。 | 1 | equality | S2T7 |
| 脖子最长，头次之，腿、尾巴较短。 | 1 | ranking | S1T58 |
| 脖子最长，头次之，腿和尾巴较短。 | 1 | ranking | S1T50 |
| 脖子比躯干长。 | 1 | body_ref | S2T58 |
| 脖子较长，头次之，腿最短。 | 1 | ranking | S1T240 |
| 腿最长，尾巴、脖子次之。 | 1 | ranking | S1T39 |
| 腿最长，脖子次之，头第三，尾巴最短。 | 1 | ranking | S1T49 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头较短。 | 3 | 0.000 | absolute_short:头 < 0.50 | S2T48, S2T78, S2T84 |
| 脖子最长。 | 3 | 0.111 | superlative:脖子 > 腿; superlative:脖子 > 尾巴; superlative:脖子 > 头 | S1T273, S1T283, S1T300 |
| 腿比躯干短。 | 2 | 0.000 | body_ref:腿 < 0.50 | S2T62, S2T63 |
| 脖子较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T268 |
| 头最长。 | 1 | 0.333 | superlative:头 > 腿; superlative:头 > 尾巴 | S1T316 |
| 腿最长。 | 1 | 0.333 | superlative:腿 > 脖子; superlative:腿 > 尾巴 | S1T276 |
| 头较长，脖子一般，腿一般。 | 1 | 0.400 | absolute_long:头 > 0.50; absolute:脖子 middle_lower; absolute:腿 middle_lower | S1T256 |
| 脖子较长，头、腿一般。 | 1 | 0.400 | absolute_long:脖子 > 0.50; absolute:头 middle_lower; absolute:腿 middle_lower | S1T210 |
| 腿较长，脖子、头一般。 | 1 | 0.400 | absolute_long:腿 > 0.50; absolute:脖子 middle_lower; absolute:头 middle_lower | S1T216 |

### S110

- trial 数: 448; 非空文本: 447; fidelity 可评分率: 0.998; 平均 fidelity: 0.906; 完全忠实率: 0.732; 低 fidelity 率: 0.022.
- 旧版 region 覆盖率: 0.998; 旧版 region 有未处理片段率: 0.002.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 447 | 0.998 |
| comparison | 2 | 0.004 |
| empty | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子、腿长，头、尾巴短。 | 16 |
| 头、腿、尾巴长，脖子短。 | 16 |
| 头、脖子、腿长，尾巴短。 | 13 |
| 头、脖子、尾巴长，腿短。 | 12 |
| 头、腿长，脖子、尾巴短。 | 12 |
| 脖子、尾巴长，头、腿短。 | 9 |
| 脖子、腿、尾巴长，头短。 | 9 |
| 头、脖子长，腿、尾巴短。 | 9 |
| 头长，脖子、腿、尾巴短。 | 8 |
| 脖子长，头、腿、尾巴短。 | 7 |
| 头长，其余短。 | 7 |
| 头、尾巴长，脖子、腿短。 | 7 |
| 头短，脖子、尾巴长。 | 6 |
| 脖子长，其余短。 | 6 |
| 头、脖子、腿长。 | 6 |
| 头、尾巴长。 | 6 |
| 头短，腿长。 | 6 |
| 头短，尾巴长。 | 6 |
| 头、腿长。 | 6 |
| 头、腿、尾巴长。 | 6 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头、腿长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T42 |
| 头、腿、尾巴长，脖子短。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T183 |
| 头短，脖子、腿、尾巴中等偏长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T36 |
| 头短，脖子、腿、尾巴长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S2T36 |
| 脖子、腿长、尾巴、头短。 | 1 | 0.250 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S1T137 |
| 头、脖子、尾巴长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T64 |
| 头、腿、尾巴长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T296 |
| 头短，脖子、腿长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S2T82 |
| 头长，脖子、腿短。 | 1 | 0.333 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S2T99 |
| 脖子、腿长、尾巴短。 | 1 | 0.333 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S2T6 |

### S111

- trial 数: 320; 非空文本: 320; fidelity 可评分率: 1.000; 平均 fidelity: 0.937; 完全忠实率: 0.884; 低 fidelity 率: 0.022.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.003.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 320 | 1.000 |
| comparison | 78 | 0.244 |
| superlative | 3 | 0.009 |
| equality | 3 | 0.009 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴短。 | 37 |
| 尾巴长。 | 28 |
| 脖子比尾巴长，腿短。 | 11 |
| 尾巴短，腿短。 | 11 |
| 尾巴短，腿长。 | 9 |
| 尾巴长，腿短。 | 9 |
| 脖子比尾巴短，腿长。 | 9 |
| 腿短。 | 9 |
| 腿短，尾巴长。 | 8 |
| 尾巴长，头短。 | 7 |
| 脖子短。 | 7 |
| 脖子比尾巴长，腿长。 | 6 |
| 脖子比尾巴长。 | 6 |
| 头短。 | 5 |
| 脖子比尾巴短。 | 5 |
| 尾巴长，脖子短。 | 5 |
| 脖子长。 | 4 |
| 脖子比尾巴短，腿短。 | 4 |
| 腿短，尾巴短。 | 4 |
| 头比腿长。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位长度差不多。 | 1 | equality | S1T169 |
| 头和腿一样长，脖子和尾巴一样长。 | 1 | equality | S1T128 |
| 脖子比尾巴长，头和腿长度一样。 | 1 | equality | S1T100 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T169 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T153 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T244 |
| 尾巴较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T7 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T13 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T187 |
| 脖子比尾巴短，腿长，头长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S1T71 |

### S112

- trial 数: 384; 非空文本: 379; fidelity 可评分率: 0.974; 平均 fidelity: 0.903; 完全忠实率: 0.839; 低 fidelity 率: 0.073.
- 旧版 region 覆盖率: 0.974; 旧版 region 有未处理片段率: 0.016.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 374 | 0.974 |
| comparison | 302 | 0.786 |
| empty | 5 | 0.013 |
| equality | 5 | 0.013 |
| other | 4 | 0.010 |
| superlative | 2 | 0.005 |
| ranking | 2 | 0.005 |
| negation | 1 | 0.003 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴比腿长。 | 104 |
| 尾巴比腿短。 | 72 |
| 尾巴比较长。 | 33 |
| 尾巴比较短。 | 32 |
| 尾巴较长，腿较短。 | 11 |
| 尾巴比头长。 | 7 |
| 尾巴较短，腿较长。 | 7 |
| 尾巴比脖子长。 | 4 |
| 尾巴较长，脖子较长，腿较长。 | 4 |
| 头比脖子长。 | 4 |
| 尾巴较短。 | 4 |
| 头较长，尾巴较短。 | 3 |
| 头较长。 | 3 |
| 尾巴比腿长，尾巴比脖子长。 | 3 |
| 尾巴较长。 | 2 |
| 尾巴和头较短。 | 2 |
| 尾巴较长，脖子较长，腿较短。 | 2 |
| 尾巴非常长。 | 2 |
| 尾巴比头短。 | 2 |
| 尾巴比脖子短。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头和脖子一样长。 | 1 | equality | S1T27 |
| 尾巴和头之间有交集。 | 1 | other | S1T136 |
| 尾巴和腿之间关系。 | 1 | other | S1T29 |
| 尾巴比头和脖子间的交界要接近。 | 1 | equality | S1T137 |
| 尾巴比腿和脖子都长，尾巴第一长或者第二长。 | 1 | ranking | S1T33 |
| 尾巴比腿短，尾巴和脖子差不多长。 | 1 | equality | S1T38 |
| 尾巴跟脖子不一样长。 | 1 | equality, negation | S1T39 |
| 尾巴跟腿之间的关系以及跟脖子的关系。 | 1 | other | S1T37 |
| 尾巴跟腿差不多长，头比脖子短一点。 | 1 | equality | S1T20 |
| 脖子和腿之间的关系。 | 1 | other | S1T47 |
| 脖子最短，尾巴第二或者第三短，头比脖子长，尾巴比腿短。 | 1 | ranking | S1T31 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴比腿长。 | 19 | 0.000 | comparison:尾巴 > 腿 | S1T56, S1T57, S1T62, S1T104, S1T107, S1T108, S1T109, S1T116 |
| 头比尾巴长。 | 1 | 0.000 | comparison:头 > 尾巴 | S1T319 |
| 头比脖子长。 | 1 | 0.000 | comparison:头 > 脖子 | S1T66 |
| 尾巴、头较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50 | S1T295 |
| 尾巴比其它长。 | 1 | 0.000 | comparison:尾巴 > 脖子+头+腿 | S1T161 |
| 尾巴比头和脖子间的交界要接近。 | 1 | 0.000 | equality_range:尾巴+头+脖子 = | S1T137 |
| 尾巴比较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S2T54 |
| 尾巴较长，腿较长，尾巴比脖子短，脖子较长。 | 1 | 0.250 | absolute_long:尾巴 > 0.50; comparison:尾巴 < 脖子; absolute_long:脖子 > 0.50 | S1T218 |
| 尾巴较长，腿较短，尾巴比脖子长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; comparison:尾巴 > 脖子 | S1T226 |
| 尾巴较长，腿较短，脖子较长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S1T196 |

### S113

- trial 数: 384; 非空文本: 383; fidelity 可评分率: 0.987; 平均 fidelity: 0.925; 完全忠实率: 0.857; 低 fidelity 率: 0.047.
- 旧版 region 覆盖率: 0.987; 旧版 region 有未处理片段率: 0.068.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 375 | 0.977 |
| comparison | 216 | 0.562 |
| superlative | 41 | 0.107 |
| equality | 33 | 0.086 |
| ranking | 5 | 0.013 |
| other | 5 | 0.013 |
| body_ref | 2 | 0.005 |
| meta | 2 | 0.005 |
| empty | 1 | 0.003 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子长。 | 55 |
| 脖子短。 | 40 |
| 脖子很长。 | 11 |
| 脖子比腿短。 | 10 |
| 头比脖子短。 | 10 |
| 头比腿短很多。 | 9 |
| 头比脖子长。 | 7 |
| 脖子长，脖子比头短。 | 7 |
| 脖子足够长。 | 7 |
| 头比腿长很多。 | 7 |
| 脖子比腿长。 | 7 |
| 头比腿短一些。 | 6 |
| 头比脖子短很多。 | 5 |
| 脖子很短。 | 5 |
| 脖子比尾巴短。 | 5 |
| 头很长。 | 5 |
| 头比脖子短一些。 | 5 |
| 头比脖子长很多。 | 5 |
| 脖子和腿差不多长。 | 4 |
| 头和腿的长度差不多。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头和脖子的长度差不多。 | 4 | equality | S1T198, S1T199, S1T207, S1T208 |
| 头和腿差距不大。 | 4 | other | S1T157, S1T158, S1T159, S1T160 |
| 头和腿的长度差不多。 | 4 | equality | S1T178, S1T182, S1T184, S1T185 |
| 脖子和腿差不多长。 | 4 | equality | S1T210, S1T213, S1T216, S1T217 |
| 脖子和头的长度差不多。 | 3 | equality | S1T253, S1T254, S1T256 |
| 头和腿的长度差不多，头比腿长一些。 | 2 | equality | S1T177, S1T179 |
| 头比躯干短。 | 2 | body_ref | S1T133, S1T134 |
| 不知道。 | 1 | meta | S1T218 |
| 头和尾巴一样长。 | 1 | equality | S1T129 |
| 头和尾巴很长，脖子短一些，腿第二短，但也比较长。 | 1 | ranking | S1T24 |
| 头和脖子差不多长。 | 1 | equality | S1T91 |
| 头和脖子差不多长，腿比较长，尾巴短一些。 | 1 | equality | S1T84 |
| 头和脖子差不多长，都很长，尾巴和腿比较短。 | 1 | equality | S1T62 |
| 头和脖子差不多长，都比较长，腿和尾巴很短。 | 1 | equality | S1T80 |
| 头和腿一样长，头和腿比尾巴和脖子长。 | 1 | equality | S1T2 |
| 头和腿一样长，都比较长，脖子和尾巴比较短。 | 1 | equality | S1T4 |
| 头和腿差不多。 | 1 | equality | S1T166 |
| 头和腿的长度差不多，头比腿短一些。 | 1 | equality | S1T181 |
| 头最长，尾巴最短，脖子和腿差不多长，都比较长。 | 1 | equality | S1T13 |
| 头比脖子短很多，尾巴和头一样短，腿有点长。 | 1 | equality | S1T54 |
| 头比脖子长，头和腿差不多长，尾巴最短。 | 1 | equality | S1T74 |
| 头比脖子长，尾巴最短，腿和脖子差不多长。 | 1 | equality | S1T79 |
| 头比脖子长，尾巴第二长，腿最短。 | 1 | ranking | S1T48 |
| 尾巴最长，脖子第二长，头和腿比较短。 | 1 | ranking | S1T39 |
| 点错了。 | 1 | other | S2T32 |
| 脖子比头长，脖子最长，尾巴最短，腿和头差不多长。 | 1 | equality | S1T56 |
| 脖子短，但是和头的长度差不多。 | 1 | equality | S1T262 |
| 腿和尾巴都很长，腿最长，脖子第二短，头最短。 | 1 | ranking | S1T21 |
| 腿和脖子的长度差不多。 | 1 | equality | S1T194 |
| 腿最长，尾巴第二长，头最短，脖子有点短。 | 1 | ranking | S1T17 |
| 选错了。 | 1 | meta | S1T308 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子长。 | 9 | 0.000 | absolute_long:脖子 > 0.50 | S1T240, S1T278, S1T290, S1T291, S1T300, S1T316, S2T11, S2T17 |
| 头和腿差距不大。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50 | S1T157 |
| 头比尾巴长一点。 | 1 | 0.000 | comparison:头 > 尾巴 | S1T130 |
| 头比脖子长。 | 1 | 0.000 | comparison:头 > 脖子 | S1T106 |
| 头比腿长。 | 1 | 0.000 | comparison:头 > 腿 | S1T138 |
| 脖子和头的长度差不多。 | 1 | 0.000 | equality_range:脖子+头 = | S1T254 |
| 脖子足够长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T272 |
| 脖子长，比头长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T318 |
| 腿和脖子的长度差不多。 | 1 | 0.000 | equality_range:腿+脖子 = | S1T194 |
| 头和尾巴有些长，脖子比较短，尾巴适中。 | 1 | 0.400 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50; absolute:尾巴 middle_lower | S1T82 |

### S114

- trial 数: 512; 非空文本: 512; fidelity 可评分率: 0.994; 平均 fidelity: 0.918; 完全忠实率: 0.875; 低 fidelity 率: 0.066.
- 旧版 region 覆盖率: 0.994; 旧版 region 有未处理片段率: 0.006.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 512 | 1.000 |
| comparison | 203 | 0.396 |
| equality | 35 | 0.068 |
| superlative | 8 | 0.016 |
| negation | 1 | 0.002 |
| body_ref | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头比脖子长。 | 96 |
| 脖子比头长。 | 87 |
| 头很长。 | 16 |
| 头和脖子都很长。 | 16 |
| 四个部位都很短。 | 13 |
| 尾巴很短。 | 12 |
| 四个部位都差不多长。 | 10 |
| 腿特别短。 | 10 |
| 脖子长。 | 10 |
| 四个部位都很长。 | 8 |
| 头和脖子都很短。 | 8 |
| 腿很长。 | 8 |
| 腿很短。 | 8 |
| 头最长。 | 7 |
| 尾巴很长。 | 7 |
| 头比脖子短。 | 7 |
| 脖子很长。 | 7 |
| 头特别长。 | 6 |
| 头很短。 | 5 |
| 头长。 | 5 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位都差不多长。 | 10 | equality | S1T31, S1T33, S1T50, S1T102, S1T188, S1T189, S1T231, S1T243 |
| 头和脖子差不多长。 | 5 | equality | S1T291, S2T131, S2T135, S2T140, S2T146 |
| 头、脖子、尾巴差不多长。 | 4 | equality | S1T276, S2T27, S2T29, S2T78 |
| 头跟脖子差不多长。 | 3 | equality | S1T217, S1T304, S2T42 |
| 头、脖子、尾巴都差不多长。 | 2 | equality | S1T228, S1T229 |
| 头、脖子、腿、尾巴都差不多长。 | 2 | equality | S1T27, S1T29 |
| 脖子和头差不多长。 | 2 | equality | S2T70, S2T138 |
| 都差不多长。 | 2 | equality | S1T237, S2T73 |
| 四个部位差不多长。 | 1 | equality | S1T144 |
| 头和脖子一样长。 | 1 | equality | S2T126 |
| 头短于躯干。 | 1 | body_ref | S1T197 |
| 头跟脖子一样长。 | 1 | equality | S1T320 |
| 尾巴和腿差不多长。 | 1 | equality | S1T43 |
| 脖子和头一样长。 | 1 | equality | S2T118 |
| 都不长。 | 1 | negation | S1T96 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位都差不多长。 | 10 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T31, S1T33, S1T50, S1T102, S1T188, S1T189, S1T231, S1T243 |
| 头、脖子、尾巴差不多长。 | 3 | 0.000 | equality_range:头+脖子+尾巴 = | S1T276, S2T27, S2T29 |
| 头和脖子都很长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T92, S2T175, S2T188 |
| 头、脖子、尾巴都差不多长。 | 2 | 0.000 | equality_range:头+脖子+尾巴 = | S1T228, S1T229 |
| 头、脖子、腿、尾巴都差不多长。 | 2 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T27, S1T29 |
| 脖子和头差不多长。 | 2 | 0.000 | equality_range:脖子+头 = | S2T70, S2T138 |
| 四个部位差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T144 |
| 头和脖子差不多长。 | 1 | 0.000 | equality_range:头+脖子 = | S2T140 |
| 头很长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T211 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T206 |
| 尾巴和腿差不多长。 | 1 | 0.000 | equality_range:尾巴+腿 = | S1T43 |
| 尾巴很长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T214 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T173 |
| 脖子和头都很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S2T171 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T176 |
| 腿很短，其他都很长。 | 1 | 0.250 | complement:脖子 > 0.50; complement:头 > 0.50; complement:尾巴 > 0.50 | S1T155 |
| 头、尾巴长，脖子短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T4 |
| 头最长。 | 1 | 0.333 | superlative:头 > 腿; superlative:头 > 尾巴 | S2T33 |

### S115

- trial 数: 128; 非空文本: 128; fidelity 可评分率: 1.000; 平均 fidelity: 0.916; 完全忠实率: 0.883; 低 fidelity 率: 0.062.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 128 | 1.000 |
| comparison | 34 | 0.266 |
| count_abstract | 2 | 0.016 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿短。 | 31 |
| 腿长。 | 31 |
| 腿和脖子短。 | 6 |
| 腿和脖子长。 | 4 |
| 腿长，脖子短。 | 4 |
| 尾巴比其他部位短。 | 3 |
| 腿短，脖子长。 | 3 |
| 腿比脖子短。 | 3 |
| 腿和脖子中等。 | 2 |
| 腿长，脖子中等。 | 2 |
| 头长，腿比尾巴长。 | 2 |
| 腿中等，脖子长。 | 2 |
| 头比尾巴短。 | 2 |
| 腿比尾巴短。 | 2 |
| 头和尾巴长，脖子和腿短。 | 2 |
| 腿比其他三个部位短。 | 2 |
| 头比尾巴长。 | 2 |
| 腿比尾巴长。 | 2 |
| 腿比脖子长。 | 2 |
| 腿比脖子和尾巴短。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿比其他三个部位短。 | 2 | count_abstract | S1T31, S1T32 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短。 | 7 | 0.000 | absolute_short:腿 < 0.50 | S1T3, S1T55, S1T60, S1T61, S1T96, S1T110, S1T116 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T92 |

### S116

- trial 数: 128; 非空文本: 128; fidelity 可评分率: 0.984; 平均 fidelity: 0.894; 完全忠实率: 0.750; 低 fidelity 率: 0.031.
- 旧版 region 覆盖率: 0.984; 旧版 region 有未处理片段率: 0.039.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 128 | 1.000 |
| comparison | 27 | 0.211 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 只有头比较长。 | 3 |
| 腿和尾巴比较长。 | 3 |
| 腿太长了。 | 3 |
| 脖子很长。 | 3 |
| 腿和头很长。 | 2 |
| 只有尾巴长。 | 2 |
| 头、腿、尾巴长。 | 2 |
| 头和腿很长。 | 2 |
| 脖子特别长。 | 2 |
| 腿特别长。 | 2 |
| 腿和头长。 | 2 |
| 头和尾巴很长。 | 2 |
| 腿长。 | 2 |
| 腿、头、尾巴长。 | 2 |
| 头和脖子很长。 | 2 |
| 脖子非常长。 | 2 |
| 只有脖子非常长。 | 2 |
| 头和尾巴长。 | 2 |
| 头和腿长。 | 2 |
| 脖子太长。 | 2 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头和脖子长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T122 |
| 脖子、头和尾巴比较长一点。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T113 |
| 腿和头长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S1T123 |
| 腿、头、脖子都很长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S1T89 |

### S117

- trial 数: 128; 非空文本: 128; fidelity 可评分率: 1.000; 平均 fidelity: 0.988; 完全忠实率: 0.977; 低 fidelity 率: 0.008.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 128 | 1.000 |
| count_abstract | 8 | 0.062 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴较长。 | 60 |
| 尾巴较短。 | 52 |
| 尾巴和脖子较短。 | 3 |
| 除了脖子之外，其他三个部位较短。 | 2 |
| 四个部位都较短。 | 1 |
| 除了脖子，其他三个部位非常短。 | 1 |
| 除了头，其他三个部位都很长。 | 1 |
| 除了腿之外，其他三个部位较长。 | 1 |
| 除了尾巴之外，其他三个部位较长。 | 1 |
| 腿较短，其他三个部位一般长。 | 1 |
| 除了尾巴之外，其他三个部位都较长。 | 1 |
| 尾巴和脖子较长。 | 1 |
| 头和脖子较长。 | 1 |
| 脖子和尾巴较长。 | 1 |
| 脖子和腿较长。 | 1 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 除了脖子之外，其他三个部位较短。 | 2 | count_abstract | S1T5, S1T6 |
| 腿较短，其他三个部位一般长。 | 1 | count_abstract | S1T4 |
| 除了头，其他三个部位都很长。 | 1 | count_abstract | S1T1 |
| 除了尾巴之外，其他三个部位较长。 | 1 | count_abstract | S1T7 |
| 除了尾巴之外，其他三个部位都较长。 | 1 | count_abstract | S1T9 |
| 除了脖子，其他三个部位非常短。 | 1 | count_abstract | S1T2 |
| 除了腿之外，其他三个部位较长。 | 1 | count_abstract | S1T8 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T89 |

### S118

- trial 数: 256; 非空文本: 256; fidelity 可评分率: 0.801; 平均 fidelity: 0.915; 完全忠实率: 0.613; 低 fidelity 率: 0.031.
- 旧版 region 覆盖率: 0.801; 旧版 region 有未处理片段率: 0.254.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 254 | 0.992 |
| comparison | 123 | 0.480 |
| count_abstract | 50 | 0.195 |
| superlative | 35 | 0.137 |
| equality | 15 | 0.059 |
| ranking | 4 | 0.016 |
| negation | 3 | 0.012 |
| body_ref | 1 | 0.004 |
| other | 1 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴比较长。 | 51 |
| 尾巴比较短。 | 32 |
| 三个部位很长。 | 17 |
| 只有一个部位很长。 | 7 |
| 两个部位很长。 | 6 |
| 头比较长。 | 5 |
| 一个部位很长。 | 5 |
| 四个部位都很短。 | 4 |
| 三个部位都很长。 | 3 |
| 两个部位长，两个部位短。 | 3 |
| 只有两个部位很长。 | 3 |
| 头长，脖子长，腿短，尾巴短。 | 3 |
| 头短，脖子长，腿长，尾巴长。 | 2 |
| 头长，脖子长，尾巴长，腿短。 | 2 |
| 腿比较短。 | 2 |
| 四个部位都很长。 | 2 |
| 三个部位长。 | 2 |
| 头长，脖子短，尾巴短，腿短。 | 2 |
| 脖子比较长。 | 2 |
| 头长，脖子长，腿长，尾巴长，都很长。 | 1 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 三个部位很长。 | 17 | count_abstract | S1T123, S1T124, S1T125, S1T126, S1T134, S1T135, S1T136, S1T137 |
| 只有一个部位很长。 | 7 | count_abstract | S1T118, S1T129, S1T132, S1T133, S1T140, S1T141, S1T142 |
| 两个部位很长。 | 6 | count_abstract | S1T119, S1T120, S1T127, S1T128, S1T145, S1T146 |
| 一个部位很长。 | 5 | count_abstract | S1T147, S1T148, S1T150, S1T153, S1T155 |
| 三个部位都很长。 | 3 | count_abstract | S1T107, S1T117, S1T130 |
| 两个部位长，两个部位短。 | 3 | count_abstract | S1T109, S1T110, S1T112 |
| 只有两个部位很长。 | 3 | count_abstract | S1T131, S1T143, S1T159 |
| 三个部位长。 | 2 | count_abstract | S1T121, S1T122 |
| 三个部位都很短。 | 1 | count_abstract | S1T113 |
| 三个部位都很长，只有头是最短的。 | 1 | count_abstract | S1T108 |
| 三个部位长，一个部位短。 | 1 | count_abstract | S1T111 |
| 两个部位很长，两个部位很短。 | 1 | count_abstract | S1T114 |
| 头、脖子、尾巴、腿都很短，都差不多长度。 | 1 | equality | S1T52 |
| 头、脖子、尾巴一样长，腿中等长度。 | 1 | equality | S1T68 |
| 头和尾巴一样长，都很长，脖子稍微比头和尾巴短一点，尾巴比腿长，脖子和腿差不多长。 | 1 | equality | S1T13 |
| 头和尾巴差不多，脖子和腿差不多长，脖子和腿都很长。 | 1 | equality | S1T79 |
| 头和脖子都是它们长度范围的1/2，头和脖子一样长，尾巴也非常长，腿也比较长，但没有达到最大长度。 | 1 | equality, negation | S1T26 |
| 头和脖子都比较短，尾巴非常长，达到了最长长度，腿也非常长，接近于最长长度。 | 1 | equality | S1T16 |
| 头是最长的，脖子第二长，大概在最长长度的1/2，腿也是在它最大长度的1/2，尾巴比较短。 | 1 | ranking | S1T4 |
| 尾巴、腿、脖子、头都挺长的，都差不多长。 | 1 | equality | S1T32 |
| 尾巴比腿长，腿是最短的，头和脖子也比较短，脖子和头的长度应该差不多，尾巴比较长。 | 1 | equality | S1T5 |
| 按错了。 | 1 | other | S1T248 |
| 脖子和头比较短，头是最短的，脖子是第二短的，尾巴稍微长一些，但应该没有比腿长，腿是最长的。 | 1 | ranking, negation | S1T14 |
| 脖子和尾巴的长度差不多，头是最长的，腿的长度也很长，超过了它最大长度的1/2，尾巴比腿短。 | 1 | equality | S1T6 |
| 脖子最长，腿第二长，尾巴比较短，大概是它最长长度的1/3，头和尾巴差不多长。 | 1 | equality, ranking | S1T11 |
| 腿和头都非常长，达到了最大长度，脖子差不多是它长度范围的一半，尾巴非常短，是最短长度。 | 1 | equality, body_ref | S1T23 |
| 腿比较短，腿和脖子一样短，头和尾巴非常长，接近于它们自身最长长度，头和尾巴最长。 | 1 | equality | S1T12 |
| 腿长，尾巴长，脖子和头都比较长，但没有腿和尾巴那么长。 | 1 | negation | S1T30 |
| 腿非常短，是它的最短长度，尾巴和脖子的长度差不多，头是最长的。 | 1 | equality | S1T10 |
| 腿非常短，是它自身最短长度，脖子和尾巴的长度差不多，头比脖子和尾巴稍微长一些。 | 1 | equality | S1T20 |
| 腿非常短，是最短的部位，也是是它自身最短长度，尾巴是第二长的，头和脖子都比较长。 | 1 | ranking | S1T21 |
| 腿非常短，脖子很长，尾巴也非常长，头和腿一样短。 | 1 | equality | S1T27 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴比较长。 | 5 | 0.000 | absolute_long:尾巴 > 0.50 | S1T191, S1T195, S1T209, S1T223, S1T247 |
| 头和尾巴差不多，脖子和腿差不多长，脖子和腿都很长。 | 1 | 0.250 | equality_range:头+尾巴 =; equality_range:脖子+腿 =; absolute_long:脖子 > 0.50 | S1T79 |
| 头和尾巴一样长，都很长，脖子稍微比头和尾巴短一点，尾巴比腿长，脖子和腿差不多长。 | 1 | 0.333 | equality_range:头+尾巴 =; equality_range:脖子+腿 = | S1T13 |
| 头和脖子都很长，脖子很长，腿和尾巴都很短。 | 1 | 0.400 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T102 |

### S119

- trial 数: 768; 非空文本: 762; fidelity 可评分率: 0.725; 平均 fidelity: 0.925; 完全忠实率: 0.661; 低 fidelity 率: 0.048.
- 旧版 region 覆盖率: 0.725; 旧版 region 有未处理片段率: 0.267.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 755 | 0.983 |
| comparison | 556 | 0.724 |
| superlative | 80 | 0.104 |
| body_ref | 55 | 0.072 |
| count_abstract | 49 | 0.064 |
| negation | 26 | 0.034 |
| empty | 6 | 0.008 |
| equality | 2 | 0.003 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头比较长。 | 84 |
| 头短于中间值。 | 62 |
| 腿比较长。 | 58 |
| 头长于中间值。 | 53 |
| 头长。 | 40 |
| 尾巴比较长。 | 31 |
| 腿长。 | 30 |
| 头比腿长。 | 28 |
| 脖子比较长。 | 26 |
| 腿比头长。 | 21 |
| 脖子长。 | 18 |
| 尾巴长。 | 17 |
| 头最长。 | 17 |
| 头不是最长。 | 14 |
| 脖子最长。 | 14 |
| 有一个部位长于躯干。 | 13 |
| 尾巴最长。 | 11 |
| 腿最长。 | 10 |
| 腿长于头。 | 9 |
| 头长于腿。 | 9 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头不是最长。 | 14 | negation | S2T46, S2T47, S2T48, S2T164, S2T165, S2T166, S2T168, S2T169 |
| 有一个部位长于躯干。 | 13 | body_ref, count_abstract | S2T281, S2T284, S2T293, S2T295, S2T296, S2T298, S2T299, S2T301 |
| 有三个部位长于躯干。 | 7 | body_ref, count_abstract | S2T282, S2T283, S2T286, S2T288, S2T292, S2T300, S2T306 |
| 有两个部位长于躯干。 | 7 | body_ref, count_abstract | S2T280, S2T285, S2T290, S2T291, S2T303, S2T307, S2T310 |
| 腿不是最短。 | 5 | negation | S2T157, S2T158, S2T159, S2T161, S2T162 |
| 两个部位比中间值长。 | 4 | count_abstract | S2T149, S2T150, S2T151, S2T152 |
| 头长于躯干。 | 4 | body_ref | S2T313, S2T314, S2T315, S2T317 |
| 两个部位长于中间值。 | 3 | count_abstract | S2T100, S2T102, S2T103 |
| 有两个部位长于中间值。 | 3 | count_abstract | S2T97, S2T98, S2T268 |
| 没有部位长于躯干。 | 3 | body_ref, negation | S2T287, S2T294, S2T297 |
| 腿长于躯干。 | 3 | body_ref | S1T32, S1T33, S2T312 |
| 一个部位比中间值长。 | 2 | count_abstract | S2T153, S2T154 |
| 一个部位长于中间值。 | 2 | count_abstract | S2T99, S2T101 |
| 头低于躯干。 | 2 | body_ref | S2T190, S2T191 |
| 脖子、尾巴长于躯干。 | 2 | body_ref | S2T316, S2T319 |
| 三个部位长于中间值。 | 1 | count_abstract | S2T215 |
| 只有一个部位长于躯干。 | 1 | body_ref, count_abstract | S1T35 |
| 四个部位都短于躯干。 | 1 | body_ref | S1T31 |
| 大部分长于躯干。 | 1 | body_ref | S2T183 |
| 头和腿差不多长，尾巴和脖子差不多长。 | 1 | equality | S1T49 |
| 头没有长于腿。 | 1 | negation | S2T67 |
| 头短于躯干。 | 1 | body_ref | S2T185 |
| 头高于躯干。 | 1 | body_ref | S2T192 |
| 少于两个部位长于中间值。 | 1 | count_abstract | S2T269 |
| 尾巴和脖子不是最短的。 | 1 | negation | S2T193 |
| 有一个部位长于中间值。 | 1 | count_abstract | S2T267 |
| 有一个部位长长于躯干。 | 1 | body_ref, count_abstract | S2T289 |
| 有三个部位比较长。 | 1 | count_abstract | S2T96 |
| 脖子最长，脖子长于躯干。 | 1 | body_ref | S2T320 |
| 脖子没有长于尾巴。 | 1 | negation | S2T69 |
| 脖子长于躯干。 | 1 | body_ref | S2T311 |
| 腿、尾巴长于躯干。 | 1 | body_ref | S2T318 |
| 腿不是最长。 | 1 | negation | S2T111 |
| 腿短于躯干。 | 1 | body_ref | S2T184 |
| 超过两个等于躯干。 | 1 | equality, body_ref | S2T182 |
| 超过两个部位短于躯干。 | 1 | body_ref, count_abstract | S2T180 |
| 超过两个部位长于躯干。 | 1 | body_ref, count_abstract | S1T34 |
| 都短于躯干。 | 1 | body_ref | S2T181 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头比较长。 | 11 | 0.000 | absolute_long:头 > 0.50 | S1T131, S1T133, S1T155, S1T174, S1T184, S1T185, S1T211, S1T270 |
| 头长。 | 5 | 0.000 | absolute_long:头 > 0.50 | S1T320, S2T44, S2T55, S2T81, S2T121 |
| 头和尾巴比较长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T81, S1T84, S1T125 |
| 尾巴比较长。 | 2 | 0.000 | absolute_long:尾巴 > 0.50 | S1T223, S1T298 |
| 尾巴长。 | 2 | 0.000 | absolute_long:尾巴 > 0.50 | S2T60, S2T125 |
| 腿比较长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S1T75, S1T230 |
| 腿长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S2T94, S2T236 |
| 头很长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S2T212 |
| 头最长。 | 1 | 0.000 | superlative:头 > 脖子; superlative:头 > 腿; superlative:头 > 尾巴 | S2T80 |
| 头比腿短。 | 1 | 0.000 | comparison:头 < 腿 | S2T206 |
| 头比腿长。 | 1 | 0.000 | comparison:头 > 腿 | S2T30 |
| 头长于腿。 | 1 | 0.000 | comparison:头 > 腿 | S3T2 |
| 头高于躯干。 | 1 | 0.000 | body_ref:头 > 0.50 | S2T192 |
| 尾巴低于头和脖子的转折点。 | 1 | 0.000 | comparison:尾巴 < 头+脖子 | S2T189 |
| 脖子比较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T180 |
| 腿、尾巴长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S2T230 |
| 腿长于其他部位。 | 1 | 0.000 | comparison:腿 > 脖子+头+尾巴 | S1T86 |

### S120

- trial 数: 512; 非空文本: 511; fidelity 可评分率: 0.955; 平均 fidelity: 0.883; 完全忠实率: 0.756; 低 fidelity 率: 0.041.
- 旧版 region 覆盖率: 0.955; 旧版 region 有未处理片段率: 0.057.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 504 | 0.984 |
| comparison | 474 | 0.926 |
| body_ref | 344 | 0.672 |
| equality | 28 | 0.055 |
| superlative | 15 | 0.029 |
| count_abstract | 8 | 0.016 |
| group_sum | 4 | 0.008 |
| negation | 3 | 0.006 |
| other | 1 | 0.002 |
| empty | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头和腿都短于躯干。 | 93 |
| 头长于腿，头长于躯干。 | 62 |
| 头长于躯干。 | 47 |
| 头短于腿。 | 21 |
| 头和腿短于躯干。 | 15 |
| 头长于腿，头短于躯干。 | 14 |
| 腿长于躯干。 | 14 |
| 头和腿都长于躯干。 | 10 |
| 头短于躯干。 | 10 |
| 头长于腿，脖子长于尾巴。 | 9 |
| 头短于腿，腿长于躯干。 | 9 |
| 脖子最长。 | 5 |
| 头短于腿，脖子长于尾巴。 | 5 |
| 尾巴最长。 | 5 |
| 头长于脖子，尾巴长于腿。 | 4 |
| 头长于脖子，腿长于尾巴。 | 4 |
| 有奇数个部位长于躯干。 | 4 |
| 头长于尾巴，脖子长于腿。 | 4 |
| 脖子长于尾巴。 | 3 |
| 头最长。 | 3 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头和腿都短于躯干。 | 93 | body_ref | S1T251, S1T252, S1T253, S1T254, S1T255, S1T266, S1T272, S1T285 |
| 头长于腿，头长于躯干。 | 62 | body_ref | S1T198, S1T203, S1T206, S1T208, S1T210, S1T212, S1T213, S1T214 |
| 头长于躯干。 | 47 | body_ref | S1T149, S2T68, S2T69, S2T71, S2T74, S2T90, S2T92, S2T94 |
| 头和腿短于躯干。 | 15 | body_ref | S1T258, S1T259, S1T260, S1T273, S1T274, S1T275, S1T276, S1T277 |
| 头长于腿，头短于躯干。 | 14 | body_ref | S1T201, S1T202, S1T205, S1T209, S1T217, S1T224, S1T225, S1T229 |
| 腿长于躯干。 | 14 | body_ref | S1T146, S1T262, S1T313, S2T48, S2T65, S2T70, S2T72, S2T77 |
| 头和腿都长于躯干。 | 10 | body_ref | S2T25, S2T34, S2T41, S2T55, S2T80, S2T89, S2T119, S2T151 |
| 头短于躯干。 | 10 | body_ref | S2T98, S2T177, S2T178, S2T179, S2T180, S2T182, S2T183, S2T186 |
| 头短于腿，腿长于躯干。 | 9 | body_ref | S1T244, S1T245, S1T247, S1T250, S1T265, S1T282, S1T286, S2T6 |
| 有奇数个部位长于躯干。 | 4 | body_ref, count_abstract | S1T153, S1T154, S1T155, S1T159 |
| 只有腿长于躯干。 | 3 | body_ref | S2T133, S2T138, S2T146 |
| 有偶数个部位长于躯干。 | 3 | body_ref, count_abstract | S1T156, S1T157, S1T158 |
| 只有头比躯干长。 | 2 | body_ref | S1T81, S1T84 |
| 四个部位都短于躯干。 | 2 | body_ref | S1T96, S1T140 |
| 头和脖子的长度之和大于腿和尾巴的长度之和。 | 2 | group_sum | S1T60, S1T61 |
| 头和脖子长于躯干，尾巴和腿短于躯干。 | 2 | body_ref | S1T100, S1T101 |
| 头和腿长于躯干，脖子和尾巴短于躯干。 | 2 | body_ref | S1T105, S1T139 |
| 头短于腿，头短于躯干。 | 2 | body_ref | S1T211, S1T264 |
| 头短于腿，脖子和尾巴差不多。 | 2 | equality | S1T167, S1T177 |
| 头短于躯干，腿长于躯干。 | 2 | body_ref | S2T164, S2T172 |
| 头长于腿，腿长于躯干。 | 2 | body_ref | S2T20, S2T54 |
| 脖子和尾巴的长度不一样。 | 2 | equality, negation | S1T42, S1T43 |
| 脖子和尾巴长于躯干。 | 2 | body_ref | S1T106, S1T151 |
| 五个部位的长度都差不多。 | 1 | equality | S1T25 |
| 其他部位差不多，脖子比较短。 | 1 | equality | S1T32 |
| 只有头比躯干短。 | 1 | body_ref | S1T83 |
| 只有头长于躯干。 | 1 | body_ref | S1T142 |
| 只有尾巴比躯干长。 | 1 | body_ref | S1T85 |
| 只有尾巴长于躯干。 | 1 | body_ref | S1T94 |
| 只有脖子和尾巴比躯干长。 | 1 | body_ref | S1T86 |
| 只有脖子比较短，其他四个部位长度差不多。 | 1 | equality | S1T50 |
| 只有脖子短于躯干。 | 1 | body_ref | S1T144 |
| 只有脖子长于躯干，其他都比躯干短。 | 1 | body_ref | S1T97 |
| 只有腿比躯干长。 | 1 | body_ref | S1T88 |
| 头、尾巴、腿差不多，脖子长于尾巴。 | 1 | equality | S1T196 |
| 头、尾巴和腿长于躯干。 | 1 | body_ref | S1T93 |
| 头、脖子长于躯干。 | 1 | body_ref | S1T92 |
| 头与腿相等。 | 1 | equality | S2T78 |
| 头和尾巴一样长，脖子长于腿。 | 1 | equality | S1T73 |
| 头和尾巴加起来短于脖子和躯干，也短于脖子和腿。 | 1 | body_ref, group_sum | S1T58 |
| 头和尾巴差不多，腿和脖子差不多。 | 1 | equality | S1T54 |
| 头和尾巴比躯干长。 | 1 | body_ref | S1T82 |
| 头和尾巴都非常长，长于躯干。 | 1 | body_ref | S1T4 |
| 头和尾巴长于躯干。 | 1 | body_ref | S1T103 |
| 头和脖子差不多。 | 1 | equality | S1T39 |
| 头和脖子比躯干长。 | 1 | body_ref | S1T87 |
| 头和脖子短于尾巴和躯干和腿。 | 1 | body_ref | S1T20 |
| 头和腿差不多，尾巴和脖子差不多。 | 1 | equality | S1T193 |
| 头和腿是短于躯干的。 | 1 | body_ref | S2T171 |
| 头和腿短于躯干，脖子和尾巴长于躯干。 | 1 | body_ref | S1T143 |
| 头和腿短于躯干，脖子长于躯干。 | 1 | body_ref | S1T141 |
| 头和腿长度差不多，脖子和尾巴长度差不多。 | 1 | equality | S1T64 |
| 头比尾巴短很多，脖子和腿差不多。 | 1 | equality | S1T76 |
| 头比腿短，长于躯干。 | 1 | body_ref | S2T1 |
| 头比较长，和躯干差不多。 | 1 | equality, body_ref | S1T5 |
| 头短于腿，但是长于躯干。 | 1 | body_ref | S2T16 |
| 头短于腿，头和腿都长于躯干。 | 1 | body_ref | S1T296 |
| 头短于腿，尾巴脖子差不多。 | 1 | equality | S1T192 |
| 头短于腿，短于躯干。 | 1 | body_ref | S2T12 |
| 头短于腿，脖子尾巴差不多。 | 1 | equality | S1T164 |
| 头短于腿，腿比躯干长。 | 1 | body_ref | S2T8 |
| 头短于腿，腿短于躯干。 | 1 | body_ref | S2T13 |
| 头短于腿，都短于躯干。 | 1 | body_ref | S1T204 |
| 头长与腿，脖子和尾巴差不多。 | 1 | equality | S1T165 |
| 头长于脖子，其他差不多。 | 1 | equality | S1T125 |
| 头长于腿，尾巴长于躯干。 | 1 | body_ref | S1T199 |
| 头长于腿，腿短于躯干。 | 1 | body_ref | S1T242 |
| 头长于躯干，尾巴长于躯干。 | 1 | body_ref | S1T150 |
| 头非常长，长于腿和躯干。 | 1 | body_ref | S1T10 |
| 尾巴和脖子长于躯干。 | 1 | body_ref | S1T138 |
| 尾巴和脖子长度不一样。 | 1 | equality, negation | S1T41 |
| 尾巴明显长于躯干，头和脖子差不多。 | 1 | equality, body_ref | S1T104 |
| 尾巴长于躯干。 | 1 | body_ref | S1T147 |
| 点错了。 | 1 | other | S1T310 |
| 脖子和头长于躯干。 | 1 | body_ref | S1T98 |
| 脖子和尾巴差不多。 | 1 | equality | S1T102 |
| 脖子和尾巴的长度差不多。 | 1 | equality | S1T44 |
| 脖子和腿长于躯干，有偶数个部位长于躯干。 | 1 | body_ref, count_abstract | S1T152 |
| 脖子比较长，跟腿差不多，也跟躯干差不多。 | 1 | equality, body_ref | S1T9 |
| 脖子长于其他部位，其他部位都差不多短。 | 1 | equality | S1T53 |
| 脖子长于躯干。 | 1 | body_ref | S1T145 |
| 脖子长于躯干，腿长于尾巴。 | 1 | body_ref | S1T99 |
| 脖子非常长，跟躯干差不多。 | 1 | equality, body_ref | S1T2 |
| 腿、脖子和尾巴长于躯干。 | 1 | body_ref | S1T91 |
| 腿和脖子加起来跟头差不多。 | 1 | equality, group_sum | S1T57 |
| 腿短于躯干，头比较长。 | 1 | body_ref | S1T3 |
| 腿长于躯干，脖子和尾巴短。 | 1 | body_ref | S1T148 |
| 腿非常长，长于躯干。 | 1 | body_ref | S1T1 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头和腿都短于躯干。 | 7 | 0.000 | body_ref:头 < 0.50; body_ref:腿 < 0.50 | S1T252, S1T307, S1T308, S1T318, S2T39, S2T103, S2T114 |
| 头与腿相等。 | 1 | 0.000 | equality_range:头+腿 = | S2T78 |
| 头和脖子差不多。 | 1 | 0.000 | equality_range:头+脖子 = | S1T39 |
| 头和腿短于躯干。 | 1 | 0.000 | body_ref:头 < 0.50; body_ref:腿 < 0.50 | S2T60 |
| 头比脖子短。 | 1 | 0.000 | comparison:头 < 脖子 | S1T77 |
| 头短于其他部位。 | 1 | 0.000 | comparison:头 < 脖子+腿+尾巴 | S1T112 |
| 头短于躯干。 | 1 | 0.000 | body_ref:头 < 0.50 | S2T98 |
| 头长于腿，头短于躯干。 | 1 | 0.000 | comparison:头 > 腿; body_ref:头 < 0.50 | S1T234 |
| 脖子比较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T11 |
| 脖子比较长，跟腿差不多，也跟躯干差不多。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T9 |
| 腿和脖子加起来跟头差不多。 | 1 | 0.000 | equality_range:腿+脖子+头 = | S1T57 |
| 腿长于躯干。 | 1 | 0.000 | body_ref:腿 > 0.50 | S1T313 |
| 脖子长于其他部位，其他部位都差不多短。 | 1 | 0.250 | complement:头 > 0.50; complement:腿 > 0.50; complement:尾巴 > 0.50 | S1T53 |
| 只有头比躯干长。 | 1 | 0.400 | exclusive_case:脖子 < 0.50; exclusive_case:腿 < 0.50; exclusive_case:尾巴 < 0.50 | S1T81 |
| 头和脖子短于尾巴和躯干和腿。 | 1 | 0.400 | body_ref:脖子 < 0.50; body_ref:尾巴 < 0.50; body_ref:腿 < 0.50 | S1T20 |

### S121

- trial 数: 448; 非空文本: 446; fidelity 可评分率: 0.971; 平均 fidelity: 0.910; 完全忠实率: 0.799; 低 fidelity 率: 0.040.
- 旧版 region 覆盖率: 0.971; 旧版 region 有未处理片段率: 0.060.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 439 | 0.980 |
| comparison | 306 | 0.683 |
| equality | 42 | 0.094 |
| group_sum | 36 | 0.080 |
| superlative | 19 | 0.042 |
| body_ref | 14 | 0.031 |
| negation | 8 | 0.018 |
| empty | 2 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿比较长。 | 39 |
| 腿比较短。 | 33 |
| 腿较长。 | 15 |
| 腿较短。 | 12 |
| 头和尾巴的长度总和大于脖子和腿的长度总和。 | 11 |
| 头比脖子长，腿比尾巴长。 | 7 |
| 头比脖子短，腿比尾巴长。 | 7 |
| 头和脖子的长度总和小于腿和尾巴的长度总和。 | 5 |
| 头比脖子短，腿比尾巴短。 | 5 |
| 头比脖子长，腿比尾巴短。 | 5 |
| 头比脖子短，腿较长。 | 5 |
| 脖子和腿的长度之和大于腿的长度。 | 4 |
| 腿不是所有部位里最短的。 | 4 |
| 头比脖子长，腿较长，尾巴较短。 | 4 |
| 头和脖子差不多，腿比尾巴长。 | 4 |
| 头、脖子、尾巴较长，腿较短。 | 3 |
| 头比脖子短，尾巴很短。 | 3 |
| 头比脖子短，腿很短。 | 3 |
| 头比脖子短，尾巴和腿较长。 | 3 |
| 头比脖子短，腿较短。 | 3 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头和尾巴的长度总和大于脖子和腿的长度总和。 | 11 | group_sum | S1T259, S1T260, S1T261, S1T262, S1T263, S1T266, S1T268, S1T270 |
| 头和脖子的长度总和小于腿和尾巴的长度总和。 | 5 | group_sum | S1T251, S1T252, S1T253, S1T254, S1T256 |
| 头和脖子差不多，腿比尾巴长。 | 4 | equality | S1T165, S1T176, S1T186, S1T187 |
| 脖子和腿的长度之和大于腿的长度。 | 4 | group_sum | S1T307, S1T308, S1T309, S1T310 |
| 腿不是所有部位里最短的。 | 4 | negation | S2T30, S2T34, S2T35, S2T36 |
| 头和脖子的长度总和大于腿和尾巴的长度总和。 | 3 | group_sum | S1T255, S1T257, S1T258 |
| 头和腿的长度总和小于脖子和尾巴的长度总和。 | 3 | group_sum | S1T264, S1T265, S1T267 |
| 头在躯干上方，腿比较长。 | 3 | body_ref | S1T291, S1T294, S1T296 |
| 头比脖子短，腿和尾巴差不多。 | 3 | equality | S1T164, S1T166, S1T248 |
| 头不是所有部位里最短的。 | 2 | negation | S2T38, S2T39 |
| 头和尾巴的长度总和小于脖子和腿的长度总和。 | 2 | group_sum | S1T269, S1T274 |
| 头和脖子差不多，腿很长，尾巴很短。 | 2 | equality | S1T175, S1T188 |
| 头和脖子差不多，腿较长，尾巴较短。 | 2 | equality | S1T99, S1T109 |
| 头脖子腿都差不多，尾巴较短。 | 2 | equality | S1T58, S1T59 |
| 脖子和头的长度总和大于腿和尾巴的长度总和。 | 2 | group_sum | S1T279, S1T281 |
| 脖子和头的长度总和小于腿和尾巴的长度总和。 | 2 | group_sum | S1T278, S1T280 |
| 腿不是最短的部位。 | 2 | negation | S2T56, S2T57 |
| 头、尾巴和脖子都差不多，腿比较短。 | 1 | equality | S1T167 |
| 头、脖子、尾巴差不多，腿较短。 | 1 | equality | S1T102 |
| 头、脖子、腿和尾巴一样长。 | 1 | equality | S1T287 |
| 头和尾巴差不多。 | 1 | equality | S1T228 |
| 头和尾巴差不多一样长。 | 1 | equality | S1T33 |
| 头和尾巴差不多长，脖子较短，腿较长。 | 1 | equality | S1T32 |
| 头和尾巴差不多，腿比尾巴长。 | 1 | equality | S1T185 |
| 头和脖子差不多长，尾巴较短，腿较长。 | 1 | equality | S1T85 |
| 头和脖子差不多长，腿和尾巴较长。 | 1 | equality | S1T84 |
| 头和脖子差不多，尾巴较长。 | 1 | equality | S1T139 |
| 头和脖子差不多，尾巴较长，腿很短。 | 1 | equality | S1T138 |
| 头和脖子差不多，腿和尾巴较短。 | 1 | equality | S1T199 |
| 头和脖子差不多，腿较短。 | 1 | equality | S1T120 |
| 头和脖子差不多，腿较长。 | 1 | equality | S1T92 |
| 头和脖子差不多，都比较长，腿较短。 | 1 | equality | S1T237 |
| 头和脖子的长度总和大于腿的长度。 | 1 | group_sum | S1T302 |
| 头和脖子的长度总和小于腿和尾巴长度的总和。 | 1 | group_sum | S1T250 |
| 头和脖子的长度总和小于腿的长度。 | 1 | group_sum | S1T301 |
| 头和腿差不多。 | 1 | equality | S1T45 |
| 头在躯干上方，腿比较短。 | 1 | body_ref | S1T293 |
| 头在躯干上方，腿比较长，尾巴很短。 | 1 | body_ref | S1T295 |
| 头在躯干上方，腿较短。 | 1 | body_ref | S1T299 |
| 头在躯干上方，腿较长。 | 1 | body_ref | S1T298 |
| 头在躯干下方。 | 1 | body_ref | S1T246 |
| 头在躯干下方，腿比较短。 | 1 | body_ref | S1T292 |
| 头在躯干下方，腿较短。 | 1 | body_ref | S1T297 |
| 头在躯干下方，腿较长。 | 1 | body_ref | S1T300 |
| 头在躯干的上方。 | 1 | body_ref | S1T289 |
| 头比脖子短一点，腿和尾巴差不多。 | 1 | equality | S1T153 |
| 头比脖子短，腿和尾巴差不多，都很长。 | 1 | equality | S1T207 |
| 头比脖子短，腿比尾巴，差不多。 | 1 | equality | S1T212 |
| 头比脖子长一点。腿和尾巴差不多长。 | 1 | equality | S1T241 |
| 头比脖子长一点。腿和尾巴差不多，都很短。 | 1 | equality | S1T145 |
| 头比脖子长，尾巴和腿差不多。 | 1 | equality | S1T88 |
| 头比脖子长，腿和尾巴差不多。 | 1 | equality | S1T226 |
| 头脖子和尾巴差不多，腿较短。 | 1 | equality | S1T61 |
| 头脖子，尾巴，头脖子腿差不多，尾巴较短。 | 1 | equality | S1T140 |
| 脖子和头的总宽度大于躯干和尾巴的总宽度。 | 1 | body_ref | S1T282 |
| 脖子和头的总宽度小于躯干和尾巴的总宽度。 | 1 | body_ref | S1T283 |
| 脖子和头的长度之和大于腿的长度。 | 1 | group_sum | S1T305 |
| 脖子和腿差不多，头较长，尾巴较短。 | 1 | equality | S1T28 |
| 脖子和腿差不多，尾巴较短，头较短。 | 1 | equality | S1T60 |
| 脖子和腿差不多，尾巴较长。 | 1 | equality | S1T25 |
| 腿较短，尾巴、头、脖子差不多。 | 1 | equality | S1T121 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿较长。 | 3 | 0.000 | absolute_long:腿 > 0.50 | S1T318, S2T25, S2T27 |
| 头、尾巴和脖子都差不多，腿比较短。 | 1 | 0.000 | equality_range:头+尾巴+脖子 =; absolute_short:腿 < 0.50 | S1T167 |
| 头、脖子、腿和尾巴一样长。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T287 |
| 头和尾巴差不多一样长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T33 |
| 头和尾巴差不多，腿比尾巴长。 | 1 | 0.000 | equality_range:头+尾巴 =; comparison:腿 > 尾巴 | S1T185 |
| 头和尾巴的长度总和大于脖子和腿的长度总和。 | 1 | 0.000 | group_sum:头+尾巴 > 脖子+腿 | S1T273 |
| 头和脖子差不多，腿较长。 | 1 | 0.000 | equality_range:头+脖子 =; absolute_long:腿 > 0.50 | S1T92 |
| 头脖子腿都差不多，尾巴较短。 | 1 | 0.000 | equality_range:头+脖子+腿 =; absolute_short:尾巴 < 0.50 | S1T59 |
| 头脖子，尾巴，头脖子腿差不多，尾巴较短。 | 1 | 0.000 | equality_range:头+脖子+腿 =; absolute_short:尾巴 < 0.50 | S1T140 |
| 尾巴相对于头、脖子、腿来说较短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50; absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T37 |
| 脖子较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T306 |
| 腿较短，是所有部位里最短的。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50; absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S2T28 |
| 头比脖子短，腿和尾巴较长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T100 |
| 头比脖子长一点，腿较长，尾巴较长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T224 |
| 脖子和腿差不多，头较长，尾巴较短。 | 1 | 0.333 | equality_range:脖子+腿 =; absolute_short:尾巴 < 0.50 | S1T28 |
| 腿比所有其他部位短。 | 1 | 0.333 | complement:脖子 < 0.50; complement:尾巴 < 0.50 | S2T29 |

### S122

- trial 数: 256; 非空文本: 256; fidelity 可评分率: 1.000; 平均 fidelity: 0.766; 完全忠实率: 0.559; 低 fidelity 率: 0.148.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 256 | 1.000 |
| comparison | 75 | 0.293 |
| group_sum | 75 | 0.293 |
| body_ref | 1 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头、脖子、尾巴之和比四条腿之和短。 | 40 |
| 头、脖子、尾巴之和比四条腿之和长。 | 34 |
| 头短，脖子短，尾巴短，腿长。 | 25 |
| 头长，脖子长，尾巴短，腿短。 | 19 |
| 头长，脖子长，尾巴长，腿短。 | 12 |
| 头短，脖子短，尾巴长，腿长。 | 11 |
| 头长，尾巴长，脖子短，腿短。 | 8 |
| 头长，脖子长，尾巴短，腿长。 | 8 |
| 头短，尾巴短，脖子长，腿长。 | 7 |
| 头长，脖子短，尾巴短，腿短。 | 6 |
| 头长，腿长，脖子短，尾巴短。 | 5 |
| 头长，脖子短，尾巴短，腿长。 | 5 |
| 头短，脖子短，尾巴长，腿短。 | 5 |
| 头长，脖子长，腿长，尾巴短。 | 4 |
| 头短，脖子长，尾巴长，腿长。 | 4 |
| 头短，脖子短，尾巴短，腿短。 | 4 |
| 头长，脖子短，尾巴长，腿短。 | 4 |
| 头长，脖子长，尾巴长，腿长。 | 4 |
| 头长，脖子短，尾巴长，腿长。 | 3 |
| 头短，尾巴短，脖子长，腿短。 | 3 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头、脖子、尾巴之和比四条腿之和短。 | 40 | group_sum | S1T185, S1T186, S1T187, S1T188, S1T190, S1T191, S1T192, S1T194 |
| 头、脖子、尾巴之和比四条腿之和长。 | 34 | group_sum | S1T182, S1T183, S1T184, S1T189, S1T193, S1T196, S1T198, S1T199 |
| 头、脖子、尾巴之和比躯干短。 | 1 | body_ref, group_sum | S1T227 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、尾巴之和比四条腿之和短。 | 37 | 0.000 | group_sum:头+脖子+尾巴 < 腿 | S1T185, S1T186, S1T187, S1T188, S1T190, S1T191, S1T192, S1T194 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T2 |

### S123

- trial 数: 512; 非空文本: 510; fidelity 可评分率: 0.994; 平均 fidelity: 0.908; 完全忠实率: 0.781; 低 fidelity 率: 0.029.
- 旧版 region 覆盖率: 0.994; 旧版 region 有未处理片段率: 0.014.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 509 | 0.994 |
| superlative | 75 | 0.146 |
| comparison | 71 | 0.139 |
| equality | 15 | 0.029 |
| ranking | 3 | 0.006 |
| empty | 2 | 0.004 |
| body_ref | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头和腿长。 | 48 |
| 头长。 | 22 |
| 尾巴长。 | 20 |
| 脖子长。 | 18 |
| 脖子长，头长于尾巴。 | 17 |
| 脖子和尾巴长。 | 17 |
| 尾巴最长。 | 12 |
| 头最长。 | 12 |
| 脖子长，尾巴长于头。 | 12 |
| 脖子和腿长。 | 10 |
| 头、脖子、尾巴长。 | 9 |
| 腿长。 | 9 |
| 头和尾巴长。 | 9 |
| 头和脖子长。 | 8 |
| 脖子和头长。 | 7 |
| 脖子最长，头长于尾巴。 | 7 |
| 脖子、腿、尾巴长。 | 6 |
| 头、脖子、腿长。 | 6 |
| 头、脖子、腿、尾巴短。 | 6 |
| 腿和头短，脖子和尾巴长。 | 6 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿长，头和尾巴差不多长。 | 3 | equality | S1T171, S1T178, S1T265 |
| 头、脖子、腿、尾巴、躯干这些部位的长度进行描述，但不一定用的是这几个词语，且可能会涉及到这几个部位之间的比较，包括大小和长短关系。 | 1 | body_ref | S1T73 |
| 头、脖子、腿、尾巴差不多。 | 1 | equality | S2T178 |
| 头、腿、尾巴差不多长，脖子最短。 | 1 | equality | S1T299 |
| 头最长，脖子和腿差不多长，尾巴最短。 | 1 | equality | S1T70 |
| 头最长，脖子最短，尾巴最短，脖子和腿差不多。 | 1 | equality | S1T67 |
| 头最长，腿和脖子短，尾巴第二长。 | 1 | ranking | S1T74 |
| 头短，腿和脖子和尾巴差不多。 | 1 | equality | S1T9 |
| 头长，腿、脖子、尾巴差不多长。 | 1 | equality | S1T124 |
| 尾巴长，然后头短，腿和脖子都偏短。 | 1 | ranking | S1T32 |
| 脖子最长，腿第二长，头和尾巴都短。 | 1 | ranking | S1T46 |
| 脖子长腿长，头和尾巴差不多长。 | 1 | equality | S1T180 |
| 脖子长，头和尾巴差不多长。 | 1 | equality | S1T183 |
| 脖子长，尾巴和头差不多长。 | 1 | equality | S1T234 |
| 腿短，头长，脖子和尾巴差不多长。 | 1 | equality | S1T75 |
| 腿长脖子短，头和尾巴差不多。 | 1 | equality | S1T169 |
| 腿长，脖子短，头和尾巴差不多长。 | 1 | equality | S1T88 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头和腿长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T208, S1T290, S2T73 |
| 脖子长。 | 2 | 0.000 | absolute_long:脖子 > 0.50 | S1T155, S2T66 |
| 头、脖子、尾巴长。 | 2 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T176, S2T183 |
| 头、脖子、腿、尾巴差不多。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S2T178 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T235 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S2T7 |
| 脖子和尾巴长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T29 |
| 头、尾巴和腿长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T104 |
| 头、腿、尾巴长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T184 |
| 腿短，脖子和头偏长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T106 |
| 腿长脖子短，头和尾巴差不多。 | 1 | 0.333 | absolute_short:腿 < 0.50; equality_range:头+尾巴 = | S1T169 |

### S124

- trial 数: 128; 非空文本: 128; fidelity 可评分率: 1.000; 平均 fidelity: 0.943; 完全忠实率: 0.859; 低 fidelity 率: 0.023.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 128 | 1.000 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴较长。 | 24 |
| 尾巴较短。 | 16 |
| 头较长，尾巴较短。 | 6 |
| 头较长，腿中等。 | 4 |
| 头较短，腿中等。 | 3 |
| 头较短，脖子较长，腿较长，尾巴较短。 | 3 |
| 头较短，尾巴中等。 | 3 |
| 尾巴特别长。 | 3 |
| 脖子较长，腿中等。 | 2 |
| 脖子较短，腿中等。 | 2 |
| 头较长，脖子中等，腿较长，尾巴中等。 | 2 |
| 头中等，脖子较短，腿中等，尾巴较长。 | 2 |
| 头较长，脖子中等，腿较长，尾巴较短。 | 2 |
| 头较长，尾巴中等。 | 2 |
| 头较短，尾巴较长。 | 2 |
| 脖子较短，腿较短。 | 2 |
| 尾巴中等。 | 2 |
| 头中等，腿较长。 | 2 |
| 头较长，脖子中等，腿中等，尾巴较短。 | 1 |
| 头较短，脖子较长，腿特别短，尾巴中等。 | 1 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴较长。 | 2 | 0.000 | absolute_long:尾巴 > 0.50 | S1T108, S1T113 |
| 头较长，腿中等。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute:腿 middle_lower | S1T36 |

### S125

- trial 数: 128; 非空文本: 128; fidelity 可评分率: 1.000; 平均 fidelity: 0.976; 完全忠实率: 0.906; 低 fidelity 率: 0.000.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.125.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 128 | 1.000 |
| comparison | 62 | 0.484 |
| superlative | 7 | 0.055 |
| equality | 6 | 0.047 |
| group_sum | 1 | 0.008 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿很长。 | 15 |
| 腿比较短。 | 13 |
| 腿很短。 | 13 |
| 腿长。 | 5 |
| 腿比较长。 | 4 |
| 脖子比腿长。 | 2 |
| 头长，脖子长，腿长，尾巴短。 | 2 |
| 腿短。 | 2 |
| 腿长度适中，腿比脖子短。 | 2 |
| 腿很长，基本是最长。 | 2 |
| 腿非常短，几乎是最短。 | 2 |
| 腿很长，头和脖子比较长。 | 1 |
| 腿比较短，脖子和头都很长。 | 1 |
| 头长度适中，脖子短，腿短，尾巴较短，整体看起来比较短小。 | 1 |
| 头短，脖子较长，腿较长，尾巴较短，整体看起来比较修长。 | 1 |
| 头和尾巴比较短，脖子较长，腿较长，脖子比腿要长一点。 | 1 |
| 头长，脖子短，腿短，尾巴短，脖子和腿的长度差不多，头比其他的部位要长很多。 | 1 |
| 腿短，脖子较长。 | 1 |
| 腿很长，腿比脖子长。 | 1 |
| 头长，脖子长度适中，腿长度适中，尾巴短。 | 1 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头、脖子、腿都比较长，尾巴较短，脖子和腿差不多长。 | 1 | equality | S1T28 |
| 头短，脖子长度适中，腿长度适中，尾巴长度适中，脖子、腿和尾巴的长度差不多。 | 1 | equality | S1T15 |
| 头长，脖子短，腿短，尾巴短，脖子和腿的长度差不多，头比其他的部位要长很多。 | 1 | equality | S1T48 |
| 头长，脖子长，腿长，尾巴较短，脖子和腿长度差不多。 | 1 | equality | S1T33 |
| 脖子和腿都比较长，脖子和腿差不多长，头比较短。 | 1 | equality | S1T32 |
| 腿和脖子差不多长，头部较长。 | 1 | equality | S1T26 |
| 腿长度适中，头和脖子比较长，尾巴长度适中，在整体比例中腿显得比较短。 | 1 | group_sum | S1T58 |

低忠实率对应试次（fidelity < 0.5）：
无。

### S126

- trial 数: 128; 非空文本: 128; fidelity 可评分率: 1.000; 平均 fidelity: 0.901; 完全忠实率: 0.688; 低 fidelity 率: 0.008.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 128 | 1.000 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头很长，脖子较长。 | 4 |
| 头较短，脖子很长。 | 4 |
| 头很长，脖子很长。 | 4 |
| 头较短，脖子较长。 | 3 |
| 头较短，脖子很短。 | 3 |
| 头很短，脖子较长。 | 3 |
| 头很长，脖子较长，腿很短，尾巴较长。 | 2 |
| 头很长，脖子较长，腿很短，尾巴较短。 | 2 |
| 头和脖子较短，腿较长，尾巴较短。 | 2 |
| 头很长，脖子很短，腿较长，尾巴较长。 | 2 |
| 头很长，脖子较短，腿很长，尾巴较短。 | 2 |
| 头较短，脖子很长，腿较长，尾巴较长。 | 2 |
| 头较长，脖子较长，腿较短，尾巴较短。 | 2 |
| 头很长，脖子较短，腿较长，尾巴较长。 | 2 |
| 头很长，脖子很短。 | 2 |
| 头很长，脖子很长，腿很短，尾巴较短。 | 1 |
| 头很长，脖子很短，腿很长，尾巴较短。 | 1 |
| 头较短，脖子较长，腿很长，尾巴很长。 | 1 |
| 头很长，脖子很长，腿较长，尾巴较长。 | 1 |
| 头较短，脖子很长，腿较短，尾巴较短。 | 1 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头和脖子较长，腿较长，尾巴中等。 | 1 | 0.400 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute:尾巴 middle_upper | S1T4 |

### S127

- trial 数: 256; 非空文本: 256; fidelity 可评分率: 0.996; 平均 fidelity: 0.952; 完全忠实率: 0.906; 低 fidelity 率: 0.023.
- 旧版 region 覆盖率: 0.996; 旧版 region 有未处理片段率: 0.008.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 256 | 1.000 |
| superlative | 64 | 0.250 |
| comparison | 59 | 0.230 |
| equality | 9 | 0.035 |
| ranking | 1 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子短。 | 52 |
| 脖子长。 | 46 |
| 脖子比较长。 | 20 |
| 脖子比较短。 | 13 |
| 脖子最短。 | 8 |
| 腿最短。 | 7 |
| 脖子最长。 | 5 |
| 尾巴最短。 | 5 |
| 尾巴很短。 | 4 |
| 尾巴和脖子很短。 | 4 |
| 腿最长。 | 4 |
| 头和腿短。 | 3 |
| 头最短。 | 3 |
| 腿长，脖子短。 | 3 |
| 头最长。 | 2 |
| 头长，腿短。 | 2 |
| 尾巴和腿最短。 | 2 |
| 四个部分差不多一样长。 | 2 |
| 头和腿最长。 | 2 |
| 头和尾巴比较短。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部分差不多一样长。 | 2 | equality | S1T84, S1T86 |
| 四个部分差不多都一样长。 | 1 | equality | S1T42 |
| 头和尾巴最短，脖子最长，腿次之。 | 1 | ranking | S1T5 |
| 头和腿一样长，脖子和尾巴最短。 | 1 | equality | S1T88 |
| 头很长，脖子很短，尾巴和腿差不多一样长。 | 1 | equality | S1T2 |
| 头最短，其余三部分差不多一样长。 | 1 | equality | S1T16 |
| 尾巴很短，脖子、腿、头差不多一样长。 | 1 | equality | S1T4 |
| 差不多，都一样长。 | 1 | equality | S1T90 |
| 脖子最短，其余三部分差不多一样长。 | 1 | equality | S1T13 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部分差不多一样长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T84, S1T86 |
| 脖子短。 | 2 | 0.000 | absolute_short:脖子 < 0.50 | S1T210, S1T243 |
| 脖子比较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T145 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T171 |

### S128

- trial 数: 192; 非空文本: 192; fidelity 可评分率: 1.000; 平均 fidelity: 0.926; 完全忠实率: 0.688; 低 fidelity 率: 0.010.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 192 | 1.000 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴较长。 | 16 |
| 尾巴较短。 | 15 |
| 尾巴短。 | 9 |
| 头、脖子、腿、尾巴中等长度。 | 9 |
| 尾巴长。 | 8 |
| 腿短，头、脖子、尾巴中等长度。 | 6 |
| 腿长，头、脖子、尾巴中等长度。 | 5 |
| 尾巴中等长度。 | 4 |
| 头、脖子长，腿、尾巴中等长度。 | 4 |
| 头、脖子长，腿长，尾巴短。 | 3 |
| 腿、尾巴短，头、脖子较长。 | 3 |
| 脖子长，头、腿、尾巴中等长度。 | 3 |
| 腿短，尾巴、脖子、头中等长度。 | 3 |
| 头、脖子、尾巴长，腿较短。 | 3 |
| 头长，腿短，脖子、尾巴中等长度。 | 2 |
| 尾巴长，头、脖子中等长度，腿较短。 | 2 |
| 脖子、尾巴较短，头、腿较长。 | 2 |
| 尾巴、脖子长，头、腿短。 | 2 |
| 尾巴短，腿、脖子、头中等长度。 | 2 |
| 腿较长，头、脖子、尾巴较短。 | 2 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴较短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T183 |
| 脖子短，头、腿、尾巴较长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T29 |

### S129

- trial 数: 256; 非空文本: 256; fidelity 可评分率: 0.762; 平均 fidelity: 0.829; 完全忠实率: 0.621; 低 fidelity 率: 0.125.
- 旧版 region 覆盖率: 0.762; 旧版 region 有未处理片段率: 0.242.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 209 | 0.816 |
| body_ref | 101 | 0.395 |
| comparison | 94 | 0.367 |
| equality | 44 | 0.172 |
| count_abstract | 21 | 0.082 |
| negation | 17 | 0.066 |
| other | 13 | 0.051 |
| superlative | 5 | 0.020 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子比躯干短。 | 26 |
| 脖子比躯干长。 | 24 |
| 脖子长。 | 24 |
| 脖子短。 | 15 |
| 一个部位很长。 | 10 |
| 头在躯干之下。 | 9 |
| 尾巴长。 | 9 |
| 尾巴和脖子不一样长。 | 7 |
| 头低于尾巴。 | 7 |
| 有两个部位一样长。 | 6 |
| 头比躯干长。 | 6 |
| 腿比躯干短。 | 6 |
| 四个部位都差不多。 | 5 |
| 尾巴短。 | 5 |
| 头比腿短。 | 5 |
| 腿短。 | 4 |
| 腿和躯干一样长。 | 4 |
| 头在躯干之上。 | 4 |
| 低头。 | 4 |
| 腿长。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子比躯干短。 | 26 | body_ref | S1T16, S1T17, S1T137, S1T211, S1T212, S1T214, S1T217, S1T218 |
| 脖子比躯干长。 | 24 | body_ref | S1T15, S1T213, S1T215, S1T216, S1T221, S1T222, S1T223, S1T224 |
| 一个部位很长。 | 10 | count_abstract | S1T32, S1T33, S1T36, S1T37, S1T38, S1T39, S1T43, S1T44 |
| 头在躯干之下。 | 9 | body_ref | S1T108, S1T111, S1T114, S1T116, S1T117, S1T118, S1T119, S1T121 |
| 尾巴和脖子不一样长。 | 7 | equality, negation | S1T26, S1T27, S1T54, S1T55, S1T57, S1T58, S1T155 |
| 头比躯干长。 | 6 | body_ref | S1T8, S1T10, S1T13, S1T14, S1T162, S1T164 |
| 有两个部位一样长。 | 6 | equality, count_abstract | S1T59, S1T60, S1T61, S1T62, S1T64, S1T66 |
| 腿比躯干短。 | 6 | body_ref | S1T1, S1T2, S1T3, S1T5, S1T172, S1T174 |
| 四个部位都差不多。 | 5 | equality | S1T40, S1T41, S1T45, S1T47, S1T52 |
| 低头。 | 4 | other | S1T94, S1T95, S1T97, S1T98 |
| 头在躯干之上。 | 4 | body_ref | S1T110, S1T112, S1T113, S1T126 |
| 腿和躯干一样长。 | 4 | equality, body_ref | S1T170, S1T173, S1T175, S1T176 |
| 四个部位和躯干都不一样长。 | 3 | equality, body_ref, negation | S1T158, S1T159, S1T160 |
| 头与躯干持平。 | 3 | body_ref | S1T109, S1T115, S1T120 |
| 头在腿之上。 | 3 | other | S1T89, S1T91, S1T101 |
| 头比躯干短。 | 3 | body_ref | S1T9, S1T11, S1T12 |
| 尾巴和脖子一样长。 | 3 | equality | S1T25, S1T53, S1T56 |
| 头和尾巴不一样长。 | 2 | equality, negation | S1T23, S1T24 |
| 头在腿上。 | 2 | other | S1T92, S1T93 |
| 头朝左。 | 2 | other | S1T6, S1T20 |
| 头等于尾巴。 | 2 | equality | S1T70, S1T78 |
| 尾巴比躯干短。 | 2 | body_ref | S1T18, S1T19 |
| 有一个部位比躯干长。 | 2 | body_ref, count_abstract | S1T130, S1T131 |
| 没有两个部位一样长。 | 2 | equality, count_abstract, negation | S1T63, S1T67 |
| 腿比躯干长。 | 2 | body_ref | S1T4, S1T21 |
| 一个部位很短。 | 1 | count_abstract | S1T34 |
| 四个部位都比躯干短。 | 1 | body_ref | S1T129 |
| 头、尾巴和腿都比躯干长。 | 1 | body_ref | S1T132 |
| 头和尾巴一样长。 | 1 | equality | S1T22 |
| 头和躯干一样长。 | 1 | equality, body_ref | S1T156 |
| 头接近于腿的高度。 | 1 | equality | S1T90 |
| 头的高度和尾巴类似。 | 1 | other | S1T79 |
| 尾巴和躯干一样长。 | 1 | equality, body_ref | S1T163 |
| 尾巴比躯干长。 | 1 | body_ref | S1T7 |
| 抬头。 | 1 | other | S1T96 |
| 脖子和尾巴一样长。 | 1 | equality | S1T168 |
| 脖子和尾巴不一样长。 | 1 | equality, negation | S1T165 |
| 脖子和躯干一样长。 | 1 | equality, body_ref | S1T157 |
| 脖子尾巴一样长。 | 1 | equality | S1T167 |
| 脖子尾巴不一样长。 | 1 | equality, negation | S1T166 |
| 腿和躯干不一样长。 | 1 | equality, body_ref, negation | S1T171 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位都差不多。 | 5 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T40, S1T41, S1T45, S1T47, S1T52 |
| 头与躯干持平。 | 3 | 0.000 | body_ref:头 = 0.50 | S1T109, S1T115, S1T120 |
| 脖子比躯干短。 | 3 | 0.000 | body_ref:脖子 < 0.50 | S1T16, S1T137, S1T255 |
| 腿和躯干一样长。 | 3 | 0.000 | body_ref:腿 = 0.50 | S1T170, S1T173, S1T176 |
| 头低于尾巴。 | 2 | 0.000 | comparison:头 < 尾巴 | S1T72, S1T74 |
| 头等于尾巴。 | 2 | 0.000 | comparison:头 = 尾巴 | S1T70, S1T78 |
| 脖子长。 | 2 | 0.000 | absolute_long:脖子 > 0.50 | S1T125, S1T191 |
| 腿比躯干短。 | 2 | 0.000 | body_ref:腿 < 0.50 | S1T2, S1T172 |
| 头和躯干一样长。 | 1 | 0.000 | body_ref:头 = 0.50 | S1T156 |
| 头接近于腿的高度。 | 1 | 0.000 | equality_range:头+腿 = | S1T90 |
| 头比腿短。 | 1 | 0.000 | comparison:头 < 腿 | S1T81 |
| 头的高度和尾巴类似。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T79 |
| 头高于尾巴。 | 1 | 0.000 | comparison:头 > 尾巴 | S1T73 |
| 尾巴和脖子一样长。 | 1 | 0.000 | equality_range:尾巴+脖子 = | S1T53 |
| 尾巴和躯干一样长。 | 1 | 0.000 | body_ref:尾巴 = 0.50 | S1T163 |
| 尾巴比躯干短。 | 1 | 0.000 | body_ref:尾巴 < 0.50 | S1T18 |
| 脖子和躯干一样长。 | 1 | 0.000 | body_ref:脖子 = 0.50 | S1T157 |
| 腿最短。 | 1 | 0.333 | superlative:腿 < 头; superlative:腿 < 尾巴 | S1T49 |

### S130

- trial 数: 448; 非空文本: 419; fidelity 可评分率: 0.850; 平均 fidelity: 0.841; 完全忠实率: 0.656; 低 fidelity 率: 0.121.
- 旧版 region 覆盖率: 0.850; 旧版 region 有未处理片段率: 0.085.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 397 | 0.886 |
| comparison | 249 | 0.556 |
| group_sum | 54 | 0.121 |
| count_abstract | 41 | 0.092 |
| empty | 29 | 0.065 |
| superlative | 21 | 0.047 |
| equality | 8 | 0.018 |
| body_ref | 6 | 0.013 |
| other | 3 | 0.007 |
| negation | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子长。 | 62 |
| 脖子短。 | 60 |
| 腿比脖子短。 | 47 |
| 四条腿之和比其他三个部位之和长。 | 24 |
| 腿比脖子长。 | 21 |
| 四条腿之和比其他三个部位之和短。 | 17 |
| 腿和尾巴都比脖子长。 | 16 |
| 腿和尾巴比脖子长。 | 15 |
| 腿和尾巴比脖子短。 | 10 |
| 腿比头加脖子短。 | 7 |
| 腿长。 | 7 |
| 腿最短。 | 7 |
| 头加脖子和腿加尾巴差不多长。 | 6 |
| 尾巴比脖子长。 | 5 |
| 腿短。 | 5 |
| 尾巴和腿都比脖子短。 | 5 |
| 头身比例比较协调。 | 5 |
| 尾巴比脖子短。 | 5 |
| 头加脖子比腿加尾巴长。 | 4 |
| 头身比例不协调。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四条腿之和比其他三个部位之和长。 | 24 | group_sum, count_abstract | S1T43, S1T45, S1T46, S1T49, S1T50, S1T51, S1T52, S1T54 |
| 四条腿之和比其他三个部位之和短。 | 17 | group_sum, count_abstract | S1T42, S1T44, S1T47, S1T48, S1T53, S1T56, S1T60, S1T61 |
| 头加脖子和腿加尾巴差不多长。 | 6 | equality | S1T90, S1T91, S1T92, S1T93, S1T94, S1T95 |
| 头身比例比较协调。 | 5 | group_sum | S1T30, S1T31, S1T32, S1T33, S1T36 |
| 头身比例不协调。 | 4 | group_sum | S1T29, S1T34, S1T35, S1T37 |
| 像爬行类的动物。 | 2 | other | S1T113, S1T114 |
| 四个部位中有两个超过了一半。 | 2 | body_ref | S1T16, S1T18 |
| 头加脖子加尾巴的长度大于四条腿的长度之和。 | 2 | group_sum | S1T21, S1T23 |
| 像直立行走的动物。 | 1 | other | S1T122 |
| 四个部位中有三个的长度超过了一半。 | 1 | body_ref | S1T19 |
| 四个部位中有三个长度超过了一半。 | 1 | body_ref | S1T15 |
| 四个部位差不多长。 | 1 | equality | S1T28 |
| 四个部位的长度都超过了一半。 | 1 | body_ref | S1T17 |
| 头加脖子加尾巴的长度小于四条腿的长度之和。 | 1 | group_sum | S1T25 |
| 尾巴和脖子一样长。 | 1 | equality | S1T254 |
| 所有部位长度都超过一半。 | 1 | body_ref | S1T130 |
| 比例看起来不是很协调。 | 1 | group_sum, negation | S1T20 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四条腿之和比其他三个部位之和长。 | 17 | 0.255 | complement:脖子 > 0.50; complement:头 > 0.50; complement:尾巴 > 0.50 | S1T45, S1T46, S1T49, S1T54, S1T55, S1T57, S1T63, S1T66 |
| 四条腿之和比其他三个部位之和短。 | 9 | 0.185 | complement:脖子 < 0.50; complement:头 < 0.50; complement:尾巴 < 0.50 | S1T42, S1T44, S1T48, S1T60, S1T61, S1T62, S1T68, S1T70 |
| 头加脖子和腿加尾巴差不多长。 | 6 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T90, S1T91, S1T92, S1T93, S1T94, S1T95 |
| 脖子长。 | 6 | 0.000 | absolute_long:脖子 > 0.50 | S2T19, S2T34, S2T45, S2T49, S2T92, S2T125 |
| 脖子短。 | 5 | 0.000 | absolute_short:脖子 < 0.50 | S2T29, S2T35, S2T77, S2T79, S2T109 |
| 腿和尾巴比脖子长。 | 3 | 0.000 | comparison:腿+尾巴 > 脖子 | S1T290, S1T295, S1T313 |
| 腿比脖子短。 | 2 | 0.000 | comparison:腿 < 脖子 | S1T189, S1T308 |
| 四个部位差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T28 |
| 头加脖子加尾巴的长度小于四条腿的长度之和。 | 1 | 0.000 | group_sum:头+脖子+尾巴 < 腿 | S1T25 |
| 头和尾巴比脖子长。 | 1 | 0.000 | comparison:头+尾巴 > 脖子 | S1T305 |
| 头比腿的位置高。 | 1 | 0.000 | comparison:头 > 腿 | S1T111 |
| 尾巴比脖子长。 | 1 | 0.000 | comparison:尾巴 > 脖子 | S1T275 |
| 四个部位中有三个的长度超过了一半。 | 1 | 0.250 | body_ref:脖子 > 0.50; body_ref:头 > 0.50; body_ref:腿 > 0.50 | S1T19 |

### S131

- trial 数: 192; 非空文本: 192; fidelity 可评分率: 0.896; 平均 fidelity: 0.833; 完全忠实率: 0.667; 低 fidelity 率: 0.089.
- 旧版 region 覆盖率: 0.896; 旧版 region 有未处理片段率: 0.203.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 171 | 0.891 |
| comparison | 38 | 0.198 |
| body_ref | 22 | 0.115 |
| equality | 18 | 0.094 |
| other | 9 | 0.047 |
| negation | 3 | 0.016 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴长。 | 34 |
| 尾巴短。 | 24 |
| 尾巴长，腿短。 | 13 |
| 尾巴比身子长。 | 11 |
| 尾巴短，腿长。 | 7 |
| 尾巴比身子短。 | 5 |
| 尾巴长，腿也很长。 | 4 |
| 体型分布的不均匀。 | 4 |
| 腿很短。 | 3 |
| 腿比身子长。 | 3 |
| 体型分布的很均匀。 | 3 |
| 朝左边。 | 3 |
| 腿比尾巴短。 | 2 |
| 它像一个小型的动物。 | 2 |
| 腿比头长，朝右边。 | 2 |
| 腿比头短。 | 2 |
| 腿比身子短。 | 2 |
| 朝右边。 | 2 |
| 尾巴很长。 | 2 |
| 尾巴很短。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 尾巴比身子长。 | 11 | body_ref | S1T106, S1T107, S1T108, S1T111, S1T112, S1T113, S1T114, S1T115 |
| 尾巴比身子短。 | 5 | body_ref | S1T109, S1T110, S1T116, S1T117, S1T120 |
| 体型分布的不均匀。 | 4 | equality | S1T70, S1T71, S1T79, S1T81 |
| 体型分布的很均匀。 | 3 | equality | S1T74, S1T75, S1T77 |
| 朝左边。 | 3 | other | S1T27, S1T51, S1T53 |
| 腿比身子长。 | 3 | body_ref | S1T33, S1T34, S1T37 |
| 它像一个小型的动物。 | 2 | other | S1T65, S1T67 |
| 朝右边。 | 2 | other | S1T52, S1T54 |
| 腿比身子短。 | 2 | body_ref | S1T35, S1T36 |
| 体型分布得不均匀。 | 1 | equality | S1T80 |
| 体型分布得很均匀。 | 1 | equality | S1T76 |
| 体型分布的不均匀，且方向是朝左。 | 1 | equality | S1T82 |
| 体型分布的还算均匀，头很长。 | 1 | equality | S1T72 |
| 体型比较大且分布均匀。 | 1 | equality | S1T69 |
| 各部位分布不均匀，腿短。 | 1 | equality | S1T62 |
| 各部位分布均匀。 | 1 | equality | S1T61 |
| 各部位分布均匀，都很长。 | 1 | equality | S1T64 |
| 头长，脖子不短，尾巴也很长。 | 1 | negation | S1T13 |
| 头长，脖子长，尾巴也不是很短。 | 1 | negation | S1T8 |
| 头长，脖子长，尾巴长，腿也不短。 | 1 | negation | S1T11 |
| 头长，腿很长，分布均匀。 | 1 | equality | S1T68 |
| 它像一个大型的动物。 | 1 | other | S1T66 |
| 是个大型动物。 | 1 | other | S1T131 |
| 腿很长，分布得很均匀。 | 1 | equality | S1T73 |
| 腿短，不均匀。 | 1 | equality | S1T63 |
| 躯干比腿短。 | 1 | body_ref | S1T78 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴长。 | 6 | 0.000 | absolute_long:尾巴 > 0.50 | S1T101, S1T102, S1T153, S1T157, S1T171, S1T177 |
| 各部位分布均匀。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T61 |
| 各部位分布均匀，都很长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T64 |
| 头很长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T94 |
| 头长，腿很长，分布均匀。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T68 |
| 尾巴有点长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T19 |
| 尾巴比身子短。 | 1 | 0.000 | body_ref:尾巴 < 0.50 | S1T116 |
| 尾巴比身子长。 | 1 | 0.000 | body_ref:尾巴 > 0.50 | S1T121 |
| 尾巴长，腿也长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50 | S1T21 |
| 腿比头短，朝左边。 | 1 | 0.000 | comparison:腿 < 头 | S1T43 |
| 躯干比腿短。 | 1 | 0.000 | body_ref:腿 < 0.50 | S1T78 |
| 头长，脖子长，尾巴长，腿也不短。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50 | S1T11 |

### S132

- trial 数: 384; 非空文本: 361; fidelity 可评分率: 0.904; 平均 fidelity: 0.968; 完全忠实率: 0.875; 低 fidelity 率: 0.029.
- 旧版 region 覆盖率: 0.904; 旧版 region 有未处理片段率: 0.036.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 361 | 0.940 |
| comparison | 259 | 0.674 |
| empty | 23 | 0.060 |
| superlative | 23 | 0.060 |
| negation | 11 | 0.029 |
| equality | 7 | 0.018 |
| group_sum | 2 | 0.005 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴比腿长。 | 34 |
| 尾巴比腿短。 | 31 |
| 尾巴比脖子和头都长。 | 29 |
| 尾巴短。 | 26 |
| 尾巴比头和脖子都短。 | 24 |
| 尾巴长。 | 24 |
| 尾巴比脖子短。 | 24 |
| 尾巴比脖子长。 | 22 |
| 尾巴比头和脖子都长。 | 14 |
| 尾巴比头短。 | 11 |
| 尾巴较短。 | 11 |
| 尾巴比脖子和头都短。 | 11 |
| 尾巴较长。 | 10 |
| 尾巴不是最短的。 | 8 |
| 尾巴比较长。 | 7 |
| 腿比较长。 | 6 |
| 尾巴比头长、比脖子短。 | 6 |
| 尾巴比头长。 | 5 |
| 尾巴和脖子一样长。 | 5 |
| 尾巴比头短、比脖子长。 | 5 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 尾巴不是最短的。 | 8 | negation | S1T118, S1T144, S1T145, S1T146, S1T147, S1T149, S1T151, S1T154 |
| 尾巴和脖子一样长。 | 5 | equality | S1T285, S1T287, S1T308, S1T309, S1T312 |
| 尾巴不是最长的。 | 3 | negation | S1T93, S1T152, S1T153 |
| 头最长，脖子、尾巴和腿差不多。 | 1 | equality | S1T171 |
| 尾巴和腿加起来比脖子和头加起来更短。 | 1 | group_sum | S1T97 |
| 尾巴和腿加起来比脖子和头加起来更长。 | 1 | group_sum | S1T98 |
| 脖子和尾巴一样长。 | 1 | equality | S1T1 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴比腿长。 | 4 | 0.000 | comparison:尾巴 > 腿 | S1T25, S1T27, S1T34, S1T41 |
| 尾巴和脖子一样长。 | 1 | 0.000 | equality_range:尾巴+脖子 = | S1T285 |
| 尾巴比较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S2T9 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S2T24 |
| 尾巴较短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T115 |
| 尾巴较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T186 |
| 腿比较短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S1T161 |
| 腿比较长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T157 |

### S201

- trial 数: 640; 非空文本: 640; fidelity 可评分率: 1.000; 平均 fidelity: 0.871; 完全忠实率: 0.759; 低 fidelity 率: 0.077.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 640 | 1.000 |
| comparison | 190 | 0.297 |
| equality | 30 | 0.047 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子长，头长。 | 51 |
| 脖子长，头短。 | 48 |
| 脖子短，腿长。 | 36 |
| 腿很长。 | 36 |
| 所有部位长度适中。 | 35 |
| 头很长。 | 32 |
| 脖子短，腿短。 | 28 |
| 头和脖子比较长。 | 23 |
| 所有部位都比较长。 | 23 |
| 所有部位长度相当。 | 19 |
| 脖子很长。 | 15 |
| 尾巴很长。 | 14 |
| 腿比较长。 | 13 |
| 腿很短。 | 12 |
| 头很短。 | 11 |
| 所有部位都比较短。 | 9 |
| 腿比较短。 | 8 |
| 脖子很短。 | 7 |
| 头比较长。 | 7 |
| 四个部位长度相当。 | 7 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 所有部位长度相当。 | 19 | equality | S2T7, S2T10, S2T17, S2T20, S2T44, S2T50, S2T51, S2T52 |
| 四个部位长度相当。 | 7 | equality | S2T6, S2T23, S2T24, S2T32, S2T33, S2T87, S2T121 |
| 所有部位长度相近。 | 2 | equality | S1T196, S1T251 |
| 腿和头的长度差不多。 | 1 | equality | S1T54 |
| 腿和头的长度相当。 | 1 | equality | S1T62 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 所有部位长度相当。 | 19 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T7, S2T10, S2T17, S2T20, S2T44, S2T50, S2T51, S2T52 |
| 四个部位长度相当。 | 7 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T6, S2T23, S2T24, S2T32, S2T33, S2T87, S2T121 |
| 头比较长。 | 3 | 0.000 | absolute_long:头 > 0.50 | S1T7, S1T27, S1T89 |
| 所有部位长度相近。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T196, S1T251 |
| 脖子以外的部位都比较长。 | 2 | 0.000 | absolute_long:脖子 > 0.50 | S1T199, S1T237 |
| 脖子很长。 | 2 | 0.000 | absolute_long:脖子 > 0.50 | S1T106, S1T107 |
| 腿比较长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S1T39, S1T41 |
| 腿很短，其他部位比较长。 | 2 | 0.250 | complement:脖子 > 0.50; complement:头 > 0.50; complement:尾巴 > 0.50 | S2T27, S2T28 |
| 头很长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T6 |
| 尾巴以外的部位都比较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T200 |
| 脖子以外的部位很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T267 |
| 腿和头的长度差不多。 | 1 | 0.000 | equality_range:腿+头 = | S1T54 |
| 腿比较短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S1T71 |
| 除尾巴以外的部位比较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S2T54 |
| 头很短，其他比较长。 | 1 | 0.250 | complement:脖子 > 0.50; complement:腿 > 0.50; complement:尾巴 > 0.50 | S2T97 |
| 所有部位都比较长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T234 |
| 腿比较短，其他部位比较长。 | 1 | 0.250 | complement:脖子 > 0.50; complement:头 > 0.50; complement:尾巴 > 0.50 | S2T60 |
| 腿很短，头和尾巴很长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T202 |

### S202

- trial 数: 256; 非空文本: 255; fidelity 可评分率: 0.992; 平均 fidelity: 0.949; 完全忠实率: 0.895; 低 fidelity 率: 0.008.
- 旧版 region 覆盖率: 0.992; 旧版 region 有未处理片段率: 0.004.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 254 | 0.992 |
| comparison | 38 | 0.148 |
| empty | 1 | 0.004 |
| meta | 1 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子长，腿长。 | 50 |
| 脖子长，腿短。 | 46 |
| 脖子短，头长。 | 40 |
| 脖子短，头短。 | 39 |
| 脖子比较短。 | 19 |
| 脖子比较长。 | 12 |
| 脖子短，腿长。 | 7 |
| 脖子短，腿短。 | 5 |
| 脖子短，尾巴长。 | 5 |
| 腿短。 | 4 |
| 腿比较短。 | 3 |
| 尾巴比较长。 | 2 |
| 脖子短，头长，腿短，尾巴长。 | 2 |
| 腿长。 | 2 |
| 头短，脖子短。 | 2 |
| 脖子短，其他长。 | 2 |
| 四个部位都长。 | 2 |
| 腿比较长，脖子比较短。 | 1 |
| 腿比较长。 | 1 |
| 脖子长，腿短，尾巴长。 | 1 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 选错了。 | 1 | meta | S1T209 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子短，头短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S1T224 |
| 脖子长，腿长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S1T134 |

### S203

- trial 数: 832; 非空文本: 807; fidelity 可评分率: 0.968; 平均 fidelity: 0.937; 完全忠实率: 0.778; 低 fidelity 率: 0.008.
- 旧版 region 覆盖率: 0.968; 旧版 region 有未处理片段率: 0.012.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 805 | 0.968 |
| superlative | 86 | 0.103 |
| comparison | 60 | 0.072 |
| empty | 25 | 0.030 |
| ranking | 15 | 0.018 |
| equality | 4 | 0.005 |
| body_ref | 2 | 0.002 |
| count_abstract | 1 | 0.001 |
| negation | 1 | 0.001 |
| meta | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头短，腿短。 | 33 |
| 头长，脖子长。 | 30 |
| 头长，脖子短。 | 30 |
| 头短，腿长。 | 17 |
| 头长，脖子长，腿短，尾巴短。 | 9 |
| 腿长，其余短。 | 7 |
| 尾巴短，其余长。 | 7 |
| 腿很长。 | 6 |
| 头和脖子很长。 | 6 |
| 头短，腿短，脖子长，尾巴长。 | 6 |
| 头和脖子很长，尾巴很短。 | 6 |
| 头短于脖子，尾巴短于腿。 | 5 |
| 腿最长。 | 5 |
| 尾巴很长。 | 4 |
| 头长于脖子，尾巴长于腿。 | 4 |
| 脖子长，头长，腿短，尾巴短。 | 4 |
| 头最长，脖子次之。 | 4 |
| 脖子短，腿短，头长，尾巴长。 | 4 |
| 所有部位都很长。 | 4 |
| 头很长。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头最长，脖子次之。 | 4 | ranking | S1T181, S1T232, S3T49, S3T50 |
| 脖子最短，尾巴次之，其余很长。 | 2 | ranking | S1T277, S1T283 |
| 头很长，脖子次之，另外两者很短。 | 1 | ranking | S1T170 |
| 头很长，腿中等，其余两个部位很短。 | 1 | count_abstract | S1T32 |
| 头很长，腿很长，脖、尾巴和脖子稍短，而且尾巴和脖子长度相近。 | 1 | equality | S1T46 |
| 头很长，腿次之，尾巴和脖子很短。 | 1 | ranking | S1T173 |
| 头最长，尾巴次之，脖子、腿都很短。 | 1 | ranking | S2T22 |
| 头最长，脖子次之，其余很短。 | 1 | ranking | S1T182 |
| 头最长，腿次之，尾巴和脖子很短。 | 1 | ranking | S1T172 |
| 头长于脖子，头和腿差不多长，尾巴很短。 | 1 | equality | S2T104 |
| 头长，脖子次之，其余短。 | 1 | ranking | S2T124 |
| 头长，脖子次之，尾巴中等，腿短。 | 1 | ranking | S2T125 |
| 尾巴很长，头最短，躯干中等。 | 1 | body_ref | S1T155 |
| 脖子最短，头次之，其余很长。 | 1 | ranking | S1T286 |
| 脖子最长，头次之，其余很短。 | 1 | ranking | S1T278 |
| 腿不长，脖子不短，头很长。 | 1 | negation | S2T82 |
| 腿很长，头很短，尾巴和脖子长度相近。 | 1 | equality | S1T54 |
| 腿很长，头很短，脖子和尾巴相近。 | 1 | equality | S1T38 |
| 身体各个部位都很匀称。 | 1 | body_ref | S1T15 |
| 选错了。 | 1 | meta | S2T112 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 除了腿都很长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S2T17, S2T178 |
| 头和脖子长，腿短，尾巴略长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T213 |
| 尾巴很长，其余很短。 | 1 | 0.250 | complement:脖子 < 0.50; complement:头 < 0.50; complement:腿 < 0.50 | S1T276 |
| 头最长，脖子次之。 | 1 | 0.333 | superlative:头 > 脖子; superlative:头 > 尾巴 | S3T50 |
| 脖子短，头和尾巴略长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T264 |
| 脖子中等，其他部位较长。 | 1 | 0.400 | complement:头 > 0.50; complement:腿 > 0.50; complement:尾巴 > 0.50 | S2T72 |

### S204

- trial 数: 512; 非空文本: 512; fidelity 可评分率: 0.996; 平均 fidelity: 0.914; 完全忠实率: 0.834; 低 fidelity 率: 0.033.
- 旧版 region 覆盖率: 0.996; 旧版 region 有未处理片段率: 0.027.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 511 | 0.998 |
| comparison | 104 | 0.203 |
| negation | 39 | 0.076 |
| equality | 29 | 0.057 |
| superlative | 11 | 0.021 |
| body_ref | 3 | 0.006 |
| count_abstract | 2 | 0.004 |
| ranking | 1 | 0.002 |
| meta | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，腿长。 | 69 |
| 头短，脖子长。 | 57 |
| 头长，腿短。 | 54 |
| 头短，脖子短。 | 33 |
| 头和尾巴短，脖子和腿长。 | 5 |
| 头很长，腿很短。 | 5 |
| 头长，腿不短。 | 4 |
| 头短，脖子不够长。 | 4 |
| 尾巴太长。 | 3 |
| 头长，腿略短。 | 3 |
| 头够长，腿够短。 | 3 |
| 四个部位都挺长。 | 3 |
| 头短，脖子没那么长。 | 3 |
| 腿和尾巴比较长。 | 2 |
| 脖子突出短。 | 2 |
| 头短，脖子不长。 | 2 |
| 尾巴长，头长。 | 2 |
| 脖子比较长。 | 2 |
| 尾巴是最长的部位。 | 2 |
| 腿短，其他三个差不多。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头短，脖子不够长。 | 4 | negation | S2T19, S2T20, S2T52, S2T111 |
| 头长，腿不短。 | 4 | negation | S1T157, S1T181, S1T272, S1T304 |
| 头、脖子、尾巴比较长、基本一样长，腿比较短。 | 2 | equality | S1T3, S1T4 |
| 头和脖子不够长。 | 2 | negation | S2T55, S2T56 |
| 头够长，腿不够短。 | 2 | negation | S1T226, S1T227 |
| 头短，脖子不长。 | 2 | negation | S1T300, S2T9 |
| 脖子和尾巴一样长。 | 2 | equality | S1T25, S1T29 |
| 腿短，其他三个差不多。 | 2 | equality | S1T41, S1T42 |
| 四个部位几乎一样长，都挺长。 | 1 | equality | S1T130 |
| 四个部位差不多一样长。 | 1 | equality | S1T56 |
| 四个部位都差不多。 | 1 | equality | S1T205 |
| 四个部位都比较长，长度都差不多。 | 1 | equality | S1T203 |
| 四个部位长度都差不多。 | 1 | equality | S1T207 |
| 头、脖子、尾巴一样长，腿比较短。 | 1 | equality | S1T13 |
| 头、脖子、腿差不多一样长，尾巴比较短。 | 1 | equality | S1T33 |
| 头不够长，尾巴不够短。 | 1 | negation | S1T239 |
| 头不够长，尾巴够长。 | 1 | negation | S1T228 |
| 头不够长，尾巴很长。 | 1 | negation | S1T269 |
| 头不够长，脖子够长。 | 1 | negation | S1T308 |
| 头不短，尾巴很短。 | 1 | negation | S1T194 |
| 头不长，腿短。 | 1 | negation | S1T240 |
| 头和尾巴差不多长，脖子和腿差不多长。 | 1 | equality | S1T15 |
| 头和尾巴比较短，脖子和腿长，选错了。 | 1 | meta | S1T186 |
| 头和脖子一样长，腿和尾巴一样长。 | 1 | equality | S1T214 |
| 头和脖子差不多短，腿和尾巴差不多长。 | 1 | equality | S1T70 |
| 头和脖子差不多长，尾巴短，腿长。 | 1 | equality | S1T9 |
| 头和脖子都不长。 | 1 | negation | S2T123 |
| 头够长，尾巴、腿不够短。 | 1 | negation | S1T229 |
| 头够长，腿不短。 | 1 | negation | S1T171 |
| 头很长，腿不够短。 | 1 | negation | S1T118 |
| 头很长，腿差不多正好。 | 1 | equality | S1T241 |
| 头是最长的部位，其他差不多。 | 1 | equality | S1T75 |
| 头有点不够长。 | 1 | negation | S1T285 |
| 头比较短，脖子、尾巴、腿比较长、差不多一样长。 | 1 | equality | S1T10 |
| 头特别长，脖子其次，腿和尾巴都很短。 | 1 | ranking | S1T64 |
| 头短，但是尾巴不短，脖子和腿都比较长。 | 1 | negation | S1T105 |
| 头短，脖子差不多刚好。 | 1 | equality | S2T91 |
| 头短，脖子长，脖子和躯干差不多。 | 1 | equality, body_ref | S1T276 |
| 头突出的短，其他差不多长。 | 1 | equality | S1T71 |
| 头长，腿不够短。 | 1 | negation | S1T189 |
| 头长，腿不短，脖子很短。 | 1 | negation | S1T182 |
| 头长，腿不短，脖子短。 | 1 | negation | S1T169 |
| 头长，腿长，很均衡。 | 1 | equality | S1T311 |
| 尾巴比较短，和头、脖子差不多，腿很长。 | 1 | equality | S1T53 |
| 尾巴相对也比较短，但是和脖子差不多长。 | 1 | equality | S1T51 |
| 尾巴相当短，腿相当长。 | 1 | equality | S1T187 |
| 脖子不够长。 | 1 | negation | S1T221 |
| 脖子很短，腿不长。 | 1 | negation | S1T208 |
| 脖子比较短，其他部位比较长、基本一样长。 | 1 | equality | S1T2 |
| 脖子比较短，腿比较长，另外两个部位在中间长度。 | 1 | count_abstract | S1T80 |
| 脖子短，头不够长。 | 1 | negation | S2T125 |
| 脖子短，头长，尾巴不太短。 | 1 | negation | S1T74 |
| 脖子短，腿不短。 | 1 | negation | S1T140 |
| 脖子长，但是头不长。 | 1 | negation | S1T69 |
| 腿不短。 | 1 | negation | S1T281 |
| 腿很短，头不够长。 | 1 | negation | S1T119 |
| 腿比躯干短。 | 1 | body_ref | S1T126 |
| 腿比躯干短，不过尾巴很短。 | 1 | body_ref | S1T287 |
| 腿比较短，其他三个部位差不多都比较长。 | 1 | equality, count_abstract | S1T21 |
| 腿比较短，头不太长。 | 1 | negation | S1T52 |
| 腿短，头不是特别够长。 | 1 | negation | S1T251 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、尾巴比较长、基本一样长，腿比较短。 | 2 | 0.000 | equality_range:头+脖子+尾巴 =; absolute_short:腿 < 0.50 | S1T3, S1T4 |
| 四个部位几乎一样长，都挺长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T130 |
| 四个部位差不多一样长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T56 |
| 四个部位都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T205 |
| 四个部位长度都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T207 |
| 头和尾巴比脖子和腿长。 | 1 | 0.000 | comparison:头+尾巴 > 脖子+腿 | S1T139 |
| 头和脖子都比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T39 |
| 头有一些长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T216 |
| 头有一点长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T237 |
| 脖子和尾巴一样长。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T29 |
| 腿不短。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T281 |
| 腿比躯干短，不过尾巴很短。 | 1 | 0.000 | body_ref:腿 < 0.50; absolute_long:尾巴 > 0.50 | S1T287 |
| 其他部位都挺长，尾巴比较短。 | 1 | 0.250 | complement:脖子 > 0.50; complement:头 > 0.50; complement:腿 > 0.50 | S1T196 |
| 头突出的短，其他差不多长。 | 1 | 0.250 | complement:脖子 < 0.50; complement:腿 < 0.50; complement:尾巴 < 0.50 | S1T71 |
| 腿比较短，其他三个部位差不多都比较长。 | 1 | 0.250 | complement:脖子 < 0.50; complement:头 < 0.50; complement:尾巴 < 0.50 | S1T21 |
| 头短，尾巴、脖子略长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S1T316 |

### S205

- trial 数: 704; 非空文本: 704; fidelity 可评分率: 0.999; 平均 fidelity: 0.839; 完全忠实率: 0.716; 低 fidelity 率: 0.114.
- 旧版 region 覆盖率: 0.999; 旧版 region 有未处理片段率: 0.006.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 703 | 0.999 |
| superlative | 522 | 0.741 |
| equality | 129 | 0.183 |
| body_ref | 30 | 0.043 |
| comparison | 3 | 0.004 |
| ranking | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子最长。 | 65 |
| 头最长。 | 43 |
| 脖子最短。 | 43 |
| 腿最短。 | 42 |
| 腿最长。 | 36 |
| 尾巴最长。 | 36 |
| 尾巴最短。 | 25 |
| 脖子和尾巴差不多长。 | 19 |
| 头和脖子最长。 | 17 |
| 头和脖子一样长。 | 16 |
| 腿和脖子最长。 | 15 |
| 脖子和尾巴最长。 | 14 |
| 腿和躯干一样长。 | 13 |
| 头最短。 | 12 |
| 头最长，腿最短。 | 11 |
| 尾巴和脖子差不多长。 | 11 |
| 腿长，脖子短。 | 10 |
| 腿最长，脖子最短。 | 10 |
| 腿短，头长。 | 10 |
| 尾巴和躯干一样长。 | 9 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子和尾巴差不多长。 | 19 | equality | S1T35, S1T37, S1T50, S1T51, S1T52, S1T57, S1T67, S1T68 |
| 头和脖子一样长。 | 16 | equality | S1T78, S1T134, S1T177, S1T217, S1T269, S1T270, S1T271, S1T277 |
| 腿和躯干一样长。 | 13 | equality, body_ref | S1T147, S1T148, S1T150, S1T151, S1T153, S1T154, S1T157, S1T158 |
| 尾巴和脖子差不多长。 | 11 | equality | S1T82, S1T83, S1T88, S1T90, S1T91, S1T93, S1T96, S1T108 |
| 尾巴和躯干一样长。 | 9 | equality, body_ref | S1T149, S1T152, S1T156, S1T159, S1T161, S1T172, S1T173, S1T191 |
| 脖子和头一样长。 | 6 | equality | S1T140, S1T146, S1T165, S1T181, S1T187, S1T197 |
| 脖子和尾巴一样长。 | 5 | equality | S1T246, S1T276, S2T18, S2T21, S2T26 |
| 脖子和躯干一样长。 | 3 | equality, body_ref | S1T145, S1T155, S1T163 |
| 腿、脖子和头差不多长。 | 3 | equality | S1T100, S1T103, S1T115 |
| 腿和脖子差不多长。 | 3 | equality | S1T98, S1T99, S1T102 |
| 四个部位长度差不多。 | 2 | equality | S1T116, S1T119 |
| 头、脖子、尾巴一样长。 | 2 | equality | S1T234, S2T35 |
| 头、脖子、尾巴差不多长。 | 2 | equality | S1T106, S1T117 |
| 头和尾巴一样长。 | 2 | equality | S1T171, S2T19 |
| 头和脖子差不多长。 | 2 | equality | S1T112, S1T205 |
| 尾巴和脖子一样长。 | 2 | equality | S1T206, S2T2 |
| 尾巴和腿一样长。 | 2 | equality | S1T167, S1T175 |
| 尾巴和躯干差不多长。 | 2 | equality, body_ref | S1T143, S1T180 |
| 脖子和头差不多长。 | 2 | equality | S1T141, S1T242 |
| 腿、尾巴和头差不多长。 | 2 | equality | S1T107, S1T118 |
| 头、尾巴、脖子一样长。 | 1 | equality | S2T61 |
| 头、脖子和腿，差不多长。 | 1 | equality | S1T126 |
| 头和尾巴差不多长。 | 1 | equality | S1T111 |
| 尾巴、头、腿一样长。 | 1 | equality | S1T176 |
| 尾巴、腿、躯干一样长。 | 1 | equality, body_ref | S1T166 |
| 尾巴和头一样长。 | 1 | equality | S1T207 |
| 所有部位差不多长。 | 1 | equality | S1T109 |
| 脖子、头、尾巴差不多长。 | 1 | equality | S1T200 |
| 脖子、头、腿差不多长。 | 1 | equality | S1T97 |
| 脖子和尾巴长度差不多。 | 1 | equality | S1T122 |
| 脖子和腿一样长。 | 1 | equality | S1T185 |
| 脖子和腿差不多长。 | 1 | equality | S1T139 |
| 腿、头、尾巴差不多长。 | 1 | equality | S1T105 |
| 腿、尾巴、脖子差不多长。 | 1 | equality | S1T110 |
| 腿、尾巴、躯干一样长。 | 1 | equality, body_ref | S1T199 |
| 腿、脖子、尾巴差不多长。 | 1 | equality | S1T104 |
| 腿和头差不多长。 | 1 | equality | S1T94 |
| 腿和尾巴一样长。 | 1 | equality | S1T170 |
| 腿和尾巴差不多长。 | 1 | equality | S1T113 |
| 腿和脖子一样长。 | 1 | equality | S1T218 |
| 腿和躯干差不多长。 | 1 | equality, body_ref | S1T144 |
| 腿最短，脖子次之。 | 1 | ranking | S1T49 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿和躯干一样长。 | 13 | 0.000 | body_ref:腿 = 0.50 | S1T147, S1T148, S1T150, S1T151, S1T153, S1T154, S1T157, S1T158 |
| 尾巴和躯干一样长。 | 9 | 0.000 | body_ref:尾巴 = 0.50 | S1T149, S1T152, S1T156, S1T159, S1T161, S1T172, S1T173, S1T191 |
| 尾巴和脖子差不多长。 | 6 | 0.000 | equality_range:尾巴+脖子 = | S1T83, S1T88, S1T90, S1T108, S1T174, S1T201 |
| 脖子和尾巴差不多长。 | 5 | 0.000 | equality_range:脖子+尾巴 = | S1T37, S1T51, S1T120, S1T128, S1T137 |
| 脖子和躯干一样长。 | 3 | 0.000 | body_ref:脖子 = 0.50 | S1T145, S1T155, S1T163 |
| 腿、脖子和头差不多长。 | 3 | 0.000 | equality_range:腿+脖子+头 = | S1T100, S1T103, S1T115 |
| 四个部位长度差不多。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T116, S1T119 |
| 头、脖子、尾巴差不多长。 | 2 | 0.000 | equality_range:头+脖子+尾巴 = | S1T106, S1T117 |
| 头和脖子一样长。 | 2 | 0.000 | equality_range:头+脖子 = | S1T177, S1T271 |
| 腿最短。 | 2 | 0.333 | superlative:腿 < 头; superlative:腿 < 尾巴; superlative:腿 < 脖子 | S2T13, S2T289 |
| 头、尾巴、脖子一样长。 | 1 | 0.000 | equality_range:头+尾巴+脖子 = | S2T61 |
| 头、脖子、尾巴一样长。 | 1 | 0.000 | equality_range:头+脖子+尾巴 = | S1T234 |
| 头和尾巴一样长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T171 |
| 尾巴、头、腿一样长。 | 1 | 0.000 | equality_range:尾巴+头+腿 = | S1T176 |
| 尾巴和头一样长。 | 1 | 0.000 | equality_range:尾巴+头 = | S1T207 |
| 尾巴和脖子一样长。 | 1 | 0.000 | equality_range:尾巴+脖子 = | S2T2 |
| 尾巴和躯干差不多长。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T143 |
| 尾巴最长，脖子和腿最短。 | 1 | 0.000 | superlative:尾巴 > 脖子; superlative:尾巴 > 头; superlative:尾巴 > 腿; superlative:腿 < 脖子 | S2T217 |
| 所有部位差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T109 |
| 脖子、头、尾巴差不多长。 | 1 | 0.000 | equality_range:脖子+头+尾巴 = | S1T200 |
| 脖子、头、腿差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿 = | S1T97 |
| 脖子和头一样长。 | 1 | 0.000 | equality_range:脖子+头 = | S1T146 |
| 脖子和头差不多长。 | 1 | 0.000 | equality_range:脖子+头 = | S1T242 |
| 脖子和尾巴一样长。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T276 |
| 脖子和尾巴长度差不多。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T122 |
| 脖子和腿差不多长。 | 1 | 0.000 | equality_range:脖子+腿 = | S1T139 |
| 腿、头、尾巴差不多长。 | 1 | 0.000 | equality_range:腿+头+尾巴 = | S1T105 |
| 腿、尾巴、脖子差不多长。 | 1 | 0.000 | equality_range:腿+尾巴+脖子 = | S1T110 |
| 腿、尾巴、躯干一样长。 | 1 | 0.000 | body_ref:腿 = 0.50; body_ref:尾巴 = 0.50; equality_range:腿+尾巴 = | S1T199 |
| 腿、脖子、尾巴差不多长。 | 1 | 0.000 | equality_range:腿+脖子+尾巴 = | S1T104 |
| 腿和尾巴一样长。 | 1 | 0.000 | equality_range:腿+尾巴 = | S1T170 |
| 腿和脖子一样长。 | 1 | 0.000 | equality_range:腿+脖子 = | S1T218 |
| 腿和脖子差不多长。 | 1 | 0.000 | equality_range:腿+脖子 = | S1T102 |
| 腿和躯干差不多长。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S1T144 |
| 腿最大。 | 1 | 0.000 | superlative:腿 > 脖子; superlative:腿 > 头; superlative:腿 > 尾巴 | S2T8 |
| 头和脖子最短。 | 1 | 0.333 | superlative:脖子 < 头; superlative:脖子 < 腿 | S3T3 |
| 头和腿最短。 | 1 | 0.333 | superlative:腿 < 头; superlative:腿 < 尾巴 | S3T13 |
| 头和腿最长。 | 1 | 0.333 | superlative:腿 > 脖子; superlative:腿 > 头 | S2T194 |
| 尾巴、腿、躯干一样长。 | 1 | 0.333 | body_ref:尾巴 = 0.50; body_ref:腿 = 0.50 | S1T166 |
| 尾巴和头最长。 | 1 | 0.333 | superlative:头 > 腿; superlative:头 > 尾巴 | S2T229 |
| 脖子和腿最短。 | 1 | 0.333 | superlative:腿 < 脖子; superlative:腿 < 头 | S2T287 |
| 脖子最短。 | 1 | 0.333 | superlative:脖子 < 腿; superlative:脖子 < 尾巴 | S2T127 |
| 腿、脖子、尾巴最短。 | 1 | 0.333 | superlative:尾巴 < 脖子; superlative:尾巴 < 腿 | S1T301 |

### S206

- trial 数: 1408; 非空文本: 1408; fidelity 可评分率: 0.994; 平均 fidelity: 0.892; 完全忠实率: 0.789; 低 fidelity 率: 0.035.
- 旧版 region 覆盖率: 0.994; 旧版 region 有未处理片段率: 0.016.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1403 | 0.996 |
| comparison | 573 | 0.407 |
| superlative | 72 | 0.051 |
| body_ref | 31 | 0.022 |
| equality | 11 | 0.008 |
| other | 3 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，腿长。 | 60 |
| 脖子长，腿短。 | 56 |
| 脖子短，腿短。 | 54 |
| 头长，腿短。 | 51 |
| 脖子长，腿长。 | 50 |
| 脖子短，腿长。 | 48 |
| 头短，腿短。 | 39 |
| 头短，腿长。 | 37 |
| 脖子很长，腿很短。 | 25 |
| 尾巴最短。 | 17 |
| 脖子比头短，腿比尾巴短。 | 16 |
| 脖子很短，腿很长。 | 16 |
| 头、脖子、腿都很长。 | 15 |
| 腿最短。 | 14 |
| 脖子最短。 | 13 |
| 脖子比较短，腿很长。 | 13 |
| 腿很短。 | 12 |
| 脖子比头长，腿比尾巴短。 | 12 |
| 头最短。 | 11 |
| 腿很长，头比脖子短。 | 11 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子比头长，腿比躯干短。 | 4 | body_ref | S4T44, S4T45, S4T47, S4T60 |
| 脖子比头短，腿比躯干短。 | 3 | body_ref | S4T46, S4T48, S4T64 |
| 脖子比头短，腿比躯干长。 | 3 | body_ref | S4T41, S4T59, S4T61 |
| 头比脖子短，腿比躯干短。 | 2 | body_ref | S4T50, S4T96 |
| 头比脖子短，腿比躯干长。 | 2 | body_ref | S4T49, S4T51 |
| 头比脖子长，腿比躯干长。 | 2 | body_ref | S4T52, S4T53 |
| 脖子和腿都比正常躯干要短。 | 2 | body_ref | S3T39, S3T40 |
| 脖子和腿都比躯干短。 | 2 | body_ref | S3T26, S3T27 |
| 脖子比躯干短，腿比躯干长。 | 2 | body_ref | S3T24, S3T42 |
| 像小狗。 | 1 | other | S2T144 |
| 像小狗一样。 | 1 | equality | S2T86 |
| 像食蚁兽。 | 1 | other | S2T92 |
| 四个部位等长。 | 1 | equality | S1T125 |
| 四肢较为等长。 | 1 | equality | S1T26 |
| 头、脖子、尾巴和腿长度相当。 | 1 | equality | S1T18 |
| 头、脖子、腿、尾巴四个部位的长度大致相等。 | 1 | equality | S1T4 |
| 头、脖子、腿、尾巴大致长度相等。 | 1 | equality | S1T12 |
| 头、脖子、腿和尾巴长度相当。 | 1 | equality | S1T33 |
| 头和脖子。 | 1 | other | S1T27 |
| 头比脖子长，腿比躯干短很多。 | 1 | body_ref | S4T78 |
| 头比躯干长，腿比躯干短。 | 1 | body_ref | S4T77 |
| 尾巴、腿、头和脖子长度相当。 | 1 | equality | S1T36 |
| 脖子和头都很长，腿比躯干长。 | 1 | body_ref | S4T43 |
| 脖子和腿都比躯干长。 | 1 | body_ref | S3T43 |
| 脖子比头短，头很长，腿比尾巴要短，头和腿和尾巴都比躯干要短。 | 1 | body_ref | S3T111 |
| 脖子比头长，腿比躯干要长。 | 1 | body_ref | S3T109 |
| 脖子比头长，腿比躯干长。 | 1 | body_ref | S4T42 |
| 脖子比躯干短，腿也比躯干短。 | 1 | body_ref | S3T25 |
| 脖子比躯干要短，腿比躯干要长。 | 1 | body_ref | S3T41 |
| 脖子比较短，头比脖子长，腿和头一样长。 | 1 | equality | S4T313 |
| 较为均衡。 | 1 | equality | S1T158 |
| 较为等长。 | 1 | equality | S1T101 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头长，腿长。 | 8 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T151, S1T178, S1T232, S1T245, S1T255, S1T260, S1T279, S1T309 |
| 脖子短，腿短。 | 4 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S2T189, S2T190, S2T257, S2T295 |
| 脖子长，腿短。 | 3 | 0.000 | absolute_long:脖子 > 0.50; absolute_short:腿 < 0.50 | S2T218, S2T221, S2T245 |
| 脖子短，腿长。 | 2 | 0.000 | absolute_short:脖子 < 0.50; absolute_long:腿 > 0.50 | S2T271, S2T302 |
| 脖子长，腿长。 | 2 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S2T157, S2T165 |
| 脖子和腿比较长，头比脖子短。 | 2 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S5T23, S5T31 |
| 像食蚁兽，脖子很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T64 |
| 四个部位等长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T125 |
| 四肢较为等长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T26 |
| 头、脖子、尾巴和腿长度相当。 | 1 | 0.000 | equality_range:头+脖子+尾巴+腿 = | S1T18 |
| 头、脖子、腿、尾巴四个部位的长度大致相等。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T4 |
| 头、脖子、腿、尾巴大致长度相等。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T12 |
| 头、脖子、腿和尾巴长度相当。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T33 |
| 头和腿都比较短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50 | S2T72 |
| 头和腿都比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T27 |
| 尾巴、腿、头和脖子长度相当。 | 1 | 0.000 | equality_range:尾巴+腿+头+脖子 = | S1T36 |
| 尾巴和腿比较短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50 | S3T280 |
| 尾巴比较短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S2T131 |
| 脖子和腿都很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S4T23 |
| 脖子和腿都比较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S3T23 |
| 脖子很短，腿也很短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S3T46 |
| 脖子很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T19 |
| 脖子很长，腿也很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S3T31 |
| 脖子比较短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S2T61 |
| 脖子比较短，腿很长。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_long:腿 > 0.50 | S3T83 |
| 脖子比较长，腿比较短。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_short:腿 < 0.50 | S3T126 |
| 脖子相对短，腿相对长。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_long:腿 > 0.50 | S3T63 |
| 脖子短但腿长。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S3T175 |
| 脖子长，腿也比较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S3T157 |
| 腿很长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S2T100 |
| 头、脖子、腿都比较长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S4T221 |
| 头比脖子长，头和腿都比较长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S4T231 |
| 腿和脖子比较长，头比脖子短。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S5T123 |
| 腿比较短，头和脖子比较长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S5T75 |

### S207

- trial 数: 960; 非空文本: 959; fidelity 可评分率: 0.996; 平均 fidelity: 0.891; 完全忠实率: 0.775; 低 fidelity 率: 0.056.
- 旧版 region 覆盖率: 0.996; 旧版 region 有未处理片段率: 0.028.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 959 | 0.999 |
| superlative | 474 | 0.494 |
| comparison | 131 | 0.136 |
| equality | 23 | 0.024 |
| ranking | 14 | 0.015 |
| count_abstract | 14 | 0.015 |
| body_ref | 2 | 0.002 |
| empty | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴最长。 | 93 |
| 脖子比头长很多。 | 71 |
| 脖子短，尾巴长。 | 57 |
| 脖子长，头短。 | 57 |
| 脖子最长。 | 50 |
| 脖子短，尾巴短。 | 47 |
| 脖子长，头长。 | 41 |
| 头和腿最长。 | 39 |
| 腿最长。 | 33 |
| 头最长。 | 22 |
| 四个部位都很长。 | 22 |
| 头和尾巴最长。 | 21 |
| 头和脖子最长。 | 19 |
| 头最短。 | 18 |
| 除了腿都很长。 | 16 |
| 腿最短。 | 13 |
| 四个部位长度差不多。 | 10 |
| 脖子最短。 | 10 |
| 四肢都很长。 | 9 |
| 腿和头最长。 | 9 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位长度差不多。 | 10 | equality | S1T94, S1T122, S1T124, S1T236, S1T265, S1T280, S1T308, S2T257 |
| 腿最长，头第二长。 | 4 | ranking | S1T42, S1T170, S2T1, S2T42 |
| 四个部位长度差不多，都比较长。 | 2 | equality | S1T63, S1T114 |
| 腿很短，其他三个部位都很长。 | 2 | count_abstract | S1T185, S2T11 |
| 腿最长，尾巴第二长。 | 2 | ranking | S1T36, S2T36 |
| 四个部位都比较长，长度接近。 | 1 | equality | S1T101 |
| 头和脖子一样长。 | 1 | equality | S2T157 |
| 头和脖子一样长，都是最长的，其他两个比较短。 | 1 | equality | S1T40 |
| 头和腿一样长，脖子最短。 | 1 | equality | S1T16 |
| 头和腿最长，和脖子差不多长。 | 1 | equality | S1T253 |
| 头和腿最长，明显长于其他两个部位。 | 1 | count_abstract | S1T146 |
| 头很长，其他三个部位都非常短。 | 1 | count_abstract | S1T107 |
| 头最短，其他三个部位都很长。 | 1 | count_abstract | S1T47 |
| 头最短，尾巴最长，四个部位都比躯干短。 | 1 | body_ref | S3T81 |
| 头最长，腿和尾巴也很长，和头比较接近，脖子很短。 | 1 | equality | S1T136 |
| 头最长，腿第二长，脖子最短。 | 1 | ranking | S1T129 |
| 尾巴很长，和前三个部位长度差不多。 | 1 | equality, count_abstract | S1T166 |
| 尾巴最长，其他三个部位一样长，都比较长。 | 1 | equality, count_abstract | S1T34 |
| 尾巴最长，头和腿一样长。 | 1 | equality | S1T38 |
| 尾巴最长，头次之。 | 1 | ranking | S1T2 |
| 尾巴最长，头第二长。 | 1 | ranking | S2T2 |
| 尾巴非常长，明显长于其他三个部位。 | 1 | count_abstract | S1T141 |
| 脖子最长，头第二长。 | 1 | ranking | S2T12 |
| 脖子最长，头第二长，其他两个很短。 | 1 | ranking | S2T27 |
| 脖子短，头最长，和躯干一样长，腿最短。 | 1 | equality, body_ref | S3T97 |
| 腿很短，其他三个部位都比较长。 | 1 | count_abstract | S1T93 |
| 腿最短，其他三个部位很长。 | 1 | count_abstract | S1T46 |
| 腿最短，其他三个部位比较长。 | 1 | count_abstract | S1T73 |
| 腿最短，其他三个部位都很长。 | 1 | count_abstract | S1T48 |
| 腿最长，头次之。 | 1 | ranking | S1T1 |
| 腿最长，头第二长，脖子很短。 | 1 | ranking | S1T150 |
| 腿最长，脖子次之。 | 1 | ranking | S1T30 |
| 除了尾巴，其他三个一样长。 | 1 | equality | S2T252 |
| 除了尾巴，其他三个部位都比较长。 | 1 | count_abstract | S1T55 |
| 除了腿，其余三个部位都非常长。 | 1 | count_abstract | S1T135 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 除了腿都很长。 | 16 | 0.000 | absolute_long:腿 > 0.50 | S1T244, S1T245, S1T256, S1T303, S2T29, S2T59, S2T127, S2T128 |
| 四个部位长度差不多。 | 8 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T94, S1T122, S1T124, S1T236, S1T265, S1T280, S1T308, S3T63 |
| 除了尾巴都很长。 | 8 | 0.000 | absolute_long:尾巴 > 0.50 | S2T55, S2T68, S2T98, S2T105, S2T119, S2T121, S2T130, S2T169 |
| 除了脖子都很长。 | 7 | 0.000 | absolute_long:脖子 > 0.50 | S1T270, S1T312, S2T5, S2T10, S2T69, S2T74, S2T90 |
| 四个部位长度差不多，都比较长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T63, S1T114 |
| 除了腿都非常长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S1T176, S1T208 |
| 头和脖子最长。 | 2 | 0.167 | superlative:脖子 > 头; superlative:脖子 > 腿; superlative:脖子 > 尾巴 | S2T45, S2T168 |
| 头和脖子比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T217 |
| 脖子和尾巴都很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T230 |
| 除了尾巴，其他三个一样长。 | 1 | 0.000 | equality_range:脖子+头+腿 = | S2T252 |
| 四个部位都很长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T94 |
| 四个部位都相对较长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T58 |
| 头、腿和尾巴都很长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T242 |
| 头和腿最长。 | 1 | 0.333 | superlative:腿 > 头; superlative:腿 > 尾巴 | S2T58 |
| 尾巴、腿、头最长。 | 1 | 0.333 | superlative:头 > 腿; superlative:头 > 尾巴 | S2T115 |
| 腿和头最长。 | 1 | 0.333 | superlative:头 > 腿; superlative:头 > 尾巴 | S2T83 |

### S208

- trial 数: 576; 非空文本: 571; fidelity 可评分率: 0.964; 平均 fidelity: 0.933; 完全忠实率: 0.811; 低 fidelity 率: 0.023.
- 旧版 region 覆盖率: 0.964; 旧版 region 有未处理片段率: 0.038.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 570 | 0.990 |
| comparison | 101 | 0.175 |
| superlative | 77 | 0.134 |
| equality | 27 | 0.047 |
| empty | 5 | 0.009 |
| meta | 1 | 0.002 |
| count_abstract | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子短，头长。 | 24 |
| 脖子最短。 | 23 |
| 脖子最长。 | 22 |
| 头比脖子长。 | 18 |
| 头和脖子短，腿和尾巴长。 | 17 |
| 脖子长，尾巴短。 | 14 |
| 头和脖子都短。 | 13 |
| 四个部位都长。 | 12 |
| 脖子长，尾巴长。 | 11 |
| 脖子比头短。 | 9 |
| 四个部位都比较长。 | 9 |
| 四个部位都比较短。 | 8 |
| 脖子和尾巴都长。 | 8 |
| 脖子长，其他三个短。 | 8 |
| 除了腿，都很长。 | 7 |
| 除了腿，都长。 | 7 |
| 尾巴短，脖子长。 | 7 |
| 脖子长，其他短。 | 6 |
| 尾巴最长。 | 6 |
| 头和脖子都较短。 | 6 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 两长两短。 | 1 | count_abstract | S1T297 |
| 四者均匀的很长。 | 1 | equality | S1T195 |
| 四者都差不多长。 | 1 | equality | S1T182 |
| 均匀的长。 | 1 | equality | S1T283 |
| 头、脖子、尾巴都一样长、较短，腿很长。 | 1 | equality | S1T70 |
| 头、脖子和尾巴一样长，腿短。 | 1 | equality | S1T61 |
| 头和脖子一样长，腿和尾巴一样长。 | 1 | equality | S1T127 |
| 头和腿一样长，尾巴和脖子一样长。 | 1 | equality | S1T218 |
| 尾巴和脖子一样长，头很长，腿很短。 | 1 | equality | S1T13 |
| 我真的不知道，没什么区别。 | 1 | meta | S1T160 |
| 脖子、腿和尾巴一样长，头较短。 | 1 | equality | S1T5 |
| 脖子和头一样长，尾巴和腿一样长。 | 1 | equality | S1T96 |
| 脖子和尾巴一样长。 | 1 | equality | S1T287 |
| 脖子和尾巴一样长，腿更长。 | 1 | equality | S1T151 |
| 脖子很长，尾巴稍短一些，腿和腿一样长。 | 1 | equality | S1T32 |
| 腿、尾巴和头一样长。 | 1 | equality | S1T100 |
| 腿偏短，头和尾巴一样长，脖子较长。 | 1 | equality | S1T10 |
| 腿偏短，脖子、尾巴和头都偏长，一样长。 | 1 | equality | S1T9 |
| 腿偏短，脖子和尾巴一样长，头很长。 | 1 | equality | S1T7 |
| 腿和尾巴最短，头和脖子一样长。 | 1 | equality | S1T62 |
| 腿和尾巴较短，头和脖子一样长。 | 1 | equality | S1T63 |
| 腿和脖子一样长，头偏长，尾巴偏短。 | 1 | equality | S1T6 |
| 腿很长，脖子和尾巴一样长，头偏短。 | 1 | equality | S1T8 |
| 腿短，头、尾巴、脖子差不多长。 | 1 | equality | S1T23 |
| 腿短，尾巴和脖子一样长，头比脖子稍短。 | 1 | equality | S1T18 |
| 腿短，尾巴长，头和脖子一样长。 | 1 | equality | S1T123 |
| 腿长，头和脖子一样长，尾巴短。 | 1 | equality | S1T60 |
| 长度比较均匀。 | 1 | equality | S1T244 |
| 除了头，都一样长。 | 1 | equality | S2T5 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 除了腿都很长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S1T191, S1T222 |
| 除了腿都比较长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S1T247, S1T248 |
| 头和脖子都较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T35 |
| 腿、尾巴和头一样长。 | 1 | 0.000 | equality_range:腿+尾巴+头 = | S1T100 |
| 除了头之外都较短。 | 1 | 0.000 | absolute_short:头 < 0.50 | S2T118 |
| 除了头，都一样长。 | 1 | 0.000 | equality_range:脖子+腿+尾巴 = | S2T5 |
| 除了尾巴都很长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T231 |
| 除了尾巴都比较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T135 |
| 除了腿都比较短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S2T145 |
| 尾巴短，头和脖子稍长一点。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T130 |
| 腿很长，脖子和尾巴都比较长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T145 |

### S209

- trial 数: 512; 非空文本: 512; fidelity 可评分率: 1.000; 平均 fidelity: 0.969; 完全忠实率: 0.885; 低 fidelity 率: 0.006.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.055.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 512 | 1.000 |
| superlative | 87 | 0.170 |
| comparison | 77 | 0.150 |
| equality | 32 | 0.062 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，脖子长。 | 33 |
| 头短，尾巴短。 | 22 |
| 头长，脖子短。 | 19 |
| 头短，尾巴长。 | 13 |
| 脖子和尾巴长，头和腿短。 | 9 |
| 头短，尾巴中等。 | 7 |
| 尾巴最长。 | 5 |
| 头、脖子和尾巴长，腿短。 | 4 |
| 腿长，尾巴短。 | 4 |
| 头中等偏长，脖子短。 | 3 |
| 头短，尾巴中等偏长。 | 3 |
| 头、脖子、尾巴长，腿短。 | 3 |
| 头和脖子中等偏长。 | 3 |
| 头中等偏短，尾巴长。 | 3 |
| 头最长，脖子中等。 | 3 |
| 腿长，脖子长，头和尾巴中等长度。 | 3 |
| 头和脖子长，腿和尾巴短。 | 3 |
| 头长，脖子中等偏短。 | 3 |
| 头比脖子长。 | 2 |
| 头长，脖子短，腿和尾巴中等长度。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头、脖子、尾巴中等长度，腿短，头和脖子接近。 | 1 | equality | S1T113 |
| 头、脖子、腿都很长，尾巴短，头和脖子长度比较接近。 | 1 | equality | S1T66 |
| 头、脖子和尾巴长，腿短，头和脖子差不多长。 | 1 | equality | S1T84 |
| 头、脖子和尾巴长，腿短，头和脖子接近。 | 1 | equality | S1T180 |
| 头、脖子和尾巴长，腿短，头和脖子长度接近。 | 1 | equality | S1T110 |
| 头、脖子和腿长，尾巴短，头和脖子差不多长。 | 1 | equality | S1T95 |
| 头、脖子和腿长，尾巴短，头和脖子接近。 | 1 | equality | S1T140 |
| 头、脖子和腿长，尾巴短，头和脖子长度接近。 | 1 | equality | S1T71 |
| 头和脖子中等长度，腿和尾巴短，头和脖子差不多长。 | 1 | equality | S1T79 |
| 头和脖子长且接近，腿和尾巴短。 | 1 | equality | S1T197 |
| 头和脖子长且长度接近，腿和尾巴短。 | 1 | equality | S1T143 |
| 头和脖子长，腿短，尾巴中等长度，头和脖子长度比较接近。 | 1 | equality | S1T67 |
| 头和脖子，所有部位都差不多长，其中头和脖子长度接近，并且是中等偏长。 | 1 | equality | S1T107 |
| 头和腿长，脖子和尾巴中等偏长，头和脖子长度相对接近。 | 1 | equality | S1T172 |
| 头短，其余部位中等，尾巴和脖子相对接近。 | 1 | equality | S1T121 |
| 头短，脖子、腿、尾巴中等，脖子和尾巴接近。 | 1 | equality | S1T114 |
| 头长，脖子长并且接近，腿和尾巴中等。 | 1 | equality | S1T118 |
| 头长，脖子长，头和脖子长度接近，腿长，尾巴中等。 | 1 | equality | S1T115 |
| 头长，腿长，脖子和尾巴中等且接近。 | 1 | equality | S1T120 |
| 尾巴最长，腿短，头和脖子中等且接近。 | 1 | equality | S1T189 |
| 所有部位中等长度，脖子和尾巴比较接近。 | 1 | equality | S1T124 |
| 所有部位都是中等偏长，而且长度差不多。 | 1 | equality | S1T18 |
| 所有部位都是中等长度，而且都差不多长。 | 1 | equality | S1T78 |
| 所有部位都是长，头和脖子长度接近。 | 1 | equality | S1T108 |
| 所有部位都长，头和脖子接近。 | 1 | equality | S1T238 |
| 脖子和尾巴长，头和腿短，头和腿差不多长。 | 1 | equality | S1T74 |
| 脖子最长，头和腿长度中等且接近，尾巴中等偏短。 | 1 | equality | S1T69 |
| 脖子长，头和腿长度接近，尾巴短。 | 1 | equality | S1T68 |
| 腿最长，头、脖子、尾巴差不多短。 | 1 | equality | S1T87 |
| 腿短，头短，尾巴和脖子长，尾巴和脖子长度接近。 | 1 | equality | S1T109 |
| 腿长，头和尾巴接近，脖子短。 | 1 | equality | S1T173 |
| 腿长，头短，脖子和尾巴中等，并且接近。 | 1 | equality | S1T117 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子和腿，尾巴长，短。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T58 |
| 头中等偏长，尾巴短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_short:尾巴 < 0.50 | S2T172 |
| 头和脖子，所有部位都差不多长，其中头和脖子长度接近，并且是中等偏长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 =; equality_range:头+脖子 = | S1T107 |

### S210

- trial 数: 1728; 非空文本: 1703; fidelity 可评分率: 0.922; 平均 fidelity: 0.921; 完全忠实率: 0.740; 低 fidelity 率: 0.035.
- 旧版 region 覆盖率: 0.922; 旧版 region 有未处理片段率: 0.069.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1693 | 0.980 |
| superlative | 677 | 0.392 |
| comparison | 420 | 0.243 |
| equality | 309 | 0.179 |
| count_abstract | 89 | 0.052 |
| ranking | 29 | 0.017 |
| empty | 25 | 0.014 |
| body_ref | 17 | 0.010 |
| other | 3 | 0.002 |
| negation | 1 | 0.001 |
| meta | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴最短。 | 136 |
| 腿最长。 | 90 |
| 头和尾巴一样长。 | 60 |
| 头比脖子短。 | 44 |
| 头比脖子长。 | 37 |
| 腿最短。 | 36 |
| 头最长。 | 31 |
| 头最短。 | 27 |
| 只有尾巴短。 | 26 |
| 尾巴最长。 | 25 |
| 四个部位都比较长。 | 24 |
| 头和尾巴长。 | 18 |
| 腿是最长。 | 17 |
| 头是最长。 | 17 |
| 头和脖子长。 | 16 |
| 尾巴明显最短。 | 16 |
| 脖子最短。 | 14 |
| 只有头短。 | 14 |
| 脖子最长。 | 14 |
| 只有腿长。 | 13 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头和尾巴一样长。 | 60 | equality | S1T47, S2T84, S2T94, S2T100, S2T107, S2T108, S2T120, S2T121 |
| 有三个部位几乎一样长。 | 13 | equality, count_abstract | S1T29, S1T106, S1T108, S1T142, S1T193, S1T257, S1T260, S1T262 |
| 三个部位几乎一样长。 | 11 | equality, count_abstract | S1T170, S1T190, S1T223, S1T224, S1T241, S1T290, S1T310, S1T314 |
| 有两个部位几乎一样长。 | 9 | equality, count_abstract | S1T30, S1T38, S1T135, S1T138, S1T141, S1T144, S1T167, S1T259 |
| 头和尾巴长度一样。 | 8 | equality | S2T216, S3T9, S3T44, S3T84, S3T159, S3T160, S3T175, S4T261 |
| 头和腿一样长。 | 8 | equality | S1T53, S2T40, S2T63, S2T130, S2T238, S2T241, S2T309, S3T76 |
| 有三个部位长度一样。 | 7 | equality, count_abstract | S3T189, S3T190, S3T207, S3T210, S3T216, S3T236, S4T216 |
| 头和脖子一样长。 | 6 | equality | S2T60, S2T160, S2T203, S2T228, S2T270, S3T60 |
| 尾巴和脖子一样长。 | 6 | equality | S2T145, S2T253, S2T261, S2T264, S2T276, S2T311 |
| 有三个部位一样长。 | 6 | equality, count_abstract | S1T64, S1T210, S2T199, S3T23, S3T51, S3T108 |
| 脖子和腿一样长。 | 6 | equality | S1T36, S1T71, S2T73, S3T17, S3T73, S3T104 |
| 两个部位一样长。 | 5 | equality, count_abstract | S1T86, S1T90, S1T93, S1T179, S1T180 |
| 两个部位几乎一样长。 | 5 | equality, count_abstract | S1T164, S2T9, S2T76, S2T112, S2T116 |
| 三个部位长度一样。 | 4 | equality, count_abstract | S3T82, S3T83, S3T120, S3T122 |
| 头和尾巴几乎一样长。 | 4 | equality | S1T21, S2T47, S2T83, S3T165 |
| 头和脖子几乎一样长。 | 4 | equality | S1T65, S1T198, S2T6, S2T31 |
| 尾巴和腿一样长。 | 4 | equality | S2T225, S2T249, S3T3, S3T27 |
| 各部位几乎一样长。 | 3 | equality | S1T120, S2T1, S2T16 |
| 四个部位都比躯干短。 | 3 | body_ref | S1T140, S1T149, S1T155 |
| 头和尾巴最接近。 | 3 | equality | S2T170, S2T292, S2T294 |
| 脖子和尾巴一样长。 | 3 | equality | S1T44, S3T29, S3T187 |
| 腿和尾巴一样长。 | 3 | equality | S1T243, S2T289, S2T293 |
| 腿是第二长。 | 3 | ranking | S3T71, S3T78, S3T79 |
| 躯干最长。 | 3 | body_ref | S1T75, S1T78, S1T81 |
| 三个部位一样长。 | 2 | equality, count_abstract | S2T272, S2T278 |
| 三个部位长度相似。 | 2 | equality, count_abstract | S1T278, S4T51 |
| 两两长度一样。 | 2 | equality | S2T214, S2T217 |
| 四个部位一样长。 | 2 | equality | S4T16, S6T1 |
| 四个部位几乎一样长。 | 2 | equality | S1T16, S3T140 |
| 四个部位都一样长。 | 2 | equality | S3T1, S4T210 |
| 四个部位都比躯干要短。 | 2 | body_ref | S1T33, S3T241 |
| 四个部位长度差不多。 | 2 | equality | S3T64, S4T251 |
| 头。 | 2 | other | S4T294, S5T134 |
| 头最短，尾巴其次，腿和脖子最长。 | 2 | ranking | S3T306, S3T307 |
| 头第二短。 | 2 | ranking | S3T56, S3T57 |
| 尾巴和脖子几乎一样长。 | 2 | equality | S1T25, S2T59 |
| 尾巴和腿最长，头其次，脖子最短。 | 2 | ranking | S4T11, S4T12 |
| 尾巴最短，其他部位长度差不多。 | 2 | equality | S3T265, S3T267 |
| 有两个部位一样长。 | 2 | equality, count_abstract | S1T40, S1T255 |
| 脖子和尾巴几乎一样长。 | 2 | equality | S2T44, S3T150 |
| 脖子和腿几乎一样长。 | 2 | equality | S1T23, S2T17 |
| 脖子最长，其他部位差不多。 | 2 | equality | S4T196, S4T200 |
| 腿和脖子一样长。 | 2 | equality | S1T52, S1T113 |
| 腿第二长。 | 2 | ranking | S3T55, S3T130 |
| 躯干是最长。 | 2 | body_ref | S1T39, S1T73 |
| 都差不多长。 | 2 | equality | S2T242, S2T312 |
| 长度两两相似。 | 2 | equality | S3T10, S3T118 |
| 三个部位一样长，头最小。 | 1 | equality, count_abstract | S4T300 |
| 三个部位明显长。 | 1 | count_abstract | S1T161 |
| 三个部位相似，脖子最长。 | 1 | equality, count_abstract | S4T84 |
| 三个部位相似，腿最短。 | 1 | equality, count_abstract | S4T83 |
| 三个部位，长最长，尾巴最短。 | 1 | count_abstract | S4T82 |
| 两两一样长。 | 1 | equality | S2T244 |
| 两两长度相似。 | 1 | equality | S3T162 |
| 两部位一样长。 | 1 | equality | S1T127 |
| 两部位长相似。 | 1 | equality | S1T303 |
| 其他三个部位相似，尾巴最长。 | 1 | equality, count_abstract | S4T150 |
| 其他部位一样长，尾巴最短。 | 1 | equality | S4T209 |
| 其他部位一样长，脖子最短。 | 1 | equality | S4T212 |
| 其他部位一样，头最短。 | 1 | equality | S4T236 |
| 几个部位几乎一样长。 | 1 | equality | S1T235 |
| 几个部位的长度差不多。 | 1 | equality | S3T196 |
| 几个部位都比躯干短。 | 1 | body_ref | S1T58 |
| 几个部位长度一样。 | 1 | equality | S4T140 |
| 又有两个部位一样长。 | 1 | equality, count_abstract | S1T32 |
| 只有头长于躯干。 | 1 | body_ref | S1T85 |
| 只有尾巴比躯干长。 | 1 | body_ref | S1T94 |
| 各部位的长度差不多。 | 1 | equality | S3T232 |
| 各部位都差不多，较短，尾巴最短。 | 1 | equality | S5T74 |
| 各部位长度一样。 | 1 | equality | S3T16 |
| 各部位长度都差不多。 | 1 | equality | S3T223 |
| 四个部位都一样长，都是1/2。 | 1 | equality | S5T1 |
| 四个部位都差不多，头最长。 | 1 | equality | S3T193 |
| 四个部位都很长，且几乎一样长。 | 1 | equality | S5T16 |
| 四个部位都比躯干长。 | 1 | body_ref | S1T130 |
| 四个部位长度一样。 | 1 | equality | S4T120 |
| 四个部位长度都一样。 | 1 | equality | S4T1 |
| 头、尾巴、腿一样长，脖子最长。 | 1 | equality | S3T298 |
| 头、尾巴、腿长度相似，脖子最长。 | 1 | equality | S3T317 |
| 头、尾巴和腿几乎一样长，脖子最短。 | 1 | equality | S3T291 |
| 头、脖子、尾巴几乎一样长，腿最短。 | 1 | equality | S3T305 |
| 头、脖子、尾巴的长度相似，腿最短。 | 1 | equality | S3T255 |
| 头、脖子和尾巴基本一样长，腿最短。 | 1 | equality | S4T29 |
| 头、脖子和腿一样长，尾巴最短。 | 1 | equality | S3T258 |
| 头、脖子和腿长度差不多。 | 1 | equality | S3T272 |
| 头、脖子，头和腿最短，脖子其次，尾巴最长。 | 1 | ranking | S3T278 |
| 头和尾巴一样短，脖子最短，腿最长。 | 1 | equality | S3T302 |
| 头和尾巴一样长，脖子和腿一样长。 | 1 | equality | S3T100 |
| 头和尾巴一样长，脖子和腿一样长，脖子和腿最长。 | 1 | equality | S4T17 |
| 头和尾巴一样长，脖子最短，腿最长。 | 1 | equality | S3T248 |
| 头和尾巴一样长，腿最短，脖子第二长。 | 1 | equality, ranking | S3T261 |
| 头和尾巴一样长，都是最长。 | 1 | equality | S3T102 |
| 头和尾巴相似，腿长于脖子。 | 1 | equality | S4T105 |
| 头和尾巴相等。 | 1 | equality | S4T160 |
| 头和尾巴长度一样，脖子最长，腿最短。 | 1 | equality | S3T252 |
| 头和尾巴长度一样，脖子长于腿。 | 1 | equality | S4T281 |
| 头和尾巴长度一样，都最短，腿第二短，脖子很长。 | 1 | equality, ranking | S4T9 |
| 头和尾巴长度一致。 | 1 | equality | S3T173 |
| 头和脖子几乎一样长，腿比较短，尾巴最短。 | 1 | equality | S4T14 |
| 头和脖子基本一样长，尾巴最长，腿最短。 | 1 | equality | S4T31 |
| 头和脖子最接近。 | 1 | equality | S2T167 |
| 头和脖子长度一样。 | 1 | equality | S2T218 |
| 头和腿几乎一样长，而且最短。 | 1 | equality | S2T32 |
| 头和腿长度相似。 | 1 | equality | S3T13 |
| 头和腿长度相似，脖子和尾巴长度相似。 | 1 | equality | S3T308 |
| 头和躯干几乎一样长。 | 1 | equality, body_ref | S1T66 |
| 头最短，其他部位一样长。 | 1 | equality | S3T259 |
| 头最短，尾巴最长，脖子和腿长度差不多。 | 1 | equality | S3T273 |
| 头最短，脖子最长，尾巴跟腿差不多。 | 1 | equality | S3T300 |
| 头最短，腿其次，脖子和尾巴最长。 | 1 | ranking | S3T271 |
| 头最长，其次是腿，尾巴和脖子差不多。 | 1 | equality, ranking | S3T231 |
| 头最长，尾巴、脖子和腿长度差不多。 | 1 | equality | S3T276 |
| 头最长，尾巴第二短。 | 1 | ranking | S3T134 |
| 头最长，尾巴第二长，腿第三长，脖子第四长。 | 1 | ranking, count_abstract | S3T105 |
| 头未超过躯干。 | 1 | body_ref, negation | S1T105 |
| 头比较短，其他三个部位都比较长。 | 1 | count_abstract | S5T23 |
| 头短，其他部位一样长。 | 1 | equality | S4T199 |
| 头长于其他三个部位。 | 1 | count_abstract | S4T59 |
| 头，第三长。 | 1 | ranking, count_abstract | S2T101 |
| 尾巴、腿和头长度一样。 | 1 | equality | S4T246 |
| 尾巴和头长度一样。 | 1 | equality | S3T235 |
| 尾巴和脖子长度一样。 | 1 | equality | S3T123 |
| 尾巴和腿一样短，头和脖子一样长。 | 1 | equality | S3T244 |
| 尾巴和腿一样长，头最长，脖子最短。 | 1 | equality | S3T294 |
| 尾巴和腿长度一样。 | 1 | equality | S2T219 |
| 尾巴和腿长度相似。 | 1 | equality | S3T180 |
| 尾巴明显最短，其他部位长度差不多。 | 1 | equality | S3T263 |
| 尾巴明显最短，头和脖子差不多，腿最长。 | 1 | equality | S3T279 |
| 尾巴最短，其他三个部位一样长。 | 1 | equality, count_abstract | S4T313 |
| 尾巴最短，其他三个部位相似。 | 1 | equality, count_abstract | S4T87 |
| 尾巴最短，头其次，腿是最长。 | 1 | ranking | S3T275 |
| 尾巴最短，头最长，脖子和腿一样长。 | 1 | equality | S3T243 |
| 尾巴最短，头最长，脖子和腿长的差不多。 | 1 | equality | S3T268 |
| 尾巴最长，其他部位长的差不多。 | 1 | equality | S3T312 |
| 尾巴最长，腿第二长，脖子和头都很短。 | 1 | ranking | S4T6 |
| 尾巴长于脖子，头和腿差不多。 | 1 | equality | S4T115 |
| 所有部位长度一样。 | 1 | equality | S4T259 |
| 有三个部位比躯干长。 | 1 | body_ref, count_abstract | S1T137 |
| 有三个部位都一样长。 | 1 | equality, count_abstract | S3T2 |
| 有三个部位长度一致。 | 1 | equality, count_abstract | S3T172 |
| 有两个部位长度一样。 | 1 | equality, count_abstract | S3T80 |
| 有两个部位长得相似。 | 1 | equality, count_abstract | S1T299 |
| 脖子、头、尾巴长度一样。 | 1 | equality | S4T172 |
| 脖子、尾巴、腿的长度，一样。 | 1 | equality | S3T199 |
| 脖子。 | 1 | other | S2T39 |
| 脖子和尾巴中等长，其他两个部位比较短。 | 1 | count_abstract | S5T44 |
| 脖子和尾巴几乎一样长，头和腿也几乎一样长。 | 1 | equality | S3T149 |
| 脖子和尾巴最短，头其次，腿最长。 | 1 | ranking | S3T310 |
| 脖子和尾巴，几乎一样长。 | 1 | equality | S2T36 |
| 脖子和腿相似，头短于尾巴。 | 1 | equality | S4T104 |
| 脖子和腿相似，头长于尾巴。 | 1 | equality | S4T106 |
| 脖子最短，其他部位一样。 | 1 | equality | S4T232 |
| 脖子最短，其他部位长度差不多。 | 1 | equality | S3T270 |
| 脖子最短，头和尾巴几乎一样长，腿最长。 | 1 | equality | S3T256 |
| 脖子最短，头最长，腿和尾巴长度一样。 | 1 | equality | S4T3 |
| 脖子最短，尾巴其次。 | 1 | ranking | S3T169 |
| 脖子最短，尾巴其次，头和腿最长。 | 1 | ranking | S3T301 |
| 脖子最短，尾巴最长，头和腿长度一样。 | 1 | equality | S4T4 |
| 脖子最长，头第二长，腿和尾巴都比较短，尾巴最短。 | 1 | ranking | S4T7 |
| 腿和尾巴很长，另两个部位非常短。 | 1 | count_abstract | S5T77 |
| 腿和脖子几乎一样长。 | 1 | equality | S1T17 |
| 腿明显最短，其他三个部位都比较长。 | 1 | count_abstract | S3T320 |
| 腿明显最短，头和尾巴几乎一样长，脖子最长。 | 1 | equality | S3T257 |
| 腿明显最短，尾巴是第二短。 | 1 | ranking | S3T152 |
| 腿最短，其他部位一样长。 | 1 | equality | S4T264 |
| 腿最短，其他部位长度差不多。 | 1 | equality | S3T264 |
| 腿最短，尾巴其次，脖子较长，头最长。 | 1 | ranking | S4T18 |
| 腿最短，尾巴第二短，头很长。 | 1 | ranking | S3T96 |
| 腿最长，其他三个部位相似。 | 1 | equality, count_abstract | S4T107 |
| 腿最长，尾巴第二，脖子第三，头最短。 | 1 | ranking | S4T5 |
| 腿最长，脖子最短，头和尾巴差不多。 | 1 | equality | S3T299 |
| 腿最长，脖子最短，头和尾巴长度差不多。 | 1 | equality | S3T277 |
| 腿比较短，其他三个部位中等长。 | 1 | count_abstract | S5T29 |
| 腿，头和尾巴一样长。 | 1 | equality | S3T146 |
| 选错了。 | 1 | meta | S5T316 |
| 长度差不多。 | 1 | equality | S3T229 |
| 长度相似，脖子最短。 | 1 | equality | S4T79 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴一样长。 | 21 | 0.000 | equality_range:头+尾巴 = | S2T107, S2T108, S2T121, S2T165, S2T189, S2T190, S2T224, S2T245 |
| 头和尾巴最接近。 | 3 | 0.000 | equality_range:头+尾巴 = | S2T170, S2T292, S2T294 |
| 四个部位长度差不多。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T64, S4T251 |
| 头和尾巴几乎一样长。 | 2 | 0.000 | equality_range:头+尾巴 = | S2T83, S3T165 |
| 头和脖子一样长。 | 2 | 0.000 | equality_range:头+脖子 = | S2T203, S2T270 |
| 头和腿一样长。 | 2 | 0.000 | equality_range:头+腿 = | S1T53, S2T63 |
| 尾巴和脖子一样长。 | 2 | 0.000 | equality_range:尾巴+脖子 = | S2T261, S2T276 |
| 脖子和腿一样长。 | 2 | 0.000 | equality_range:脖子+腿 = | S1T36, S1T71 |
| 各部位几乎一样长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T120 |
| 各部位的长度差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T232 |
| 各部位长度都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T223 |
| 四个部位长度一样。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S4T120 |
| 头和脖子几乎一样长。 | 1 | 0.000 | equality_range:头+脖子 = | S1T198 |
| 头和脖子最接近。 | 1 | 0.000 | equality_range:头+脖子 = | S2T167 |
| 头和腿几乎一样长，而且最短。 | 1 | 0.000 | equality_range:头+腿 = | S2T32 |
| 头和腿长度相似，脖子和尾巴长度相似。 | 1 | 0.000 | equality_range:头+腿 =; equality_range:脖子+尾巴 = | S3T308 |
| 头和躯干几乎一样长。 | 1 | 0.000 | body_ref:头 = 0.50 | S1T66 |
| 头未超过躯干。 | 1 | 0.000 | body_ref:头 > 0.50 | S1T105 |
| 头比脖子长。 | 1 | 0.000 | comparison:头 > 脖子 | S1T305 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S3T53 |
| 所有部位长度一样。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S4T259 |
| 脖子和尾巴几乎一样长。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S3T150 |
| 脖子和尾巴几乎一样长，头和腿也几乎一样长。 | 1 | 0.000 | equality_range:脖子+尾巴 =; equality_range:头+腿 = | S3T149 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S6T39 |
| 腿长于脖子、头长于尾巴。 | 1 | 0.000 | comparison:腿 > 脖子+头 | S4T103 |
| 都短腿比较长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S5T90 |
| 四个部位都一样长，都是1/2。 | 1 | 0.200 | body_ref:脖子 = 0.50; body_ref:头 = 0.50; body_ref:腿 = 0.50; body_ref:尾巴 = 0.50 | S5T1 |
| 只有尾巴短。 | 1 | 0.250 | exclusive_case:脖子 > 0.50; exclusive_case:头 > 0.50; exclusive_case:腿 > 0.50 | S5T279 |
| 头最长。 | 1 | 0.333 | superlative:头 > 腿; superlative:头 > 尾巴 | S2T277 |
| 头，尾巴最短。 | 1 | 0.333 | superlative:尾巴 < 脖子; superlative:尾巴 < 腿 | S3T168 |
| 尾巴最短，头最长。 | 1 | 0.333 | superlative:尾巴 < 脖子; superlative:尾巴 < 头; superlative:尾巴 < 腿; superlative:头 > 尾巴 | S4T21 |
| 腿是最长。 | 1 | 0.333 | superlative:腿 > 脖子; superlative:腿 > 头 | S1T199 |
| 头长于脖子、长于尾巴、长于腿。 | 1 | 0.400 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; comparison:头+脖子 > 尾巴; absolute_long:尾巴 > 0.50 | S4T28 |

### S211

- trial 数: 960; 非空文本: 959; fidelity 可评分率: 0.998; 平均 fidelity: 0.897; 完全忠实率: 0.818; 低 fidelity 率: 0.064.
- 旧版 region 覆盖率: 0.998; 旧版 region 有未处理片段率: 0.006.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 959 | 0.999 |
| superlative | 287 | 0.299 |
| equality | 127 | 0.132 |
| comparison | 38 | 0.040 |
| ranking | 8 | 0.008 |
| negation | 3 | 0.003 |
| empty | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子最长。 | 63 |
| 头最长。 | 53 |
| 腿最长。 | 47 |
| 尾巴最长。 | 42 |
| 尾巴长。 | 39 |
| 尾巴很长。 | 36 |
| 脖子和尾巴一样长。 | 31 |
| 尾巴长，脖子短。 | 29 |
| 脖子和尾巴长。 | 21 |
| 头和脖子长。 | 21 |
| 脖子和尾巴都长。 | 20 |
| 脖子长。 | 18 |
| 头长。 | 17 |
| 头和尾巴一样长。 | 17 |
| 头和腿长。 | 16 |
| 脖子和腿长。 | 16 |
| 头很长。 | 15 |
| 头和脖子都很长。 | 14 |
| 四个部位都短。 | 13 |
| 只有头长。 | 13 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子和尾巴一样长。 | 31 | equality | S1T65, S1T93, S1T94, S1T107, S1T118, S1T158, S1T237, S1T285 |
| 头和尾巴一样长。 | 17 | equality | S1T25, S1T38, S1T42, S1T85, S1T105, S1T106, S1T121, S1T159 |
| 头和腿一样长。 | 12 | equality | S1T62, S1T80, S1T83, S1T147, S2T48, S2T84, S2T123, S2T164 |
| 头和脖子一样长。 | 10 | equality | S1T22, S1T78, S1T79, S1T97, S1T102, S2T78, S2T86, S2T121 |
| 尾巴和腿一样长。 | 10 | equality | S1T10, S1T63, S2T104, S2T105, S2T127, S2T189, S2T192, S2T212 |
| 脖子和腿一样长。 | 8 | equality | S1T157, S1T176, S2T85, S2T120, S2T136, S2T166, S2T203, S2T282 |
| 头、脖子、尾巴一样长。 | 6 | equality | S1T23, S1T58, S2T187, S2T233, S2T245, S2T246 |
| 头最长，其次是尾巴。 | 4 | ranking | S1T33, S1T34, S1T67, S1T68 |
| 脖子和尾巴一样长，腿很短。 | 3 | equality | S1T5, S1T6, S1T59 |
| 腿和尾巴一样长。 | 3 | equality | S1T32, S2T99, S2T113 |
| 头和尾巴基本一样长。 | 2 | equality | S1T76, S1T77 |
| 头和脖子基本一样长。 | 2 | equality | S1T56, S1T75 |
| 尾巴最长，其次是脖子。 | 2 | ranking | S1T54, S3T54 |
| 只有头和腿长，脖子和尾巴一样短。 | 1 | equality | S3T48 |
| 四个部位都不长。 | 1 | negation | S3T127 |
| 四个部位长度差不多，比较长。 | 1 | equality | S3T57 |
| 头、尾巴、腿基本一样长。 | 1 | equality | S1T43 |
| 头、脖子、尾巴和腿都一样长。 | 1 | equality | S3T3 |
| 头、脖子、尾巴都很长，头、尾巴、腿一样长。 | 1 | equality | S2T124 |
| 头、脖子、腿一样长。 | 1 | equality | S2T45 |
| 头、脖子、腿一样长，尾巴很短。 | 1 | equality | S3T1 |
| 头、脖子和腿基本一样长。 | 1 | equality | S1T45 |
| 头和尾巴一样长，脖子也很长。 | 1 | equality | S2T122 |
| 头和尾巴一样长，腿很短。 | 1 | equality | S1T7 |
| 头和尾巴基本一样长，腿很短。 | 1 | equality | S1T46 |
| 头和尾巴最长一样长。 | 1 | equality | S2T169 |
| 头和尾巴都比较长，一样长，腿也比较长。 | 1 | equality | S2T175 |
| 头和脖子，尾巴和腿中有三个是一样长。 | 1 | equality | S1T57 |
| 头和腿一样长，四个部位都长。 | 1 | equality | S1T60 |
| 头和腿基本一样长，尾巴很短。 | 1 | equality | S1T48 |
| 头最长，腿和尾巴一样长，四个部位都比较长。 | 1 | equality | S2T184 |
| 头长，脖子短，尾巴不太长。 | 1 | negation | S3T235 |
| 头，和腿基本一样长。 | 1 | equality | S1T55 |
| 尾巴和其他部位都一样长。 | 1 | equality | S3T52 |
| 尾巴和脖子、头一样长。 | 1 | equality | S2T186 |
| 脖子、尾巴都比较短，头和脖子基本一样长。 | 1 | equality | S3T149 |
| 脖子和尾巴一样长，四个部位都比较长。 | 1 | equality | S2T171 |
| 脖子最长，其次是头。 | 1 | ranking | S2T287 |
| 脖子最长，其次是尾巴。 | 1 | ranking | S3T13 |
| 脖子长，头和尾巴一样长。 | 1 | equality | S2T163 |
| 都不太长，只有腿最短。 | 1 | negation | S3T58 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子和尾巴一样长。 | 8 | 0.000 | equality_range:脖子+尾巴 = | S1T93, S1T285, S2T83, S2T140, S2T195, S2T224, S2T289, S2T302 |
| 头和尾巴一样长。 | 6 | 0.000 | equality_range:头+尾巴 = | S1T25, S1T38, S1T42, S1T105, S1T121, S2T291 |
| 头、脖子、尾巴一样长。 | 5 | 0.000 | equality_range:头+脖子+尾巴 = | S1T23, S1T58, S2T187, S2T245, S2T246 |
| 头和脖子一样长。 | 3 | 0.000 | equality_range:头+脖子 = | S1T79, S1T97, S2T300 |
| 尾巴长。 | 3 | 0.000 | absolute_long:尾巴 > 0.50 | S1T304, S1T310, S2T34 |
| 脖子和腿一样长。 | 3 | 0.000 | equality_range:脖子+腿 = | S1T157, S2T85, S2T203 |
| 头和腿一样长。 | 2 | 0.000 | equality_range:头+腿 = | S1T80, S1T147 |
| 头和腿长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T165, S1T289 |
| 头很长。 | 2 | 0.000 | absolute_long:头 > 0.50 | S1T232, S1T287 |
| 尾巴和腿一样长。 | 2 | 0.000 | equality_range:尾巴+腿 = | S1T10, S2T285 |
| 头最长。 | 2 | 0.167 | superlative:头 > 脖子; superlative:头 > 腿; superlative:头 > 尾巴 | S2T27, S2T100 |
| 四个部位长度差不多，比较长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T57 |
| 头、尾巴、腿基本一样长。 | 1 | 0.000 | equality_range:头+尾巴+腿 = | S1T43 |
| 头、脖子、尾巴和腿都一样长。 | 1 | 0.000 | equality_range:头+脖子+尾巴+腿 = | S3T3 |
| 头和尾巴基本一样长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T77 |
| 头和尾巴比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S3T46 |
| 头和脖子基本一样长。 | 1 | 0.000 | equality_range:头+脖子 = | S1T56 |
| 头和腿很长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T234 |
| 头和腿都很长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T128 |
| 尾巴和其他部位都一样长。 | 1 | 0.000 | equality_range:脖子+头+腿 = | S3T52 |
| 尾巴和脖子都长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S3T128 |
| 脖子和腿长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S2T15 |
| 脖子很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T288 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T178 |
| 除了头都很长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S3T18 |
| 除了头都长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S3T32 |
| 除了尾巴都很长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S3T39 |
| 除了脖子都很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S3T43 |
| 除了腿都很长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S3T19 |
| 头、尾巴、腿、脖子都长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S2T57 |
| 尾巴长，脖子长，头短。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S3T119 |
| 脖子长，尾巴长，头长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50 | S3T195 |
| 四个部位都比较长，只有腿短。 | 1 | 0.375 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50; exclusive_case:脖子 > 0.50 | S3T56 |
| 头和腿一样长，四个部位都长。 | 1 | 0.400 | equality_range:头+腿 =; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T60 |

### S212

- trial 数: 1792; 非空文本: 1782; fidelity 可评分率: 0.975; 平均 fidelity: 0.912; 完全忠实率: 0.816; 低 fidelity 率: 0.028.
- 旧版 region 覆盖率: 0.975; 旧版 region 有未处理片段率: 0.076.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1748 | 0.975 |
| comparison | 751 | 0.419 |
| body_ref | 211 | 0.118 |
| superlative | 188 | 0.105 |
| equality | 35 | 0.020 |
| other | 28 | 0.016 |
| empty | 10 | 0.006 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴长，头长。 | 67 |
| 尾巴长，头短。 | 47 |
| 脖子长，尾巴短。 | 42 |
| 脖子和尾巴都比躯干短。 | 40 |
| 脖子比尾巴长。 | 39 |
| 脖子最长。 | 39 |
| 脖子短，尾巴短。 | 38 |
| 尾巴短，脖子长。 | 38 |
| 脖子短，尾巴长。 | 38 |
| 尾巴最长。 | 32 |
| 尾巴短，脖子短。 | 28 |
| 尾巴比脖子长。 | 27 |
| 头比脖子长。 | 26 |
| 脖子长，尾巴长。 | 25 |
| 尾巴短，头长。 | 25 |
| 脖子较长。 | 24 |
| 脖子和尾巴都比躯干长。 | 21 |
| 脖子比躯干短，尾巴比躯干长。 | 21 |
| 头比尾巴长，脖子比腿长。 | 20 |
| 尾巴短，头短。 | 19 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子和尾巴都比躯干短。 | 40 | body_ref | S4T111, S4T112, S4T117, S4T118, S4T119, S4T123, S5T1, S5T3 |
| 脖子和尾巴都比躯干长。 | 21 | body_ref | S4T115, S4T116, S4T121, S5T7, S5T10, S5T26, S5T35, S5T38 |
| 脖子比躯干短，尾巴比躯干长。 | 21 | body_ref | S4T113, S4T114, S4T120, S4T131, S5T6, S5T13, S5T15, S5T16 |
| 四个部位较均等。 | 17 | other | S1T61, S1T65, S1T68, S1T72, S1T93, S1T119, S1T124, S1T126 |
| 脖子比躯干长，尾巴比躯干短。 | 16 | body_ref | S4T122, S4T124, S4T125, S4T130, S5T2, S5T5, S5T9, S5T11 |
| 脖子和尾巴一样长。 | 13 | equality | S2T99, S2T233, S2T249, S2T250, S2T255, S2T262, S2T263, S2T264 |
| 尾巴比躯干短，头比脖子长。 | 10 | body_ref | S5T98, S5T101, S5T107, S5T108, S5T110, S5T113, S5T118, S5T122 |
| 四个部位较匀称。 | 6 | other | S1T182, S1T242, S1T246, S1T255, S1T259, S2T24 |
| 尾巴比躯干短，脖子比头长。 | 6 | body_ref | S5T102, S5T103, S5T106, S5T111, S5T114, S5T117 |
| 四个部位都比躯干短。 | 5 | body_ref | S4T233, S4T234, S4T235, S4T241, S4T242 |
| 尾巴和脖子一样长。 | 5 | equality | S2T278, S2T283, S2T284, S2T289, S2T310 |
| 四个部位较均衡。 | 4 | equality | S1T76, S1T137, S1T147, S2T295 |
| 尾巴比脖子短，尾巴比躯干短。 | 4 | body_ref | S3T300, S3T311, S3T312, S3T313 |
| 尾巴比脖子长，尾巴比躯干长。 | 4 | body_ref | S3T299, S3T301, S3T305, S3T309 |
| 尾巴比躯干长，头比脖子长。 | 4 | body_ref | S5T112, S5T116, S5T119, S5T120 |
| 四个部位较为匀称。 | 3 | other | S1T236, S1T274, S1T278 |
| 头和尾巴比躯干短。 | 3 | body_ref | S4T237, S4T238, S4T239 |
| 尾巴比躯干长，头比脖子短。 | 3 | body_ref | S5T99, S5T100, S5T121 |
| 脖子、尾巴比躯干短。 | 3 | body_ref | S4T126, S4T128, S4T129 |
| 腿比躯干长。 | 3 | body_ref | S2T129, S2T131, S2T154 |
| 头比躯干长。 | 2 | body_ref | S2T157, S2T160 |
| 尾巴比脖子长，尾巴比躯干短。 | 2 | body_ref | S3T308, S3T310 |
| 尾巴比躯干短，比脖子短。 | 2 | body_ref | S3T316, S3T318 |
| 尾巴比躯干短，脖子和头比躯干长。 | 2 | body_ref | S4T257, S5T84 |
| 尾巴比躯干长，尾巴比脖子长。 | 2 | body_ref | S3T298, S3T307 |
| 尾巴比躯干长，脖子比头长。 | 2 | body_ref | S5T104, S5T115 |
| 尾巴比躯干长，脖子比躯干短。 | 2 | body_ref | S5T93, S5T97 |
| 尾巴比躯干长，脖子比躯干长。 | 2 | body_ref | S5T81, S5T82 |
| 尾巴比躯干长，腿比躯干长。 | 2 | body_ref | S4T245, S4T247 |
| 脖子和尾巴、腿一样长。 | 2 | equality | S2T229, S2T230 |
| 脖子和尾巴比躯干短。 | 2 | body_ref | S4T157, S4T160 |
| 脖子和尾巴比躯干长。 | 2 | body_ref | S4T158, S4T159 |
| 脖子和腿都比躯干短。 | 2 | body_ref | S4T255, S4T256 |
| 脖子比尾巴长，尾巴比躯干短。 | 2 | body_ref | S3T296, S3T302 |
| 脖子比躯干长，腿比躯干短。 | 2 | body_ref | S4T252, S4T254 |
| 腿比躯干短，脖子比躯干长。 | 2 | body_ref | S4T226, S5T94 |
| 四个部位均等。 | 1 | other | S1T9 |
| 四个部位都很小。 | 1 | other | S1T85 |
| 四个部位都比躯干长。 | 1 | body_ref | S4T232 |
| 四个部位长度均衡。 | 1 | equality | S1T35 |
| 四个部位长度较均衡。 | 1 | equality | S1T48 |
| 头、尾巴和脖子一样长。 | 1 | equality | S2T260 |
| 头、脖子比躯干短。 | 1 | body_ref | S4T240 |
| 头和尾巴一样长。 | 1 | equality | S2T274 |
| 头和尾巴比躯干长。 | 1 | body_ref | S4T236 |
| 头和尾巴都比躯干短。 | 1 | body_ref | S4T231 |
| 头和尾巴长度均衡。 | 1 | equality | S1T44 |
| 头和脖子一样。 | 1 | equality | S2T280 |
| 头和脖子一样长。 | 1 | equality | S2T137 |
| 头和脖子比躯干短，腿比躯干长。 | 1 | body_ref | S4T253 |
| 头和脖子比躯干长。 | 1 | body_ref | S4T156 |
| 头和脖子都比躯干短，尾巴比躯干长。 | 1 | body_ref | S5T85 |
| 头和脖子都比躯干长，尾巴比躯干短。 | 1 | body_ref | S5T86 |
| 头和腿比躯干长。 | 1 | body_ref | S5T37 |
| 头比躯干短。 | 1 | body_ref | S2T158 |
| 头比躯干短，尾巴比躯干短。 | 1 | body_ref | S4T229 |
| 头比躯干短，尾巴比躯干长。 | 1 | body_ref | S4T230 |
| 头比躯干长，尾巴比躯干长。 | 1 | body_ref | S4T228 |
| 尾巴和脖子比躯干长。 | 1 | body_ref | S4T250 |
| 尾巴和腿比躯干短。 | 1 | body_ref | S4T243 |
| 尾巴和腿比躯干长。 | 1 | body_ref | S5T36 |
| 尾巴比脖子短，比躯干短。 | 1 | body_ref | S3T315 |
| 尾巴比脖子短，比躯干长。 | 1 | body_ref | S3T317 |
| 尾巴比脖子长，比躯干长。 | 1 | body_ref | S3T320 |
| 尾巴比躯干短，头比脖子短。 | 1 | body_ref | S5T109 |
| 尾巴比躯干短，尾巴比脖子长。 | 1 | body_ref | S3T306 |
| 尾巴比躯干短，比脖子长。 | 1 | body_ref | S3T319 |
| 尾巴比躯干短，脖子比头短。 | 1 | body_ref | S5T105 |
| 尾巴比躯干短，脖子比躯干长。 | 1 | body_ref | S5T83 |
| 尾巴比躯干短，腿比躯干长。 | 1 | body_ref | S4T246 |
| 尾巴比躯干长。 | 1 | body_ref | S4T251 |
| 尾巴比躯干长，头比躯干短。 | 1 | body_ref | S4T249 |
| 脖子、尾巴一样长。 | 1 | equality | S2T190 |
| 脖子、尾巴比躯干长。 | 1 | body_ref | S4T127 |
| 脖子、尾巴都比躯干长。 | 1 | body_ref | S4T133 |
| 脖子、腿、尾巴比躯干短。 | 1 | body_ref | S4T248 |
| 脖子和尾巴一样。 | 1 | equality | S2T266 |
| 脖子和尾巴一样长，比腿长。 | 1 | equality | S2T243 |
| 脖子和尾巴和腿一样长。 | 1 | equality | S2T242 |
| 脖子比尾巴短，尾巴比躯干长。 | 1 | body_ref | S3T314 |
| 脖子比尾巴长，尾巴比躯干长。 | 1 | body_ref | S3T304 |
| 脖子比躯干短，尾巴比躯干短，尾巴比脖子长。 | 1 | body_ref | S3T297 |
| 脖子比躯干短，脖子比尾巴短，尾巴比躯干长。 | 1 | body_ref | S3T303 |
| 脖子比躯干长，尾巴比躯干长。 | 1 | body_ref | S5T19 |
| 腿比躯干短，脖子比躯干短。 | 1 | body_ref | S4T227 |
| 腿比躯干长，尾巴比躯干短。 | 1 | body_ref | S4T244 |
| 腿比躯干长，脖子比躯干短。 | 1 | body_ref | S4T225 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位较均衡。 | 3 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T137, S1T147, S2T295 |
| 尾巴较长。 | 3 | 0.000 | absolute_long:尾巴 > 0.50 | S1T178, S1T187, S2T48 |
| 脖子和尾巴一样长。 | 2 | 0.000 | equality_range:脖子+尾巴 = | S2T264, S2T301 |
| 脖子和尾巴都比躯干短。 | 2 | 0.000 | body_ref:脖子 < 0.50; body_ref:尾巴 < 0.50 | S5T71, S5T77 |
| 四个部位都比躯干短。 | 2 | 0.125 | body_ref:头 < 0.50; body_ref:腿 < 0.50; body_ref:尾巴 < 0.50; body_ref:脖子 < 0.50 | S4T233, S4T235 |
| 四个部位长度均衡。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T35 |
| 四个部位长度较均衡。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T48 |
| 头、尾巴和脖子一样长。 | 1 | 0.000 | equality_range:头+尾巴+脖子 = | S2T260 |
| 头和尾巴一样长。 | 1 | 0.000 | equality_range:头+尾巴 = | S2T274 |
| 头和脖子比尾巴长。 | 1 | 0.000 | comparison:头+脖子 > 尾巴 | S5T67 |
| 头比尾巴短。 | 1 | 0.000 | comparison:头 < 尾巴 | S2T125 |
| 头比脖子短，比尾巴短。 | 1 | 0.000 | comparison:头 < 脖子 | S4T162 |
| 头长，尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S5T316 |
| 尾巴和脖子一样长。 | 1 | 0.000 | equality_range:尾巴+脖子 = | S2T278 |
| 尾巴和腿比躯干短。 | 1 | 0.000 | body_ref:尾巴 < 0.50; body_ref:腿 < 0.50 | S4T243 |
| 尾巴比脖子短，比头短。 | 1 | 0.000 | comparison:尾巴 < 脖子 | S3T265 |
| 尾巴比腿长。 | 1 | 0.000 | comparison:尾巴 > 腿 | S2T200 |
| 尾巴短，头长。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_long:头 > 0.50 | S6T125 |
| 尾巴较短，脖子较长。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_long:脖子 > 0.50 | S5T142 |
| 尾巴长，头长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50 | S5T262 |
| 尾巴长，脖子短。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_short:脖子 < 0.50 | S6T88 |
| 特别尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S2T134 |
| 脖子、尾巴一样长。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S2T190 |
| 脖子和头长于尾巴。 | 1 | 0.000 | comparison:脖子+头 > 尾巴 | S2T269 |
| 脖子和尾巴、腿一样长。 | 1 | 0.000 | equality_range:脖子+尾巴+腿 = | S2T229 |
| 脖子和尾巴一样长，比腿长。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S2T243 |
| 脖子和尾巴都比躯干长。 | 1 | 0.000 | body_ref:脖子 > 0.50; body_ref:尾巴 > 0.50 | S5T38 |
| 脖子比头长。 | 1 | 0.000 | comparison:脖子 > 头 | S2T107 |
| 脖子比尾巴短。 | 1 | 0.000 | comparison:脖子 < 尾巴 | S3T246 |
| 脖子比尾巴长。 | 1 | 0.000 | comparison:脖子 > 尾巴 | S5T74 |
| 脖子短，腿短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S3T233 |
| 脖子较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T41 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T162 |
| 脖子长，尾巴长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S5T208 |
| 腿短，头短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S4T278 |
| 腿较短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S2T39 |
| 腿较长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T55 |
| 头、脖子、尾巴依次增长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S4T193 |
| 头、脖子、尾巴较长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T174 |
| 头和脖子都比躯干长，尾巴比躯干短。 | 1 | 0.333 | body_ref:头 > 0.50; body_ref:尾巴 < 0.50 | S5T86 |
| 头和腿和尾巴是最长。 | 1 | 0.333 | superlative:尾巴 > 头; superlative:尾巴 > 腿 | S1T33 |
| 尾巴比脖子短，比躯干短。 | 1 | 0.333 | body_ref:尾巴 < 0.50; body_ref:脖子 < 0.50 | S3T315 |
| 尾巴长，脖子比腿长，头比尾巴长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; comparison:头 > 尾巴 | S3T32 |

### S213

- trial 数: 1536; 非空文本: 1535; fidelity 可评分率: 0.986; 平均 fidelity: 0.911; 完全忠实率: 0.783; 低 fidelity 率: 0.028.
- 旧版 region 覆盖率: 0.986; 旧版 region 有未处理片段率: 0.020.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1534 | 0.999 |
| comparison | 956 | 0.622 |
| body_ref | 36 | 0.023 |
| equality | 31 | 0.020 |
| count_abstract | 24 | 0.016 |
| superlative | 15 | 0.010 |
| negation | 9 | 0.006 |
| ranking | 3 | 0.002 |
| empty | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头比脖子长，尾巴比较长。 | 43 |
| 头比脖子长，尾巴比较短。 | 42 |
| 头长，脖子长，腿长，尾巴短。 | 29 |
| 腿和脖子都很长。 | 24 |
| 腿比较短，脖子比较长。 | 24 |
| 腿和脖子都比较长。 | 20 |
| 只有脖子比较短。 | 20 |
| 腿和脖子都非常长。 | 18 |
| 脖子比头长，尾巴比较长。 | 18 |
| 脖子和尾巴都比较长。 | 18 |
| 脖子和尾巴都很短。 | 18 |
| 脖子和尾巴都比较短。 | 17 |
| 脖子比头长，腿比较短。 | 17 |
| 脖子比头长，尾巴比较短。 | 16 |
| 头和尾巴都比较长。 | 15 |
| 脖子比头长，腿比较长。 | 15 |
| 腿比较短，脖子和尾巴都比较长。 | 15 |
| 头和尾巴都非常长。 | 14 |
| 腿和尾巴都很长。 | 14 |
| 脖子比头长，腿比尾巴长。 | 14 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 有两个部位比较长。 | 7 | count_abstract | S2T306, S2T309, S2T310, S2T312, S2T314, S2T317, S2T318 |
| 四个部位差不多长。 | 4 | equality | S1T84, S1T209, S1T212, S1T317 |
| 头长，其他三个部位中等。 | 3 | count_abstract | S1T223, S1T255, S1T256 |
| 有三个部位比较长。 | 3 | count_abstract | S2T305, S2T311, S2T316 |
| 腿比躯干长，头比脖子长。 | 3 | body_ref | S2T18, S2T23, S2T28 |
| 尾巴比躯干长。 | 2 | body_ref | S3T243, S3T249 |
| 尾巴比较长，脖子和头一样长。 | 2 | equality | S2T268, S2T270 |
| 尾巴第二短。 | 2 | ranking | S1T305, S1T306 |
| 有一个部位比较长。 | 2 | count_abstract | S2T308, S2T315 |
| 脖子比头长，脖子不是最长。 | 2 | negation | S2T297, S2T301 |
| 腿比躯干短，头比脖子长。 | 2 | body_ref | S2T13, S2T21 |
| 腿短，尾巴长，头和脖子差不多长。 | 2 | equality | S1T318, S1T319 |
| 腿长，其他三个部位中等长度。 | 2 | count_abstract | S1T287, S1T296 |
| 四个身体部位都比较长。 | 1 | body_ref | S3T89 |
| 四个部位都不短，头比脖子长。 | 1 | negation | S1T273 |
| 四部位差不多长。 | 1 | equality | S1T56 |
| 头与脖子差不多长，尾巴较长。 | 1 | equality | S2T92 |
| 头与脖子差不多长，腿较长。 | 1 | equality | S2T80 |
| 头和尾巴差不多长，腿短。 | 1 | equality | S1T113 |
| 头和脖子一样长。 | 1 | equality | S1T257 |
| 头和脖子均比躯干长，尾巴和腿都很短。 | 1 | body_ref | S2T7 |
| 头和脖子均比躯干长，腿比躯干长。 | 1 | body_ref | S2T14 |
| 头和脖子均长于躯干，尾巴较短。 | 1 | body_ref | S2T1 |
| 头和脖子差不多长。 | 1 | equality | S4T15 |
| 头比脖子短，腿比躯干短。 | 1 | body_ref | S2T17 |
| 头比脖子短，腿比躯干长。 | 1 | body_ref | S2T183 |
| 头比脖子长，头不是最长。 | 1 | negation | S2T298 |
| 头比脖子长，尾巴没有特别长。 | 1 | negation | S3T210 |
| 头比脖子长，脖子不是最短。 | 1 | negation | S2T300 |
| 头比脖子长，腿比躯干短。 | 1 | body_ref | S2T16 |
| 头比躯干低。 | 1 | body_ref | S1T63 |
| 头比较短，其他三个部位都比较长。 | 1 | count_abstract | S3T93 |
| 头短，其他三个部位中等。 | 1 | count_abstract | S1T222 |
| 尾巴比其他三个部位都短很多。 | 1 | count_abstract | S3T125 |
| 尾巴比躯干长，头比脖子长。 | 1 | body_ref | S3T306 |
| 尾巴第二长。 | 1 | ranking | S1T307 |
| 脖子与头差不多长，腿比较长。 | 1 | equality | S3T180 |
| 脖子比头长，尾巴比躯干长。 | 1 | body_ref | S2T164 |
| 脖子比头长，腿比躯干长。 | 1 | body_ref | S2T168 |
| 脖子比头长，都比躯干长。 | 1 | body_ref | S2T257 |
| 脖子比较长，其他三个部位都比较短。 | 1 | count_abstract | S3T82 |
| 脖子比较长，腿比躯干短。 | 1 | body_ref | S2T181 |
| 脖子短于躯干，头较短。 | 1 | body_ref | S2T2 |
| 腿、尾巴、头、脖子都差不多长。 | 1 | equality | S4T150 |
| 腿中等长度，头和脖子差不多长。 | 1 | equality | S2T50 |
| 腿和尾巴一样长。 | 1 | equality | S1T109 |
| 腿和尾巴都比躯干短，头和脖子差不多长。 | 1 | equality, body_ref | S2T9 |
| 腿很长，其他三个部位差不多长。 | 1 | equality, count_abstract | S1T272 |
| 腿比躯干短，头和脖子差不多长。 | 1 | equality, body_ref | S2T22 |
| 腿比躯干短，脖子比头长。 | 1 | body_ref | S2T20 |
| 腿比躯干短，脖子比躯干短。 | 1 | body_ref | S2T26 |
| 腿比躯干长，其他部位均比躯干短。 | 1 | body_ref | S2T25 |
| 腿比躯干长，头不比脖子短。 | 1 | body_ref | S2T24 |
| 腿比躯干长，头和脖子均比较长。 | 1 | body_ref | S2T6 |
| 腿比躯干长，头和脖子差不多长。 | 1 | equality, body_ref | S2T15 |
| 腿比躯干长，头比脖子长，尾巴也不短。 | 1 | body_ref, negation | S2T27 |
| 腿比躯干长，脖子比头长。 | 1 | body_ref | S2T19 |
| 腿比躯干长，脖子非常长。 | 1 | body_ref | S2T8 |
| 腿比较短，尾巴比较短，头和脖子差不多长。 | 1 | equality | S1T310 |
| 腿比较短，脖子和尾巴都不算短。 | 1 | negation | S4T247 |
| 腿比较长，其余三个身体部位都比较短。 | 1 | body_ref | S3T188 |
| 腿比较长，其余三个部位都比较短。 | 1 | count_abstract | S3T213 |
| 腿比较长，四个部位差不多。 | 1 | equality | S1T157 |
| 腿比较长，头比脖子长，脖子短于躯干。 | 1 | body_ref | S2T37 |
| 腿比较长，尾巴长于躯干，头比脖子长。 | 1 | body_ref | S2T36 |
| 腿短，其他三个差不多长。 | 1 | equality | S1T236 |
| 腿短，头和脖子一样长。 | 1 | equality | S1T185 |
| 腿短，头和脖子一样长，尾巴长。 | 1 | equality | S1T186 |
| 腿较短，尾巴较长，头和脖子差不多长。 | 1 | equality | S2T40 |
| 腿长于躯干，头比脖子短。 | 1 | body_ref | S2T3 |
| 腿长，其他三个部位都很短。 | 1 | count_abstract | S1T275 |
| 腿长，头和脖子一样长。 | 1 | equality | S1T263 |
| 腿长，头和脖子差不多长，尾巴不短。 | 1 | equality, negation | S1T291 |
| 腿长，尾巴长，头和脖子一样长。 | 1 | equality | S1T271 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿比较短，脖子和尾巴都比较长。 | 5 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50; absolute_short:腿 < 0.50 | S4T275, S4T319, S5T4, S5T27, S5T34 |
| 四个部位差不多长。 | 4 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T84, S1T209, S1T212, S1T317 |
| 脖子和尾巴都比较长。 | 2 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S5T165, S5T188 |
| 除了尾巴都很长。 | 2 | 0.000 | absolute_long:尾巴 > 0.50 | S1T117, S1T118 |
| 腿比较短，头和脖子都比较长。 | 2 | 0.167 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_short:腿 < 0.50 | S4T106, S4T125 |
| 头、脖子、尾巴都比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S3T62 |
| 头、脖子和尾巴均比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T62 |
| 头和脖子一样长。 | 1 | 0.000 | equality_range:头+脖子 = | S1T257 |
| 头比躯干低。 | 1 | 0.000 | body_ref:头 < 0.50 | S1T63 |
| 头比较长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S3T79 |
| 头长，脖子长，尾巴长，腿和尾巴。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T91 |
| 尾巴和头都比较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50 | S3T83 |
| 脖子和头都比较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S5T106 |
| 脖子比头长，腿比尾巴短。 | 1 | 0.000 | comparison:脖子 > 头; comparison:腿 < 尾巴 | S2T216 |
| 脖子比较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S5T160 |
| 腿、尾巴、头、脖子都差不多长。 | 1 | 0.000 | equality_range:腿+尾巴+头+脖子 = | S4T150 |
| 腿和尾巴一样长。 | 1 | 0.000 | equality_range:腿+尾巴 = | S1T109 |
| 腿和尾巴比较长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S2T56 |
| 腿比较长，头和脖子无明显差距。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S3T167 |
| 除了头都比较长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T227 |
| 除了脖子都很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S4T227 |
| 头长，脖子长，尾巴长，腿短。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T276 |
| 头长，脖子长，腿短，尾巴长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T62 |
| 腿比较长，头、脖子、尾巴比较长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S5T15 |
| 头和脖子比较长，腿比较短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S3T309 |
| 尾巴、脖子、头都很长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S5T76 |
| 脖子和头比较长，尾巴比较短。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S3T106 |
| 脖子比头长，尾巴和腿都比较长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50 | S2T135 |
| 腿很短，脖子和尾巴都比较长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S4T62 |
| 腿很长，脖子和尾巴都比较长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S4T207 |
| 腿比较短，尾巴、脖子都很长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S4T165 |
| 腿短，尾巴长，头和脖子差不多长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; equality_range:头+脖子 = | S1T319 |
| 四个部位都不短，头比脖子长。 | 1 | 0.400 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T273 |

### S214

- trial 数: 1792; 非空文本: 1724; fidelity 可评分率: 0.935; 平均 fidelity: 0.895; 完全忠实率: 0.769; 低 fidelity 率: 0.052.
- 旧版 region 覆盖率: 0.935; 旧版 region 有未处理片段率: 0.030.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1624 | 0.906 |
| equality | 76 | 0.042 |
| empty | 68 | 0.038 |
| comparison | 41 | 0.023 |
| meta | 39 | 0.022 |
| superlative | 12 | 0.007 |
| body_ref | 9 | 0.005 |
| count_abstract | 3 | 0.002 |
| other | 2 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子短，腿长。 | 183 |
| 脖子短，腿短。 | 146 |
| 脖子长。 | 129 |
| 头短，尾巴短。 | 127 |
| 腿短。 | 39 |
| 尾巴短。 | 39 |
| 选错了。 | 39 |
| 各部位差不多。 | 37 |
| 脖子长，尾巴长。 | 34 |
| 四个部位都长。 | 33 |
| 脖子短。 | 31 |
| 腿长，脖子短。 | 29 |
| 脖子长，尾巴短。 | 28 |
| 头短，脖子长。 | 28 |
| 腿长。 | 24 |
| 脖子长，头短，尾巴长。 | 24 |
| 尾巴特别短。 | 23 |
| 腿特别长。 | 20 |
| 腿短，尾巴短。 | 19 |
| 四个部位差不多。 | 19 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 选错了。 | 39 | meta | S1T97, S1T131, S1T148, S1T166, S1T249, S1T311, S2T27, S2T53 |
| 各部位差不多。 | 37 | equality | S1T185, S1T187, S1T191, S1T195, S1T206, S1T232, S1T260, S1T268 |
| 四个部位差不多。 | 19 | equality | S2T223, S3T88, S3T169, S3T170, S3T192, S3T204, S3T247, S3T295 |
| 四个部位差不多长。 | 6 | equality | S1T129, S1T136, S1T140, S1T143, S1T150, S1T155 |
| 脖子比躯干长。 | 5 | body_ref | S4T158, S4T233, S4T234, S4T235, S4T240 |
| 两长两短。 | 3 | count_abstract | S3T220, S3T221, S3T222 |
| 差不多。 | 3 | equality | S2T35, S2T110, S4T18 |
| 各部位差不多，头略长。 | 2 | equality | S1T243, S1T246 |
| 脖子比躯干长，尾巴比躯干短。 | 2 | body_ref | S4T156, S4T157 |
| 各部位差不多长。 | 1 | equality | S1T179 |
| 各部位差不多，头短，脖子长。 | 1 | equality | S1T203 |
| 各部位差不多，尾巴和脖子略长。 | 1 | equality | S2T2 |
| 各部位差不多，脖子略长。 | 1 | equality | S1T199 |
| 各部位差不多，脖子长。 | 1 | equality | S1T202 |
| 四个部位差不多长，腿略长一点。 | 1 | equality | S1T151 |
| 四个部位都还行。 | 1 | other | S5T246 |
| 头和腿略长，其他差不多。 | 1 | equality | S1T252 |
| 差距不大。 | 1 | other | S3T189 |
| 所有躯干部位均略长，除了尾巴中等长度。 | 1 | body_ref | S1T56 |
| 脖子比躯干短，腿比躯干长。 | 1 | body_ref | S4T239 |
| 脖子长，其他地方差不多。 | 1 | equality | S1T128 |
| 腿短，头和脖子一样长。 | 1 | equality | S2T180 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 各部位差不多。 | 35 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T185, S1T187, S1T191, S1T195, S1T206, S1T232, S1T260, S1T268 |
| 四个部位差不多。 | 15 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T223, S3T88, S3T169, S3T170, S3T192, S3T204, S3T295, S4T2 |
| 脖子长。 | 8 | 0.000 | absolute_long:脖子 > 0.50 | S1T156, S4T48, S4T108, S4T116, S4T209, S5T23, S5T138, S5T222 |
| 四个部位差不多长。 | 6 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T129, S1T136, S1T140, S1T143, S1T150, S1T155 |
| 头短，尾巴短。 | 6 | 0.000 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S4T125, S4T151, S4T266, S4T283, S5T151, S5T192 |
| 尾巴长。 | 3 | 0.000 | absolute_long:尾巴 > 0.50 | S1T314, S2T52, S5T122 |
| 各部位差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T179 |
| 各部位差不多，头略长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 =; absolute_long:头 > 0.50 | S1T243 |
| 头短，脖子短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50 | S4T206 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S3T244 |
| 尾巴短，脖子长。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_long:脖子 > 0.50 | S4T86 |
| 尾巴短，腿短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50 | S4T24 |
| 脖子比躯干长。 | 1 | 0.000 | body_ref:脖子 > 0.50 | S4T158 |
| 脖子略短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S1T288 |
| 脖子略长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T223 |
| 脖子略长，尾巴略长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T60 |
| 脖子短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S3T186 |
| 脖子短，尾巴短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50 | S3T298 |
| 脖子短，腿短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S5T111 |
| 脖子稍短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S2T288 |
| 脖子还挺长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S4T169 |
| 头和脖子长，腿短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T198 |
| 头长、脖子长、腿短。 | 1 | 0.333 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50 | S2T226 |
| 脖子短，尾巴和腿短。 | 1 | 0.333 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50 | S6T26 |
| 脖子长，头长，尾巴短。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S6T4 |
| 脖子长，头长，尾巴长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S6T129 |

### S215

- trial 数: 768; 非空文本: 763; fidelity 可评分率: 0.947; 平均 fidelity: 0.910; 完全忠实率: 0.773; 低 fidelity 率: 0.016.
- 旧版 region 覆盖率: 0.947; 旧版 region 有未处理片段率: 0.051.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 763 | 0.993 |
| comparison | 52 | 0.068 |
| superlative | 45 | 0.059 |
| count_abstract | 34 | 0.044 |
| equality | 19 | 0.025 |
| empty | 5 | 0.007 |
| negation | 4 | 0.005 |
| group_sum | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿短，尾巴长。 | 120 |
| 腿长，尾巴长。 | 103 |
| 腿长，尾巴短。 | 99 |
| 腿短，尾巴短。 | 85 |
| 腿长，脖子长。 | 41 |
| 腿长，脖子短。 | 28 |
| 四个部位都长。 | 21 |
| 腿最长。 | 14 |
| 腿长，头长。 | 12 |
| 腿短，头长。 | 9 |
| 腿短，尾巴长，脖子比头长。 | 8 |
| 两个部位长，两个部位短。 | 7 |
| 腿长，头短。 | 7 |
| 腿和尾巴长度一致，头和脖子长度一致。 | 6 |
| 三个部位长，一个部位短。 | 6 |
| 尾巴最长。 | 6 |
| 三个部位长。 | 6 |
| 脖子长，头短。 | 6 |
| 头长，脖子短。 | 5 |
| 腿长，尾巴长，脖子比头长。 | 5 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 两个部位长，两个部位短。 | 7 | count_abstract | S2T52, S2T62, S2T63, S2T87, S2T225, S2T232, S2T233 |
| 三个部位长。 | 6 | count_abstract | S2T35, S2T50, S2T184, S2T188, S2T189, S2T194 |
| 三个部位长，一个部位短。 | 6 | count_abstract | S2T51, S2T57, S2T58, S2T64, S2T231, S2T236 |
| 腿和尾巴长度一致，头和脖子长度一致。 | 6 | equality | S2T65, S2T68, S2T70, S2T76, S2T79, S2T81 |
| 三个部位短，一个部位长。 | 3 | count_abstract | S2T59, S2T86, S2T237 |
| 三长一短。 | 3 | count_abstract | S2T175, S2T238, S2T240 |
| 两个部位长。 | 3 | count_abstract | S2T185, S2T193, S2T195 |
| 腿和尾巴长度一致，头和脖子长度不一致。 | 3 | equality | S2T66, S2T83, S2T85 |
| 腿和尾巴长度不一致，头和脖子长度不一致。 | 3 | equality | S2T69, S2T82, S2T84 |
| 两长两短。 | 2 | count_abstract | S2T173, S2T174 |
| 只有一个部位长。 | 2 | count_abstract | S2T183, S2T187 |
| 腿和尾巴长度不一致，头和脖子长度一致。 | 2 | equality | S2T77, S2T78 |
| 一个部位长，三个部位短。 | 1 | count_abstract | S2T229 |
| 三长一短，头最短。 | 1 | count_abstract | S2T176 |
| 腿和尾巴不一样长，头比脖子长。 | 1 | equality, negation | S2T282 |
| 腿和尾巴不一样，头和脖子不一样长。 | 1 | equality, negation | S2T284 |
| 腿和尾巴不够长。 | 1 | negation | S1T266 |
| 腿和尾巴加起来等于脖子的长度。 | 1 | equality, group_sum | S1T13 |
| 腿和尾巴都长，头和脖子不一样长。 | 1 | equality, negation | S2T280 |
| 腿和脖子一样长。 | 1 | equality | S1T8 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿和尾巴长度一致，头和脖子长度不一致。 | 3 | 0.000 | equality_range:腿+尾巴 =; equality_range:头+脖子 = | S2T66, S2T83, S2T85 |
| 腿和尾巴长度不一致，头和脖子长度不一致。 | 3 | 0.000 | equality_range:腿+尾巴 =; equality_range:头+脖子 = | S2T69, S2T82, S2T84 |
| 腿和尾巴长度不一致，头和脖子长度一致。 | 2 | 0.000 | equality_range:腿+尾巴 =; equality_range:头+脖子 = | S2T77, S2T78 |
| 腿和尾巴不够长。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T266 |
| 腿和尾巴长度一致，头和脖子长度一致。 | 1 | 0.000 | equality_range:腿+尾巴 =; equality_range:头+脖子 = | S2T65 |
| 腿长，尾巴长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T28 |
| 腿长，脖子长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S3T127 |

### S216

- trial 数: 768; 非空文本: 765; fidelity 可评分率: 0.988; 平均 fidelity: 0.901; 完全忠实率: 0.732; 低 fidelity 率: 0.025.
- 旧版 region 覆盖率: 0.988; 旧版 region 有未处理片段率: 0.023.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 764 | 0.995 |
| comparison | 130 | 0.169 |
| superlative | 104 | 0.135 |
| equality | 45 | 0.059 |
| negation | 4 | 0.005 |
| ranking | 4 | 0.005 |
| empty | 3 | 0.004 |
| other | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴比脖子和腿长。 | 35 |
| 尾巴长，腿短。 | 24 |
| 脖子比尾巴和腿长。 | 23 |
| 每个部位都长。 | 23 |
| 每个部位都挺长。 | 20 |
| 脖子最长。 | 13 |
| 尾巴最长。 | 13 |
| 腿长，脖子长，尾巴短。 | 13 |
| 腿长，尾巴长。 | 13 |
| 尾巴长，腿短，脖子短。 | 12 |
| 每个部位都很长。 | 12 |
| 腿长，尾巴短。 | 10 |
| 腿长，尾巴和脖子短。 | 10 |
| 脖子和腿长，尾巴短。 | 9 |
| 尾巴和腿长，脖子短。 | 9 |
| 每个部位都短。 | 9 |
| 尾巴和腿都长。 | 9 |
| 腿长，尾巴长，脖子短。 | 8 |
| 腿最长。 | 8 |
| 脖子和尾巴短，头和腿长。 | 8 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 尾巴和腿一样长，脖子短。 | 4 | equality | S2T124, S2T134, S2T186, S2T233 |
| 每个部位差不多长。 | 3 | equality | S1T232, S1T239, S1T249 |
| 尾巴和腿一样长。 | 2 | equality | S1T295, S2T285 |
| 尾巴和腿差不多长。 | 2 | equality | S1T316, S2T244 |
| 尾巴和腿差不多长，都比脖子长。 | 2 | equality | S2T121, S2T122 |
| 每个部位都一样长。 | 2 | equality | S2T140, S2T182 |
| 脖子和腿一样长，尾巴短。 | 2 | equality | S2T136, S2T149 |
| 腿和脖子一样长，尾巴短。 | 2 | equality | S2T131, S2T191 |
| 腿最长，脖子和尾巴差不多长。 | 2 | equality | S2T137, S2T138 |
| 四个部位都差不多长。 | 1 | equality | S2T161 |
| 头、脖子和尾巴一样长，腿短。 | 1 | equality | S1T146 |
| 尾巴、腿和脖子一样长，头更长一点。 | 1 | equality | S1T147 |
| 尾巴和脖子一样长。 | 1 | equality | S2T127 |
| 尾巴和腿一样长，脖子短一点。 | 1 | equality | S2T291 |
| 尾巴和腿差不多长，偏短。 | 1 | equality | S1T319 |
| 尾巴和腿差不多长，头短，脖子短。 | 1 | equality | S1T267 |
| 尾巴和腿差不多长，脖子短。 | 1 | equality | S2T240 |
| 尾巴和腿都一样长，偏短。 | 1 | equality | S1T294 |
| 尾巴和腿都长，脖子也不短。 | 1 | negation | S2T117 |
| 尾巴最长，脖子、腿和头都不长。 | 1 | negation | S2T92 |
| 尾巴跟腿一样短，脖子长，头长。 | 1 | equality | S2T74 |
| 每个部位。 | 1 | other | S1T105 |
| 每个部位都不短，尾巴最长。 | 1 | negation | S2T297 |
| 每个部位都差不多长。 | 1 | equality | S1T240 |
| 每个部位都很均匀的短。 | 1 | equality | S2T239 |
| 每个部分都差不多长。 | 1 | equality | S2T267 |
| 脖子、尾巴、腿差不多长。 | 1 | equality | S2T98 |
| 脖子和尾巴一样长。 | 1 | equality | S1T235 |
| 脖子和尾巴一样长，腿短。 | 1 | equality | S2T145 |
| 脖子最长，尾巴其次，腿最短。 | 1 | ranking | S2T148 |
| 脖子最长，腿其次，尾巴最短。 | 1 | ranking | S2T103 |
| 脖子最长，腿其次，尾巴短。 | 1 | ranking | S2T150 |
| 脖子最长，腿和尾巴一样长。 | 1 | equality | S2T104 |
| 脖子短，头、腿、尾巴一样长。 | 1 | equality | S1T148 |
| 脖子跟腿差不多长，尾巴最短。 | 1 | equality | S2T79 |
| 脖子长，腿和尾巴一样长。 | 1 | equality | S2T36 |
| 腿和尾巴一样长，偏长。 | 1 | equality | S2T37 |
| 腿和尾巴一样长，脖子短。 | 1 | equality | S2T147 |
| 腿和尾巴都不短。 | 1 | negation | S1T304 |
| 腿最长，脖子其次，尾巴最短。 | 1 | ranking | S2T102 |
| 都差不多长。 | 1 | equality | S2T214 |
| 都差不多长，中等长度。 | 1 | equality | S2T292 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 每个部位差不多长。 | 3 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T232, S1T239, S1T249 |
| 尾巴和腿一样长。 | 2 | 0.000 | equality_range:尾巴+腿 = | S1T295, S2T285 |
| 每个部位都一样长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T140, S2T182 |
| 四个部位都差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T161 |
| 尾巴和腿差不多长。 | 1 | 0.000 | equality_range:尾巴+腿 = | S2T244 |
| 尾巴和腿差不多长，都比脖子长。 | 1 | 0.000 | equality_range:尾巴+腿 = | S2T121 |
| 每个部位都很均匀的短。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T239 |
| 脖子、尾巴、腿差不多长。 | 1 | 0.000 | equality_range:脖子+尾巴+腿 = | S2T98 |
| 脖子和尾巴都挺，腿都挺长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S2T206 |
| 脖子比腿，腿和尾巴长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S2T83 |
| 每个部位都很长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T235 |
| 每个部位都长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S3T18 |
| 腿特别长，头、脖子、尾巴都中等偏长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T48 |
| 脖子和腿长，尾巴短。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S2T226 |
| 腿长，尾巴长，脖子短。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S3T29 |

### S217

- trial 数: 1408; 非空文本: 1407; fidelity 可评分率: 0.998; 平均 fidelity: 0.853; 完全忠实率: 0.712; 低 fidelity 率: 0.060.
- 旧版 region 覆盖率: 0.998; 旧版 region 有未处理片段率: 0.004.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1407 | 0.999 |
| comparison | 158 | 0.112 |
| superlative | 58 | 0.041 |
| body_ref | 34 | 0.024 |
| equality | 5 | 0.004 |
| count_abstract | 3 | 0.002 |
| ranking | 3 | 0.002 |
| negation | 1 | 0.001 |
| empty | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子长，尾巴短。 | 73 |
| 尾巴长，脖子短。 | 66 |
| 尾巴短，脖子长。 | 48 |
| 头和脖子长。 | 47 |
| 头和腿长。 | 42 |
| 脖子短，尾巴长。 | 41 |
| 尾巴短，腿长。 | 41 |
| 尾巴长，脖子长。 | 40 |
| 脖子和尾巴长。 | 38 |
| 腿和尾巴长。 | 37 |
| 尾巴短，腿短。 | 34 |
| 脖子长，尾巴长。 | 32 |
| 脖子和尾巴都短。 | 27 |
| 尾巴长。 | 23 |
| 尾巴短。 | 22 |
| 脖子和尾巴都长。 | 21 |
| 脖子短，尾巴短。 | 19 |
| 脖子和尾巴短。 | 18 |
| 尾巴短，脖子短。 | 15 |
| 腿短，尾巴短。 | 13 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 只有尾巴比躯干长。 | 6 | body_ref | S1T59, S1T68, S1T70, S1T208, S2T1, S2T4 |
| 四个部位都比躯干短。 | 6 | body_ref | S1T54, S1T55, S1T58, S1T67, S2T2, S2T6 |
| 所有部位都比躯干短。 | 5 | body_ref | S1T203, S1T204, S1T205, S1T206, S1T207 |
| 只有脖子比躯干长。 | 4 | body_ref | S1T56, S1T63, S1T71, S2T3 |
| 尾巴长，有两个部位短。 | 2 | count_abstract | S4T74, S4T75 |
| 只有头比躯干长。 | 1 | body_ref | S1T57 |
| 头、脖子和腿都比躯干长。 | 1 | body_ref | S1T69 |
| 头和尾巴差不多长，脖子和腿都比较短。 | 1 | equality | S1T138 |
| 头和脖子比躯干长。 | 1 | body_ref | S1T62 |
| 头和脖子比躯干长，头很长。 | 1 | body_ref | S1T66 |
| 头和腿比躯干长。 | 1 | body_ref | S1T61 |
| 头很长，腿第二长，脖子和尾巴中等。 | 1 | ranking | S1T26 |
| 头最长，腿和尾巴差不多，脖子很短。 | 1 | equality | S1T137 |
| 头比较短，其他部位差不多长。 | 1 | equality | S1T83 |
| 尾巴和脖子都比躯干长。 | 1 | body_ref | S2T5 |
| 尾巴比其他部位都短，其他部位差不多一样长。 | 1 | equality | S1T79 |
| 尾巴比躯干长，腿比躯干短。 | 1 | body_ref | S2T39 |
| 尾巴比较短，头也不算很长，其他部位都很长。 | 1 | negation | S1T12 |
| 尾巴短，腿比躯干长。 | 1 | body_ref | S4T85 |
| 尾巴长，都比躯干短。 | 1 | body_ref | S4T127 |
| 脖子和尾巴比躯干长。 | 1 | body_ref | S1T64 |
| 腿和头比躯干长。 | 1 | body_ref | S1T53 |
| 腿和尾巴都很长，另外两个部位都比较短。 | 1 | count_abstract | S1T10 |
| 腿和脖子比躯干长。 | 1 | body_ref | S1T60 |
| 腿很长，尾巴第二长，脖子中等，头比较短。 | 1 | ranking | S1T29 |
| 腿很长，脖子第二长，头和尾巴很短。 | 1 | ranking | S1T27 |
| 腿最长，脖子最短，尾巴和头差不多长。 | 1 | equality | S1T194 |
| 都比躯干短。 | 1 | body_ref | S4T126 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 由长到短是脖子、腿、头、尾巴。 | 5 | 0.200 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50; absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S1T106, S1T115, S1T119, S1T159, S1T160 |
| 由长到短是腿、尾巴、头、脖子。 | 5 | 0.250 | absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50 | S1T118, S1T149, S1T169, S1T230, S1T268 |
| 由长到短是腿、尾巴、脖子、头。 | 5 | 0.250 | absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T76, S1T91, S1T109, S1T171, S1T270 |
| 脖子和尾巴长。 | 4 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T224, S2T245, S2T290, S3T107 |
| 由长到短是尾巴、脖子、头、腿。 | 4 | 0.250 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S1T104, S1T144, S1T148, S1T186 |
| 由长到短是脖子、头、尾巴、腿。 | 3 | 0.083 | absolute_short:脖子 < 0.50; absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50 | S1T116, S1T214, S1T224 |
| 由长到短是腿、脖子、头、尾巴。 | 3 | 0.250 | absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S1T78, S1T113, S1T168 |
| 头和脖子长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T184, S2T302 |
| 头和腿长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T287, S2T315 |
| 腿和尾巴长。 | 2 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S2T185, S2T205 |
| 除了尾巴以外都很长。 | 2 | 0.000 | absolute_long:尾巴 > 0.50 | S2T18, S2T78 |
| 除了尾巴都很长。 | 2 | 0.000 | absolute_long:尾巴 > 0.50 | S1T312, S2T129 |
| 由长到短是头、腿、尾巴、脖子。 | 2 | 0.125 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T84, S1T163 |
| 由长到短是头、尾巴、脖子、腿。 | 2 | 0.250 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T73, S1T221 |
| 由长到短是尾巴、脖子、腿、头。 | 2 | 0.250 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T211, S1T228 |
| 由长到短是脖子、尾巴、头、腿。 | 2 | 0.250 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50 | S1T117, S1T162 |
| 头和尾巴比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T184 |
| 头和腿略长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T126 |
| 头短，尾巴短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S3T197 |
| 头长，脖子长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S3T235 |
| 尾巴长，有两个部位短。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S4T75 |
| 最短的是尾巴。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T245 |
| 由长到短是头、尾巴、腿和脖子。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50 | S1T238 |
| 由长到短是头、腿、脖子，尾巴。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50 | S1T240 |
| 由长到短是头和腿，脖子和尾巴。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50 | S1T226 |
| 由长到短是尾巴、头、脖子、腿。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50; absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T187 |
| 由长到短是尾巴和腿、脖子和头。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S1T151 |
| 由长到短是脖子、头、腿、尾巴。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:头 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T136 |
| 脖子短，尾巴长。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_long:尾巴 > 0.50 | S4T244 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T198 |
| 脖子长，尾巴长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S3T96 |
| 腿和头比较长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S2T135 |
| 腿比较长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S2T158 |
| 除了脖子都挺长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T309 |
| 除了腿以外都很长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S2T33 |
| 头比较短，其他部位差不多长。 | 1 | 0.250 | complement:脖子 < 0.50; complement:腿 < 0.50; complement:尾巴 < 0.50 | S1T83 |
| 尾巴比其他部位都短，其他部位差不多一样长。 | 1 | 0.250 | complement:脖子 < 0.50; complement:头 < 0.50; complement:腿 < 0.50 | S1T79 |
| 所有部位都比躯干短。 | 1 | 0.250 | body_ref:脖子 < 0.50; body_ref:头 < 0.50; body_ref:腿 < 0.50 | S1T205 |
| 有长到的是头、脖子、尾巴、腿。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50 | S1T122 |
| 由上到短是头、脖子、腿、尾巴。 | 1 | 0.250 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T125 |
| 由长到短是头、尾巴、腿、脖子。 | 1 | 0.250 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50 | S1T135 |
| 由长到短是头、脖子、尾巴、腿。 | 1 | 0.250 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50 | S1T150 |
| 由长到短是头、脖子、腿、尾巴。 | 1 | 0.250 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T99 |
| 由长到短是头、腿、脖子、尾巴。 | 1 | 0.250 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50 | S1T129 |
| 由长到短是尾巴、头、腿、脖子。 | 1 | 0.250 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50; absolute_short:腿 < 0.50 | S1T145 |
| 由长到短是尾巴、腿、脖子、头。 | 1 | 0.250 | absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50 | S1T146 |
| 由长到短是脖子、尾巴、腿、头。 | 1 | 0.250 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50 | S1T157 |
| 由长到短是脖子、腿、尾巴、头。 | 1 | 0.250 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T132 |
| 由长到短是脖子和头、尾巴和腿。 | 1 | 0.250 | absolute_short:脖子 < 0.50; absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S1T87 |
| 由长到短是腿、头、脖子、尾巴。 | 1 | 0.250 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50; absolute_short:脖子 < 0.50 | S1T141 |
| 由长到短是头、腿、尾巴、脖子，脖子很短。 | 1 | 0.400 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T110 |
| 由长到短是头、腿、脖子、尾巴，尾巴非常短。 | 1 | 0.400 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50 | S1T82 |
| 由长到短是脖子、尾巴、头、腿，腿很短。 | 1 | 0.400 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50 | S1T80 |

### S218

- trial 数: 1664; 非空文本: 1658; fidelity 可评分率: 0.978; 平均 fidelity: 0.923; 完全忠实率: 0.772; 低 fidelity 率: 0.019.
- 旧版 region 覆盖率: 0.978; 旧版 region 有未处理片段率: 0.065.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1656 | 0.995 |
| superlative | 570 | 0.343 |
| comparison | 502 | 0.302 |
| equality | 39 | 0.023 |
| body_ref | 7 | 0.004 |
| empty | 6 | 0.004 |
| count_abstract | 2 | 0.001 |
| group_sum | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴长，腿长。 | 99 |
| 尾巴长，腿短。 | 94 |
| 脖子明显长于尾巴。 | 85 |
| 尾巴短，脖子长。 | 73 |
| 尾巴短，脖子短。 | 53 |
| 脖子和尾巴明显长于腿。 | 49 |
| 脖子和尾巴明显短于腿。 | 37 |
| 四个部位都较长。 | 35 |
| 尾巴明显长于脖子。 | 31 |
| 脖子明显短于尾巴。 | 21 |
| 脖子长，尾巴短。 | 18 |
| 脖子、尾巴明显长于腿。 | 12 |
| 四个部位都较短。 | 11 |
| 腿最短，其余部位较长。 | 9 |
| 腿最长。 | 9 |
| 脖子短，尾巴短。 | 9 |
| 腿短，尾巴长。 | 8 |
| 除头外，其余三部位都较长。 | 7 |
| 脖子最短，其他部位较长。 | 7 |
| 脖子最长，其余部位较短。 | 7 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位长度较相近。 | 2 | equality | S1T144, S1T157 |
| 四个部位略短于躯干。 | 1 | body_ref | S4T62 |
| 四个部位较相近，脖子、尾巴稍短。 | 1 | equality | S1T162 |
| 四个部位较相近，腿最长。 | 1 | equality | S1T145 |
| 四个部位较相近，腿稍短。 | 1 | equality | S2T77 |
| 四个部位都略短于躯干。 | 1 | body_ref | S3T158 |
| 四个部位都略长，长度相近。 | 1 | equality | S3T101 |
| 四个部位都约等于躯干。 | 1 | equality, body_ref | S4T44 |
| 四个部位都较短，且明显短于躯干。 | 1 | body_ref | S3T144 |
| 四个部位都较短，且略短于躯干。 | 1 | body_ref | S3T146 |
| 四个部位都较短，且长度较相近。 | 1 | equality | S3T136 |
| 四个部位都较长，且长度较相近。 | 1 | equality | S3T137 |
| 四个部位长度较相近，且与躯干相当。 | 1 | equality, body_ref | S3T140 |
| 头、尾巴长度相近且较短。 | 1 | equality | S1T142 |
| 头、脖子、尾巴比例长于腿。 | 1 | group_sum | S4T242 |
| 头、脖子、腿、尾巴长度相近，且长度中等。 | 1 | equality | S1T44 |
| 头、脖子长度较相近且较长，腿较短。 | 1 | equality | S1T95 |
| 头和脖子较短，腿和尾巴较长且相等。 | 1 | equality | S1T70 |
| 头明显长于其他三个部位。 | 1 | count_abstract | S4T2 |
| 头较短，脖子、腿、尾巴长度相近且较长。 | 1 | equality | S1T51 |
| 尾巴、腿、脖子长度相近且中等，腿较短。 | 1 | equality | S1T81 |
| 尾巴明显长于其他三个部位。 | 1 | count_abstract | S3T116 |
| 尾巴最长，其他部位长度相近。 | 1 | equality | S1T117 |
| 尾巴略长于脖子，脖子约等于腿长。 | 1 | equality | S4T51 |
| 尾巴略长，脖子、腿短于躯干。 | 1 | body_ref | S4T80 |
| 尾巴约等于腿，远长于脖子。 | 1 | equality | S4T43 |
| 脖子、腿长度相等且长于尾巴。 | 1 | equality | S4T111 |
| 脖子和尾巴略短于腿，且都较长，长度相近。 | 1 | equality | S4T5 |
| 脖子和尾巴长度较相等。 | 1 | equality | S1T122 |
| 脖子明显短于尾巴，尾巴与腿长度相当。 | 1 | equality | S3T263 |
| 脖子明显短于尾巴，尾巴和腿长度相当。 | 1 | equality | S3T255 |
| 脖子明显长于尾巴，且脖子长度与腿相当。 | 1 | equality | S3T192 |
| 脖子明显长于尾巴，尾巴约等于腿长。 | 1 | equality | S4T70 |
| 脖子明显长于尾巴，尾巴长度约等于腿。 | 1 | equality | S4T6 |
| 脖子明显长于腿，腿与尾巴长度相当。 | 1 | equality | S3T269 |
| 脖子最长，腿最短，头和尾巴长度相近。 | 1 | equality | S1T141 |
| 脖子略短于尾巴，尾巴长度和腿相当。 | 1 | equality | S3T200 |
| 脖子较短，尾巴、腿长度相当且最长。 | 1 | equality | S2T255 |
| 脖子较短，腿、尾巴长度较相近。 | 1 | equality | S1T132 |
| 脖子长度约等于腿，且明显长于尾巴。 | 1 | equality | S4T25 |
| 脖子长约等于腿，且明显大于尾巴。 | 1 | equality | S4T134 |
| 脖子长，尾巴长，腿长度相近且较长。 | 1 | equality | S3T16 |
| 腿、尾巴长度相近且中等，腿较短，脖子较长。 | 1 | equality | S1T62 |
| 腿、脖子、腿、尾巴长度相近，脖子较长。 | 1 | equality | S1T67 |
| 腿长度中等，头与腿长度相近，脖子较短，尾巴最短。 | 1 | equality | S1T10 |
| 长度较相近且长度中等。 | 1 | equality | S1T231 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴长，腿长。 | 3 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50 | S5T242, S5T255, S5T278 |
| 四个部位长度较相近。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T144, S1T157 |
| 脖子、尾巴较长，且明显长于腿。 | 2 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S3T47, S3T86 |
| 脖子和尾巴较长。 | 2 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T232, S3T212 |
| 脖子明显长于腿。 | 2 | 0.000 | comparison:脖子 > 腿 | S4T84, S4T92 |
| 四个部位较相近，脖子、尾巴稍短。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 =; absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50 | S1T162 |
| 四个部位较相近，腿稍短。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 =; absolute_short:腿 < 0.50 | S2T77 |
| 头、脖子、腿、尾巴长度相近，且长度中等。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T44 |
| 尾巴短，脖子短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S5T244 |
| 尾巴较长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T284 |
| 脖子、尾巴明显长于腿。 | 1 | 0.000 | comparison:脖子+尾巴 > 腿 | S3T233 |
| 脖子、尾巴都明显短于腿，脖子略长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S3T194 |
| 脖子、腿长度相等且长于尾巴。 | 1 | 0.000 | equality_range:脖子+腿+尾巴 = | S4T111 |
| 脖子和尾巴明显长于腿。 | 1 | 0.000 | comparison:脖子+尾巴 > 腿 | S3T312 |
| 脖子和尾巴明显长于腿，明显短于腿。 | 1 | 0.000 | comparison:脖子+尾巴 > 腿 | S4T99 |
| 脖子和尾巴较短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50 | S3T49 |
| 腿、脖子、腿、尾巴长度相近，脖子较长。 | 1 | 0.000 | equality_range:腿+脖子+尾巴 =; absolute_long:脖子 > 0.50 | S1T67 |
| 四个部位略短于躯干。 | 1 | 0.250 | body_ref:脖子 < 0.50; body_ref:腿 < 0.50; body_ref:尾巴 < 0.50 | S4T62 |
| 四个部位都稍短。 | 1 | 0.250 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S4T285 |
| 四个部位都约等于躯干。 | 1 | 0.250 | body_ref:脖子 = 0.50; body_ref:头 = 0.50; body_ref:腿 = 0.50 | S4T44 |
| 腿较短，头、脖子、尾巴较长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T84 |
| 脖子最长，腿稍短，头、尾巴较短。 | 1 | 0.333 | superlative:脖子 > 腿; superlative:脖子 > 尾巴; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S2T188 |
| 头最短，尾巴轻，尾巴较短，脖子较短。 | 1 | 0.400 | superlative:头 < 脖子; superlative:头 < 腿; superlative:头 < 尾巴 | S2T20 |
| 脖子、尾巴稍短，头、腿最长。 | 1 | 0.400 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50; superlative:腿 > 头 | S2T66 |
| 脖子和尾巴较长，腿和头最短。 | 1 | 0.400 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50; superlative:头 < 腿 | S2T86 |

### S219

- trial 数: 1024; 非空文本: 1021; fidelity 可评分率: 0.959; 平均 fidelity: 0.904; 完全忠实率: 0.789; 低 fidelity 率: 0.044.
- 旧版 region 覆盖率: 0.959; 旧版 region 有未处理片段率: 0.050.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 980 | 0.957 |
| comparison | 196 | 0.191 |
| equality | 49 | 0.048 |
| superlative | 16 | 0.016 |
| group_sum | 11 | 0.011 |
| other | 8 | 0.008 |
| negation | 6 | 0.006 |
| empty | 3 | 0.003 |
| meta | 2 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头和尾巴短。 | 88 |
| 头和腿短。 | 70 |
| 头长，腿短。 | 58 |
| 脖子和尾巴短。 | 43 |
| 尾巴短。 | 33 |
| 脖子短。 | 32 |
| 脖子和腿短。 | 24 |
| 头和尾巴比较短。 | 20 |
| 头和脖子短。 | 19 |
| 比较均衡。 | 19 |
| 腿短。 | 15 |
| 尾巴长。 | 14 |
| 头长。 | 14 |
| 头短。 | 13 |
| 头微短。 | 12 |
| 四个部位都比较长。 | 12 |
| 脖子比较短。 | 12 |
| 尾巴比较短。 | 12 |
| 只有腿短。 | 12 |
| 头很长。 | 12 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 比较均衡。 | 19 | equality | S1T63, S1T69, S1T75, S1T76, S1T78, S1T160, S1T189, S1T256 |
| 四个部位比较均衡。 | 8 | equality | S1T35, S1T41, S1T44, S1T46, S1T49, S1T123, S2T15, S2T141 |
| 四个部位长度差不多。 | 4 | equality | S1T104, S1T117, S1T243, S2T262 |
| 四个部位长度均衡。 | 3 | equality | S1T13, S1T15, S1T23 |
| 均衡。 | 3 | equality | S2T49, S2T104, S2T123 |
| 头和尾巴。 | 3 | other | S1T93, S1T124, S1T309 |
| 头和尾巴加起来很长。 | 2 | group_sum | S3T119, S3T120 |
| 头和腿加起来短。 | 2 | group_sum | S3T198, S3T199 |
| 选错了。 | 2 | meta | S3T97, S3T137 |
| 四个部位都中等，比较均匀。 | 1 | equality | S3T23 |
| 四个部位长度差不多，腿有点长。 | 1 | equality | S1T119 |
| 头、脖子和尾巴加起来很长。 | 1 | group_sum | S3T131 |
| 头位呀。 | 1 | other | S3T246 |
| 头和脖子加起来很短。 | 1 | group_sum | S3T114 |
| 头和脖子加起来很长。 | 1 | group_sum | S3T139 |
| 头和脖子加起来长。 | 1 | group_sum | S3T197 |
| 头和脖子很小。 | 1 | other | S3T50 |
| 头和腿加起来很短。 | 1 | group_sum | S3T134 |
| 头和腿很短，尾巴没有很长。 | 1 | negation | S3T121 |
| 头和腿短，其他加起来长。 | 1 | group_sum | S3T192 |
| 头比较短，其他比较均匀。 | 1 | equality | S3T39 |
| 头长，腿未短。 | 1 | negation | S2T204 |
| 尾巴和腿加起来很长。 | 1 | group_sum | S3T132 |
| 尾巴很短，腿也不长。 | 1 | negation | S3T127 |
| 尾巴没有很长，头和腿短。 | 1 | negation | S3T226 |
| 比较均衡，头有点儿短。 | 1 | equality | S1T98 |
| 比较均衡，头有点短。 | 1 | equality | S3T207 |
| 比较均衡，尾巴有点长。 | 1 | equality | S1T221 |
| 比较均衡，脖子有点儿短。 | 1 | equality | S1T264 |
| 比较均衡，脖子短。 | 1 | equality | S1T291 |
| 脖子、尾巴都。 | 1 | other | S2T167 |
| 脖子。 | 1 | other | S2T208 |
| 脖子和尾巴都。 | 1 | other | S2T183 |
| 脖子和腿偏短，其他偏长，没有很长。 | 1 | negation | S3T100 |
| 脖子比较短，其他部位均衡。 | 1 | equality | S1T1 |
| 腿长，其他差不多。 | 1 | equality | S1T169 |
| 都不长，比较均衡。 | 1 | equality, negation | S3T280 |
| 都比较均衡，脖子和尾巴短。 | 1 | equality | S1T195 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位比较均衡。 | 7 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T35, S1T41, S1T44, S1T46, S1T123, S2T15, S2T141 |
| 四个部位长度差不多。 | 4 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T104, S1T117, S1T243, S2T262 |
| 尾巴短。 | 4 | 0.000 | absolute_short:尾巴 < 0.50 | S1T302, S2T46, S2T287, S2T301 |
| 脖子短。 | 4 | 0.000 | absolute_short:脖子 < 0.50 | S2T41, S2T44, S2T117, S2T303 |
| 四个部位长度均衡。 | 3 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T13, S1T15, S1T23 |
| 头最长。 | 3 | 0.222 | superlative:头 > 尾巴; superlative:头 > 脖子; superlative:头 > 腿 | S4T23, S4T44, S4T49 |
| 脖子和尾巴短。 | 2 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50 | S2T148, S2T316 |
| 脖子和腿短。 | 2 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T198, S1T257 |
| 头不会短。 | 1 | 0.000 | absolute_long:头 > 0.50 | S3T312 |
| 头也长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S2T261 |
| 头和尾巴很长很短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S3T111 |
| 头和尾巴短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S2T103 |
| 头和脖子加起来很长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S3T139 |
| 头和脖子短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50 | S1T213 |
| 头和脖子长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S3T251 |
| 头长，腿未短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T204 |
| 比较均衡，脖子有点儿短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S1T264 |
| 脖子和头长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S2T251 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T139 |
| 腿比较短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S2T74 |
| 腿短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S2T207 |
| 腿稍短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S3T221 |
| 都比较均衡，脖子和尾巴短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50 | S1T195 |
| 四个部位都比较长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S2T200 |

### S220

- trial 数: 1408; 非空文本: 1406; fidelity 可评分率: 0.984; 平均 fidelity: 0.885; 完全忠实率: 0.770; 低 fidelity 率: 0.067.
- 旧版 region 覆盖率: 0.984; 旧版 region 有未处理片段率: 0.058.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1397 | 0.992 |
| comparison | 487 | 0.346 |
| equality | 249 | 0.177 |
| superlative | 245 | 0.174 |
| count_abstract | 85 | 0.060 |
| negation | 8 | 0.006 |
| ranking | 4 | 0.003 |
| body_ref | 3 | 0.002 |
| empty | 2 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，尾巴短。 | 50 |
| 头短，腿长。 | 45 |
| 四个部位差不多长。 | 38 |
| 头长，尾巴长。 | 35 |
| 脖子和尾巴长于头和腿。 | 33 |
| 头短，腿短。 | 31 |
| 腿长于其他三个部位。 | 25 |
| 头、脖子、腿长于尾巴。 | 19 |
| 头、脖子、尾巴长于腿。 | 18 |
| 脖子和腿长于头和尾巴。 | 18 |
| 头和腿长于脖子和尾巴。 | 18 |
| 头和脖子长于腿和尾巴。 | 17 |
| 脖子长于其他三个部位。 | 16 |
| 脖子最短。 | 15 |
| 脖子最长。 | 13 |
| 腿和尾巴长于头和脖子。 | 12 |
| 头、脖子和腿长于尾巴。 | 12 |
| 头和尾巴长于脖子和腿。 | 12 |
| 头、脖子和尾巴长于腿。 | 12 |
| 尾巴最短。 | 10 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位差不多长。 | 38 | equality | S2T260, S2T264, S2T274, S2T286, S2T301, S3T24, S3T27, S3T56 |
| 腿长于其他三个部位。 | 25 | count_abstract | S3T31, S3T201, S3T234, S3T240, S3T258, S4T45, S4T129, S4T170 |
| 脖子长于其他三个部位。 | 16 | count_abstract | S3T118, S3T120, S3T190, S4T85, S4T89, S4T91, S4T102, S4T118 |
| 脖子短于其他三个部位。 | 9 | count_abstract | S3T122, S3T256, S4T44, S4T46, S4T63, S4T137, S4T144, S4T196 |
| 脖子和尾巴一样长。 | 7 | equality | S1T13, S1T18, S1T23, S1T30, S1T35, S3T35, S3T228 |
| 脖子和尾巴一样长，头和腿一样长。 | 7 | equality | S1T78, S2T277, S2T280, S2T296, S3T18, S3T52, S3T239 |
| 头长于其他三个部位。 | 6 | count_abstract | S4T110, S4T117, S4T125, S4T179, S4T219, S4T221 |
| 头、脖子、腿和尾巴差不多长。 | 5 | equality | S2T65, S2T97, S2T178, S2T232, S2T233 |
| 脖子和腿一样长。 | 5 | equality | S1T15, S1T16, S1T21, S1T22, S1T25 |
| 头和脖子一样长，腿和尾巴一样长。 | 4 | equality | S1T48, S2T255, S2T297, S3T156 |
| 头和脖子一样长，腿比尾巴长。 | 4 | equality | S3T152, S3T155, S3T157, S3T160 |
| 尾巴短于其他三个部位。 | 4 | count_abstract | S3T121, S3T191, S3T204, S3T236 |
| 尾巴长于其他三个部位。 | 4 | count_abstract | S3T10, S4T134, S4T207, S4T265 |
| 四个差不多长。 | 3 | equality | S3T232, S3T233, S3T301 |
| 头、脖子、尾巴一样长。 | 3 | equality | S1T17, S1T27, S1T33 |
| 头、脖子和腿差不多长，尾巴最短。 | 3 | equality | S2T244, S3T7, S3T37 |
| 头和尾巴一样长，脖子和腿一样长。 | 3 | equality | S2T254, S3T41, S3T238 |
| 头和尾巴不一样长，且腿长。 | 3 | equality, negation | S3T262, S3T263, S3T266 |
| 头和尾巴差不多长，脖子和腿差不多长。 | 3 | equality | S2T288, S2T303, S3T254 |
| 脖子、尾巴、腿一样长，头短。 | 3 | equality | S1T75, S1T110, S1T151 |
| 头、尾巴、脖子、腿差不多长。 | 2 | equality | S1T186, S1T274 |
| 头、脖子、腿、尾巴差不多长。 | 2 | equality | S2T64, S2T71 |
| 头、脖子、腿一样长。 | 2 | equality | S1T14, S1T29 |
| 头、脖子和尾巴一样长，腿比较短。 | 2 | equality | S3T42, S4T13 |
| 头、脖子和腿差不多。 | 2 | equality | S2T268, S2T270 |
| 头和尾巴一样长。 | 2 | equality | S1T31, S3T261 |
| 头和尾巴一样长，且脖子和腿不一样长。 | 2 | equality, negation | S3T167, S3T169 |
| 头和脖子一样长，尾巴短，腿最长。 | 2 | equality | S1T36, S1T37 |
| 头和腿一样长。 | 2 | equality | S1T34, S2T251 |
| 头和腿一样长，且脖子和尾巴一样长。 | 2 | equality | S1T32, S3T168 |
| 头和腿一样长，脖子和尾巴一样长。 | 2 | equality | S3T98, S3T292 |
| 脖子和腿一样长，头和尾巴一样长。 | 2 | equality | S2T295, S3T289 |
| 脖子最长，长于其他三个部位。 | 2 | count_abstract | S4T74, S4T81 |
| 腿比其他三个部位都短。 | 2 | count_abstract | S3T290, S3T293 |
| 三个部位长于腿。 | 1 | count_abstract | S3T33 |
| 从长到短是头、尾巴、脖子和腿。 | 1 | ranking | S1T38 |
| 从长到短是头、脖子、尾巴和腿。 | 1 | ranking | S2T293 |
| 从长到短是头和尾巴、脖子、腿。 | 1 | ranking | S1T39 |
| 四部位差不多长。 | 1 | equality | S3T260 |
| 头、尾巴和脖子差不多长，腿最长。 | 1 | equality | S1T262 |
| 头、尾巴和脖子差不多长，腿短。 | 1 | equality | S1T123 |
| 头、尾巴和腿一样长。 | 1 | equality | S2T256 |
| 头、尾巴和腿一样长，腿是最长。 | 1 | equality | S1T150 |
| 头、尾巴和腿差不多长。 | 1 | equality | S3T285 |
| 头、脖、尾差不多长，腿稍长。 | 1 | equality | S1T301 |
| 头、脖子、尾差不多长，腿较长。 | 1 | equality | S1T131 |
| 头、脖子、尾巴、腿和躯干都差不多长。 | 1 | equality, body_ref | S2T56 |
| 头、脖子、尾巴一样长，腿最长。 | 1 | equality | S1T96 |
| 头、脖子、尾巴一样长，长于腿。 | 1 | equality | S4T33 |
| 头、脖子、尾巴和腿差不多长。 | 1 | equality | S2T187 |
| 头、脖子、尾巴都很长，且长度差不多。 | 1 | equality | S1T49 |
| 头、脖子、腿一样长，尾巴很短。 | 1 | equality | S2T250 |
| 头、脖子、腿和尾巴差不多。 | 1 | equality | S2T271 |
| 头、脖子、腿和尾巴都一样长。 | 1 | equality | S1T24 |
| 头、脖子、腿差不多长。 | 1 | equality | S2T252 |
| 头、脖子、腿，和尾巴差不多长。 | 1 | equality | S2T73 |
| 头、脖子、躯干一样长。 | 1 | equality, body_ref | S1T19 |
| 头、脖子和尾巴一样长，腿短。 | 1 | equality | S1T79 |
| 头、脖子和尾巴差不多长，腿最长。 | 1 | equality | S3T171 |
| 头、脖子和尾巴差不多长，腿相对短一点点。 | 1 | equality | S1T99 |
| 头、脖子和尾巴差不多长，长于腿。 | 1 | equality | S3T54 |
| 头、脖子和尾巴长度相等，都比腿长。 | 1 | equality | S3T13 |
| 头、脖子和腿一样长，尾巴最短。 | 1 | equality | S4T16 |
| 头、脖子和腿一样长，尾巴稍长一点。 | 1 | equality | S4T10 |
| 头、脖子和腿差不多长。 | 1 | equality | S2T284 |
| 头、脖子和腿差不多长，尾巴最长。 | 1 | equality | S3T8 |
| 头、脖子和腿差不多长，尾巴短。 | 1 | equality | S1T121 |
| 头、脖子和腿是一样长。 | 1 | equality | S2T257 |
| 头、腿、尾巴、脖子差不多长。 | 1 | equality | S2T173 |
| 头、腿、尾巴一样长。 | 1 | equality | S1T28 |
| 头、腿、尾巴一样长，脖子最短。 | 1 | equality | S4T5 |
| 头、腿、尾巴差不多长。 | 1 | equality | S3T250 |
| 头、腿短，脖子和尾一样长。 | 1 | equality | S1T148 |
| 头和尾一样长，腿最长，脖子最短。 | 1 | equality | S1T45 |
| 头和尾巴一样短，脖子和腿最长，且一样长。 | 1 | equality | S1T92 |
| 头和尾巴一样长，且腿和脖子短。 | 1 | equality | S3T269 |
| 头和尾巴一样长，且腿短。 | 1 | equality | S3T267 |
| 头和尾巴一样长，脖子中等，腿最短。 | 1 | equality | S1T94 |
| 头和尾巴一样长，脖子中等，腿短。 | 1 | equality | S1T76 |
| 头和尾巴一样长，脖子和腿一样短。 | 1 | equality | S3T2 |
| 头和尾巴一样长，脖子很长，腿短。 | 1 | equality | S1T82 |
| 头和尾巴一样长，脖子最长。 | 1 | equality | S4T15 |
| 头和尾巴一样长，脖子最长，腿中等。 | 1 | equality | S1T93 |
| 头和尾巴不一样长，且脖子和腿一样长。 | 1 | equality, negation | S3T264 |
| 头和尾巴不一样长，且腿短。 | 1 | equality, negation | S3T265 |
| 头和尾巴差不多。 | 1 | equality | S2T269 |
| 头和尾巴差不多长，脖子中等，脖子和腿中等，脖子较短。 | 1 | equality | S2T5 |
| 头和尾巴差不多长，脖子中等，腿短。 | 1 | equality | S1T261 |
| 头和尾巴差不多长，脖子最长，腿最短。 | 1 | equality | S2T4 |
| 头和尾巴都较短，脖子和腿较长且差不多长。 | 1 | equality | S1T6 |
| 头和脖子一样的较长，腿中等，尾巴最短。 | 1 | equality | S1T61 |
| 头和脖子一样长。 | 1 | equality | S3T3 |
| 头和脖子一样长，尾巴和腿一样长。 | 1 | equality | S3T158 |
| 头和脖子一样长，尾巴比其他三个部位都长。 | 1 | equality, count_abstract | S4T3 |
| 头和脖子一样长，尾巴比腿长。 | 1 | equality | S3T154 |
| 头和脖子一样长，比腿和尾巴都长。 | 1 | equality | S1T47 |
| 头和脖子一样长，腿和尾巴也中等较长，四个都差不多长。 | 1 | equality | S1T72 |
| 头和脖子一样长，腿和尾巴短。 | 1 | equality | S1T187 |
| 头和脖子一样长，腿最短，尾巴中等，较短。 | 1 | equality | S1T71 |
| 头和脖子和尾巴差不多长，腿最短。 | 1 | equality | S2T304 |
| 头和脖子和腿差不多长。 | 1 | equality | S2T318 |
| 头和脖子差不多长，尾巴比腿长。 | 1 | equality | S3T146 |
| 头和脖子差不多，腿和尾巴差不多。 | 1 | equality | S3T48 |
| 头和腿一样长，脖子中等，尾巴短。 | 1 | equality | S1T114 |
| 头和腿差不多长。 | 1 | equality | S2T253 |
| 头和腿短，脖子和尾一样长。 | 1 | equality | S1T147 |
| 头最短，其他三个部位差不多长。 | 1 | equality, count_abstract | S3T315 |
| 头最长，其他三个部位差不多。 | 1 | equality, count_abstract | S3T46 |
| 头最长，四个都不一样长。 | 1 | equality, negation | S1T46 |
| 头比尾巴长，脖子和腿差不多长。 | 1 | equality | S3T284 |
| 头比较长，脖子和尾巴一样长。 | 1 | equality | S3T179 |
| 头特别短，其他三个部位差不多长。 | 1 | equality, count_abstract | S3T26 |
| 头短，尾巴短，脖子和腿差不多长，都是中等长度。 | 1 | equality | S2T6 |
| 头超长，尾巴超长，脖子较短，腿中等，躯干长度的一半。 | 1 | body_ref | S1T2 |
| 头长于脖子，头和尾巴差不多长，腿和头和尾巴都差不多长。 | 1 | equality | S1T5 |
| 头长，脖子和尾巴一样长，腿短。 | 1 | equality | S1T109 |
| 头长，腿中腿差不多，腿中等长，脖子和尾巴短。 | 1 | equality | S1T67 |
| 头长，腿和尾巴差不多长，脖子短。 | 1 | equality | S1T153 |
| 尾巴和脖子差不多长。 | 1 | equality | S1T66 |
| 尾巴和腿一样长，头很长，脖子中等。 | 1 | equality | S1T88 |
| 尾巴最短，且头、腿和脖子一样长。 | 1 | equality | S3T257 |
| 尾巴最短，其他三个部位差不多长。 | 1 | equality, count_abstract | S3T25 |
| 尾巴最长，长于其他三个部位。 | 1 | count_abstract | S3T309 |
| 尾巴比其他三个部位都短。 | 1 | count_abstract | S3T291 |
| 差不多长。 | 1 | equality | S3T237 |
| 有两个部位显著的短于另外两个部位。 | 1 | count_abstract | S3T81 |
| 脖子、头，脖子和尾巴和腿差不多长。 | 1 | equality | S2T287 |
| 脖子、尾巴、头差不多长，腿最长。 | 1 | equality | S3T170 |
| 脖子、尾巴、腿一样长。 | 1 | equality | S1T20 |
| 脖子、尾巴、腿差不多长，头短。 | 1 | equality | S1T287 |
| 脖子、尾巴、腿长度差不多，头最短。 | 1 | equality | S1T50 |
| 脖子、尾巴长长中等，头、腿差不多长。 | 1 | equality | S1T192 |
| 脖子、腿、尾差不多长，头稍短。 | 1 | equality | S1T157 |
| 脖子、腿、尾巴一样长。 | 1 | equality | S1T26 |
| 脖子、腿、尾巴一样长，头比其他部位都长。 | 1 | equality | S3T20 |
| 脖子、腿、尾巴差不多长。 | 1 | equality | S3T246 |
| 脖子、腿和尾巴差不多长，头最短。 | 1 | equality | S3T50 |
| 脖子为腿差不多长，头长。 | 1 | equality | S1T275 |
| 脖子和头一样长，尾巴比腿长。 | 1 | equality | S3T151 |
| 脖子和头比其他三个部位短一点。 | 1 | count_abstract | S4T178 |
| 脖子和尾一样长，头和腿短。 | 1 | equality | S1T277 |
| 脖子和尾巴一样长，且头和腿一样长。 | 1 | equality | S3T166 |
| 脖子和尾巴一样长，头中等，腿短。 | 1 | equality | S1T83 |
| 脖子和尾巴一样长，腿比较短。 | 1 | equality | S3T178 |
| 脖子和尾巴一样长，腿较长，头稍微短一点。 | 1 | equality | S1T74 |
| 脖子和尾巴一样长，长于头和腿。 | 1 | equality | S3T230 |
| 脖子和尾巴差不多长。 | 1 | equality | S3T248 |
| 脖子和尾巴差不多长，头和腿差不多长。 | 1 | equality | S3T253 |
| 脖子和尾巴差不多长，头较短，腿也较短，腿比头稍微长一些。 | 1 | equality | S1T12 |
| 脖子和尾巴比较一样长，头和腿比较短。 | 1 | equality | S3T308 |
| 脖子和尾巴长度差不多，头和腿长度差不多。 | 1 | equality | S3T30 |
| 脖子和尾巴长度差不多，比腿和头都长。 | 1 | equality | S3T12 |
| 脖子和腿一样短。 | 1 | equality | S3T38 |
| 脖子和腿一样长，头和尾巴比较短。 | 1 | equality | S3T58 |
| 脖子和腿一样长，比头和尾巴长。 | 1 | equality | S3T295 |
| 脖子和腿中等长度，头和尾巴一样长。 | 1 | equality | S3T15 |
| 脖子尾一样长且最长。 | 1 | equality | S1T91 |
| 脖子尾一样长，腿中等，头短。 | 1 | equality | S1T69 |
| 脖子尾差不多长，头和腿差不多长。 | 1 | equality | S1T260 |
| 脖子显著短于其他三个部位。 | 1 | count_abstract | S3T44 |
| 脖子最短，头和尾还有腿差不多长。 | 1 | equality | S3T5 |
| 脖子最长，其他三个部位差不多短。 | 1 | equality, count_abstract | S3T100 |
| 脖子比其他三个部位都短。 | 1 | count_abstract | S3T278 |
| 脖子比其他三个部位都长。 | 1 | count_abstract | S4T28 |
| 脖子比头长，尾巴和腿一样长。 | 1 | equality | S3T150 |
| 脖子长于三其他三个部位。 | 1 | count_abstract | S4T100 |
| 脖子长，腿和头差不多，尾巴比较短。 | 1 | equality | S1T65 |
| 腿会比较长，脖子和腿一样长。 | 1 | equality | S2T246 |
| 腿和尾巴一样长。 | 1 | equality | S2T307 |
| 腿和尾巴一样长，头最长，脖子中等。 | 1 | equality | S1T70 |
| 腿和尾巴一样长，比头和脖子长。 | 1 | equality | S3T294 |
| 腿和尾巴一样长，脖子稍短，头最短。 | 1 | equality | S3T1 |
| 腿和尾巴一样长，长于脖子和头。 | 1 | equality | S3T225 |
| 腿和尾巴差不多长。 | 1 | equality | S3T249 |
| 腿和尾巴最长，脖子第二，头较短。 | 1 | ranking | S1T113 |
| 腿和尾巴还有脖子差不多长，头比较短。 | 1 | equality | S3T161 |
| 腿很长，头和脖子一样长。 | 1 | equality | S3T36 |
| 腿最短，其他三个部位差不多长。 | 1 | equality, count_abstract | S3T316 |
| 腿最短，头、脖子和尾巴差不多长。 | 1 | equality | S3T39 |
| 腿最短，它的脖子和尾巴是一样长。 | 1 | equality | S2T317 |
| 腿短于其他三个部位。 | 1 | count_abstract | S3T123 |
| 腿较长，尾巴稍短，脖子和尾巴差不多长，头比脖子更短些。 | 1 | equality | S1T1 |
| 腿长，头、脖子、尾差不多长。 | 1 | equality | S1T286 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位差不多长。 | 36 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T260, S2T264, S2T274, S2T286, S2T301, S3T24, S3T27, S3T56 |
| 头、脖子、腿和尾巴差不多长。 | 4 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S2T65, S2T178, S2T232, S2T233 |
| 四个差不多长。 | 3 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T232, S3T233, S3T301 |
| 头、脖子、尾巴一样长。 | 3 | 0.000 | equality_range:头+脖子+尾巴 = | S1T17, S1T27, S1T33 |
| 脖子和尾巴一样长。 | 3 | 0.000 | equality_range:脖子+尾巴 = | S1T23, S1T30, S3T228 |
| 脖子和腿一样长。 | 3 | 0.000 | equality_range:脖子+腿 = | S1T16, S1T21, S1T22 |
| 头、尾巴、脖子、腿差不多长。 | 2 | 0.000 | equality_range:头+尾巴+脖子+腿 = | S1T186, S1T274 |
| 头、脖子、腿、尾巴差不多长。 | 2 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S2T64, S2T71 |
| 头、脖子、腿一样长。 | 2 | 0.000 | equality_range:头+脖子+腿 = | S1T14, S1T29 |
| 头、脖子和腿差不多。 | 2 | 0.000 | equality_range:头+脖子+腿 = | S2T268, S2T270 |
| 头长，尾巴长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S4T316, S5T13 |
| 脖子和尾巴一样长，头和腿一样长。 | 2 | 0.000 | equality_range:脖子+尾巴 =; equality_range:头+腿 = | S2T277, S2T296 |
| 头、尾巴和腿差不多长。 | 1 | 0.000 | equality_range:头+尾巴+腿 = | S3T285 |
| 头、脖子、尾巴、腿和躯干都差不多长。 | 1 | 0.000 | equality_range:头+脖子+尾巴+腿 = | S2T56 |
| 头、脖子、尾巴一样长，长于腿。 | 1 | 0.000 | equality_range:头+脖子+尾巴 = | S4T33 |
| 头、脖子、尾巴和腿差不多长。 | 1 | 0.000 | equality_range:头+脖子+尾巴+腿 = | S2T187 |
| 头、脖子、腿和尾巴差不多。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S2T271 |
| 头、脖子、腿和尾巴都一样长。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T24 |
| 头、脖子、腿差不多长。 | 1 | 0.000 | equality_range:头+脖子+腿 = | S2T252 |
| 头、脖子、腿，和尾巴差不多长。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S2T73 |
| 头、脖子、躯干一样长。 | 1 | 0.000 | body_ref:头 = 0.50; body_ref:脖子 = 0.50; equality_range:头+脖子 = | S1T19 |
| 头、脖子和尾巴差不多长，长于腿。 | 1 | 0.000 | equality_range:头+脖子+尾巴 = | S3T54 |
| 头、脖子和腿是一样长。 | 1 | 0.000 | equality_range:头+脖子+腿 = | S2T257 |
| 头、脖子和腿短于尾巴。 | 1 | 0.000 | comparison:头+脖子+腿 < 尾巴 | S4T138 |
| 头、腿、尾巴、脖子差不多长。 | 1 | 0.000 | equality_range:头+腿+尾巴+脖子 = | S2T173 |
| 头、腿、尾巴差不多长。 | 1 | 0.000 | equality_range:头+腿+尾巴 = | S3T250 |
| 头和尾巴一样长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T31 |
| 头和尾巴差不多。 | 1 | 0.000 | equality_range:头+尾巴 = | S2T269 |
| 头和尾巴差不多长，脖子和腿差不多长。 | 1 | 0.000 | equality_range:头+尾巴 =; equality_range:脖子+腿 = | S2T303 |
| 头和脖子和腿差不多长。 | 1 | 0.000 | equality_range:头+脖子+腿 = | S2T318 |
| 头和腿一样长，且脖子和尾巴一样长。 | 1 | 0.000 | equality_range:头+腿 =; equality_range:脖子+尾巴 = | S3T168 |
| 尾巴和脖子差不多长。 | 1 | 0.000 | equality_range:尾巴+脖子 = | S1T66 |
| 脖子、头，脖子和尾巴和腿差不多长。 | 1 | 0.000 | equality_range:脖子+尾巴+腿 = | S2T287 |
| 脖子、尾巴和头，腿长稍短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S1T216 |
| 脖子、腿、尾巴差不多长。 | 1 | 0.000 | equality_range:脖子+腿+尾巴 = | S3T246 |
| 脖子和尾巴一样长，长于头和腿。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S3T230 |
| 脖子和腿一样长，比头和尾巴长。 | 1 | 0.000 | equality_range:脖子+腿 = | S3T295 |
| 脖子和腿长于头和尾巴。 | 1 | 0.000 | comparison:脖子+腿 > 头+尾巴 | S4T273 |
| 腿和尾巴一样长，长于脖子和头。 | 1 | 0.000 | equality_range:腿+尾巴 = | S3T225 |
| 腿和尾巴差不多长。 | 1 | 0.000 | equality_range:腿+尾巴 = | S3T249 |
| 头特别短，其他三个部位差不多长。 | 1 | 0.250 | complement:脖子 < 0.50; complement:腿 < 0.50; complement:尾巴 < 0.50 | S3T26 |
| 脖子长，腿和头差不多，尾巴比较短。 | 1 | 0.333 | equality_range:腿+头 =; absolute_short:尾巴 < 0.50 | S1T65 |
| 腿和尾巴长，脖子中等，头短。 | 1 | 0.400 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50; absolute:脖子 middle_lower | S1T125 |

### S221

- trial 数: 832; 非空文本: 830; fidelity 可评分率: 0.993; 平均 fidelity: 0.864; 完全忠实率: 0.769; 低 fidelity 率: 0.069.
- 旧版 region 覆盖率: 0.993; 旧版 region 有未处理片段率: 0.005.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 823 | 0.989 |
| comparison | 299 | 0.359 |
| equality | 39 | 0.047 |
| superlative | 21 | 0.025 |
| group_sum | 3 | 0.004 |
| ranking | 3 | 0.004 |
| empty | 2 | 0.002 |
| negation | 1 | 0.001 |
| count_abstract | 1 | 0.001 |
| meta | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，头长。 | 108 |
| 腿长，头短。 | 101 |
| 腿短，尾巴长。 | 62 |
| 腿短，尾巴短。 | 41 |
| 腿短，头长，尾巴长。 | 36 |
| 腿短，脖子长。 | 34 |
| 头和尾巴比较长。 | 23 |
| 腿比头长，尾巴比脖子长。 | 22 |
| 头和腿比较长。 | 18 |
| 头比腿长，脖子比尾巴长。 | 17 |
| 脖子和尾巴比较长。 | 16 |
| 头和脖子比较长。 | 13 |
| 头长，尾巴长。 | 13 |
| 头比腿长，尾巴比脖子长。 | 12 |
| 四个部位长度相近。 | 10 |
| 头和腿都比较长。 | 10 |
| 腿比头长，脖子比尾巴长。 | 10 |
| 头比腿长。 | 9 |
| 头比较长，腿比较短。 | 9 |
| 腿比较长，头比较短。 | 9 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位长度相近。 | 10 | equality | S1T288, S2T34, S2T72, S2T74, S2T77, S2T119, S2T157, S2T198 |
| 头和腿长度相似。 | 8 | equality | S1T211, S1T212, S1T215, S1T217, S1T218, S1T219, S1T221, S1T251 |
| 腿和头长度相似。 | 3 | equality | S1T273, S2T46, S2T47 |
| 头、脖子、腿、尾巴长度均衡。 | 2 | equality | S1T313, S1T316 |
| 头和腿长度相近。 | 2 | equality | S1T295, S2T2 |
| 四个部位一样长。 | 1 | equality | S1T101 |
| 四个部位长度比较均衡。 | 1 | equality | S1T309 |
| 四个部位长度相似。 | 1 | equality | S1T131 |
| 头、脖子、尾巴和腿长度相似。 | 1 | equality | S2T6 |
| 头和尾巴长度相似。 | 1 | equality | S1T231 |
| 头和腿长度差不多，脖子比尾巴长。 | 1 | equality | S1T107 |
| 头和腿长度相近，四个部位都比较长。 | 1 | equality | S2T14 |
| 头最长，腿第三长。 | 1 | ranking, count_abstract | S1T208 |
| 头比腿长，尾巴和脖子一样长。 | 1 | equality | S1T48 |
| 头比腿长，尾巴和脖子差不多。 | 1 | equality | S1T102 |
| 头比腿长，脖子和尾巴一样长。 | 1 | equality | S1T122 |
| 脖子、腿、尾巴长度相近。 | 1 | equality | S1T138 |
| 脖子和尾巴一样长。 | 1 | equality | S1T2 |
| 脖子和尾巴长度之和大于腿和头的长度之和。 | 1 | group_sum | S1T4 |
| 腿不是最长，头不是最短。 | 1 | negation | S1T158 |
| 腿和头长度相似，尾巴和脖子长度相似。 | 1 | equality | S1T86 |
| 腿最长，头第二长。 | 1 | ranking | S1T210 |
| 腿比头长，尾巴和脖子差不多长。 | 1 | equality | S1T124 |
| 腿短，头和尾巴之和大于脖子和腿之和。 | 1 | group_sum | S2T177 |
| 腿短，腿和脖子之和小于头和尾巴之和。 | 1 | group_sum | S2T128 |
| 选错了。 | 1 | meta | S2T85 |
| 长度从长到短是头、尾巴、脖子、腿。 | 1 | ranking | S1T178 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度相近。 | 10 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T288, S2T34, S2T72, S2T74, S2T77, S2T119, S2T157, S2T198 |
| 腿比较长。 | 4 | 0.000 | absolute_long:腿 > 0.50 | S1T72, S1T132, S1T134, S1T292 |
| 头和腿比较长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T269, S1T307, S1T310 |
| 头长，脖子长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T55, S2T60, S2T139 |
| 腿和头长度相似。 | 3 | 0.000 | equality_range:腿+头 = | S1T273, S2T46, S2T47 |
| 头、脖子、腿、尾巴长度均衡。 | 2 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T313, S1T316 |
| 头和腿都比较长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T198, S2T16 |
| 头和腿长度相似。 | 2 | 0.000 | equality_range:头+腿 = | S1T212, S1T221 |
| 头和腿长度相近。 | 2 | 0.000 | equality_range:头+腿 = | S1T295, S2T2 |
| 头比较长，腿比较短。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_short:腿 < 0.50 | S1T168, S1T220 |
| 腿长，头长，尾巴长。 | 2 | 0.167 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50 | S2T286, S2T294 |
| 四个部位一样长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T101 |
| 四个部位长度比较均衡。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T309 |
| 四个部位长度相似。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T131 |
| 头、脖子、尾巴和腿长度相似。 | 1 | 0.000 | equality_range:头+脖子+尾巴+腿 = | S2T6 |
| 头和尾巴比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T293 |
| 头和尾巴长度相似。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T231 |
| 头和脖子比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T287 |
| 头和腿长度相近，四个部位都比较长。 | 1 | 0.000 | equality_range:头+腿 =; absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T14 |
| 头长，腿长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T143 |
| 脖子、腿、尾巴长度相近。 | 1 | 0.000 | equality_range:脖子+腿+尾巴 = | S1T138 |
| 腿和头长度相似，尾巴和脖子长度相似。 | 1 | 0.000 | equality_range:腿+头 =; equality_range:尾巴+脖子 = | S1T86 |
| 腿很长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T290 |
| 腿比较长，头也比较长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S1T153 |
| 腿短，头长。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_long:头 > 0.50 | S2T304 |
| 腿长，头长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S2T311 |
| 腿长，尾巴长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S2T145 |
| 腿长，脖子长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S2T295 |
| 头长，脖子长，尾巴长，腿短。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T57 |
| 腿短，头长，脖子长，尾巴长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T231 |
| 腿非常短，头、脖子和尾巴比较长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T128 |
| 头和尾巴最长。 | 1 | 0.333 | superlative:尾巴 > 脖子; superlative:尾巴 > 头 | S1T236 |
| 腿短，头长，尾巴长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T197 |

### S222

- trial 数: 832; 非空文本: 665; fidelity 可评分率: 0.770; 平均 fidelity: 0.851; 完全忠实率: 0.588; 低 fidelity 率: 0.079.
- 旧版 region 覆盖率: 0.770; 旧版 region 有未处理片段率: 0.132.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 627 | 0.754 |
| comparison | 188 | 0.226 |
| empty | 167 | 0.201 |
| superlative | 129 | 0.155 |
| equality | 117 | 0.141 |
| count_abstract | 20 | 0.024 |
| other | 10 | 0.012 |
| ranking | 4 | 0.005 |
| meta | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，尾巴短。 | 70 |
| 腿短，头长。 | 23 |
| 腿短，头短。 | 19 |
| 尾巴最长。 | 18 |
| 头最长。 | 18 |
| 头很长。 | 18 |
| 四个部位差不多长。 | 11 |
| 腿和尾巴都长。 | 11 |
| 尾巴和腿差不多长。 | 10 |
| 头比腿长。 | 9 |
| 头和脖子比腿长。 | 8 |
| 尾巴和腿都长。 | 8 |
| 尾巴和腿差不多。 | 8 |
| 脖子和尾巴比腿长。 | 8 |
| 尾巴比腿长。 | 7 |
| 尾巴和脖子比腿长。 | 6 |
| 尾巴最长，腿最短。 | 6 |
| 尾巴和腿都很长。 | 5 |
| 脖子比头和腿长。 | 5 |
| 头长。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位差不多长。 | 11 | equality | S2T34, S2T89, S2T94, S2T140, S2T150, S2T157, S2T169, S2T180 |
| 尾巴和腿差不多长。 | 10 | equality | S2T102, S2T219, S2T283, S2T284, S2T285, S2T293, S2T297, S2T310 |
| 尾巴和腿差不多。 | 8 | equality | S2T96, S2T176, S2T181, S2T240, S2T277, S2T296, S2T309, S3T27 |
| 头很大。 | 4 | other | S2T125, S2T155, S2T162, S2T243 |
| 尾巴和腿差不多一样长。 | 3 | equality | S2T22, S2T30, S2T39 |
| 腿和尾巴差不多。 | 3 | equality | S2T213, S2T245, S2T316 |
| 腿和尾巴差不多一样长。 | 3 | equality | S2T24, S2T27, S2T35 |
| 体型中小。 | 2 | other | S2T317, S2T319 |
| 体型中等，四个部位差不多长。 | 2 | equality | S2T151, S2T170 |
| 体型大，四个部位都差不多。 | 2 | equality | S2T188, S2T198 |
| 四个部位差不多长，体型偏大。 | 2 | equality | S2T161, S2T172 |
| 头和腿差不多一样长。 | 2 | equality | S2T127, S3T14 |
| 头比脖子长，腿和尾巴一样。 | 2 | equality | S1T123, S1T129 |
| 尾巴和腿差不多同样长。 | 2 | equality | S2T15, S2T16 |
| 尾巴和腿差不多同样长，脖子比头长。 | 2 | equality | S2T17, S2T18 |
| 尾巴最短，其他三个部位长。 | 2 | count_abstract | S1T6, S1T9 |
| 三个部位短，一个部位长。 | 1 | count_abstract | S1T74 |
| 两个部位很长，一个部位比较长，一个部位比较短。 | 1 | count_abstract | S1T23 |
| 两个部位很长，两个部位很短。 | 1 | count_abstract | S1T39 |
| 两个部位最短，一个部位比较长，一个部位比较短。 | 1 | count_abstract | S1T18 |
| 两个部位等长，一个，其他两个部位一长一短。 | 1 | equality, count_abstract | S1T31 |
| 从长到短是脖子、尾巴、头和腿。 | 1 | ranking | S3T19 |
| 体型中小，尾巴和腿差不多。 | 1 | equality | S2T120 |
| 体型中等，四个部位都差不多。 | 1 | equality | S2T192 |
| 体型中等，四个部位长度都差不多。 | 1 | equality | S2T41 |
| 体型中等，尾巴、头、腿差不多长。 | 1 | equality | S2T123 |
| 体型中等，尾巴和腿差不多，脖子很长。 | 1 | equality | S2T178 |
| 体型中等，腿最短，其他差不多。 | 1 | equality | S2T229 |
| 体型中等，腿长，其他差不多。 | 1 | equality | S2T134 |
| 体型偏中大。 | 1 | other | S2T114 |
| 体型偏中小，腿和尾巴差不多一样长。 | 1 | equality | S2T56 |
| 体型偏大，尾巴和腿差不多一样。 | 1 | equality | S2T177 |
| 体型偏大，尾巴和腿差不多一样长。 | 1 | equality | S2T57 |
| 体型偏小，脖子和尾巴差不多，腿最短。 | 1 | equality | S2T99 |
| 体型大，头很大。 | 1 | other | S2T196 |
| 体型小。 | 1 | other | S2T92 |
| 体型小，尾巴最长，脖子第二，头和腿最短。 | 1 | ranking | S2T174 |
| 四个部位很都不等长。 | 1 | equality | S1T33 |
| 四个部位都差不多，体型中等。 | 1 | equality | S2T234 |
| 四个部位都很小。 | 1 | other | S1T95 |
| 四个部位都比较平均。 | 1 | equality | S2T130 |
| 头、腿、尾巴差不多长。 | 1 | equality | S2T118 |
| 头、腿和尾巴差不多。 | 1 | equality | S2T207 |
| 头、腿和尾巴差不多长。 | 1 | equality | S2T226 |
| 头和尾巴一样长，脖子很长，腿很短。 | 1 | equality | S1T64 |
| 头和尾巴差不多，体型偏小。 | 1 | equality | S2T168 |
| 头和尾巴很长，其他两个部位很短。 | 1 | count_abstract | S1T5 |
| 头和脖子一样。 | 1 | equality | S2T175 |
| 头和脖子一样长，尾巴最长，腿最短。 | 1 | equality | S1T69 |
| 头和脖子一样，腿和尾巴一样。 | 1 | equality | S1T127 |
| 头和脖子差不多，腿和尾巴差不多。 | 1 | equality | S1T114 |
| 头和腿一样，脖子长，尾巴比较短。 | 1 | equality | S1T70 |
| 头和腿差不多长。 | 1 | equality | S2T142 |
| 头和腿差不多，尾巴长。 | 1 | equality | S2T262 |
| 头和腿等长，脖子比较长，尾巴比较短。 | 1 | equality | S1T77 |
| 头和腿，尾巴和腿差不多一样长。 | 1 | equality | S2T47 |
| 头和腿，差不多长。 | 1 | equality | S2T90 |
| 头很长，尾巴和脖子一样，腿最短。 | 1 | equality | S2T53 |
| 头最长了，其他部位差不多。 | 1 | equality | S1T20 |
| 头最长，脖子第二，其次是腿，然后尾巴。 | 1 | ranking | S3T32 |
| 头比脖子长，腿最长，尾巴也差不多。 | 1 | equality | S1T140 |
| 头特别长，其他部位都不等长。 | 1 | equality | S1T32 |
| 尾巴、腿、头差不多长。 | 1 | equality | S2T220 |
| 尾巴和头长，腿从腿和脖子一长一短。 | 1 | count_abstract | S1T44 |
| 尾巴和脖子一样，头和腿一样。 | 1 | equality | S1T120 |
| 尾巴和脖子中等长，头和腿一样长。 | 1 | equality | S1T41 |
| 尾巴很长，其他一样。 | 1 | equality | S1T122 |
| 尾巴最短，其他一样。 | 1 | equality | S1T82 |
| 尾巴最短，腿最长，脖子和头差不多。 | 1 | equality | S1T84 |
| 尾巴最长，脖子第二，头和尾巴最短。 | 1 | ranking | S1T133 |
| 尾巴最长，腿最短，头和脖子差不多。 | 1 | equality | S1T136 |
| 有三个部位长，面条左边脖子比较短。 | 1 | count_abstract | S1T10 |
| 脖子和尾巴一样长，头和腿一样长。 | 1 | equality | S1T60 |
| 脖子和腿一样，头和尾巴一样。 | 1 | equality | S1T87 |
| 脖子和腿差不多。 | 1 | equality | S1T154 |
| 脖子很长，其他三个部位很短。 | 1 | count_abstract | S1T98 |
| 脖子最长，其他头比较短，其他差不多。 | 1 | equality | S1T83 |
| 脖子最长，其他差不多。 | 1 | equality | S1T143 |
| 脖子比头长，其他两个部位很短。 | 1 | count_abstract | S1T101 |
| 脖子比头长，腿和尾巴差不多。 | 1 | equality | S1T128 |
| 脖子长，头和腿差不多。 | 1 | equality | S2T193 |
| 脖子长，腿和头差不多。 | 1 | equality | S2T195 |
| 腿、尾巴和腿差不多长。 | 1 | equality | S2T122 |
| 腿和尾巴一样长，头比脖子长。 | 1 | equality | S1T108 |
| 腿和尾巴差不多一样。 | 1 | equality | S2T203 |
| 腿和尾巴差不多同样长。 | 1 | equality | S2T12 |
| 腿和尾巴差不多长。 | 1 | equality | S2T269 |
| 腿最短，其他三个部位长。 | 1 | count_abstract | S1T7 |
| 腿比较短，头、脖子和尾巴一样长。 | 1 | equality | S1T148 |
| 腿长，尾巴短，头和脖子差不多。 | 1 | equality | S1T97 |
| 腿长，尾巴，头和脖子差不多。 | 1 | equality | S1T103 |
| 该体型比较小，脖子和尾巴等长，头短一点。 | 1 | equality | S1T67 |
| 选错了。 | 1 | meta | S2T173 |
| 除了腿最短，其他差不多都长。 | 1 | equality | S1T94 |
| 面朝右边，四个部位差不多长。 | 1 | equality | S1T38 |
| 面朝右边，头和尾巴比差不多一样长，比腿和脖子短。 | 1 | equality | S1T4 |
| 面朝右边，头和腿，差等长，尾巴长，脖子短。 | 1 | equality | S1T11 |
| 面朝右边，脖子比较短，其他三个部位比较长。 | 1 | count_abstract | S1T15 |
| 面朝右边，腿和尾巴比较长，头和脖子一长一短。 | 1 | count_abstract | S1T30 |
| 面朝右边，腿比较短，其他三个部位比较长。 | 1 | count_abstract | S1T13 |
| 面朝左边，两个部位比较长，一个部位更长，一个部位更短。 | 1 | count_abstract | S1T17 |
| 面朝左边，尾巴比较短，其他三个部位比较长。 | 1 | count_abstract | S1T14 |
| 面朝左边，脖子和尾巴比其他两个部位长。 | 1 | count_abstract | S1T1 |
| 面朝左边，脖子和腿一样长，头比较长，尾巴就比较短。 | 1 | equality | S1T28 |
| 面朝左边，脖子和腿一样长，头比较长，尾巴比较短。 | 1 | equality | S1T29 |
| 面朝左边，腿是最长，脖子比其他两个部位长。 | 1 | count_abstract | S1T3 |
| 面朝左边，腿特别长，其他部位差不多一样长。 | 1 | equality | S1T2 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位差不多长。 | 11 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T34, S2T89, S2T94, S2T140, S2T150, S2T157, S2T169, S2T180 |
| 尾巴和腿差不多。 | 5 | 0.000 | equality_range:尾巴+腿 = | S2T96, S2T176, S2T240, S2T296, S2T309 |
| 尾巴和腿差不多长。 | 5 | 0.000 | equality_range:尾巴+腿 = | S2T284, S2T297, S2T310, S3T76, S3T93 |
| 体型中等，四个部位差不多长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T151, S2T170 |
| 体型大，四个部位都差不多。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T188, S2T198 |
| 四个部位差不多长，体型偏大。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T161, S2T172 |
| 头和脖子比腿长。 | 2 | 0.000 | comparison:头+脖子 > 腿 | S2T28, S2T112 |
| 尾巴和腿差不多同样长。 | 2 | 0.000 | equality_range:尾巴+腿 = | S2T15, S2T16 |
| 腿和尾巴差不多。 | 2 | 0.000 | equality_range:腿+尾巴 = | S2T213, S2T245 |
| 体型中等，四个部位都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T192 |
| 体型中等，四个部位长度都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T41 |
| 体型中等，尾巴、头、腿差不多长。 | 1 | 0.000 | equality_range:尾巴+头+腿 = | S2T123 |
| 体型中等，尾巴长，腿短。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_short:腿 < 0.50 | S2T154 |
| 体型偏大，尾巴和腿差不多一样。 | 1 | 0.000 | equality_range:尾巴+腿 = | S2T177 |
| 体型偏小，脖子和头相对较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S2T179 |
| 体型比较大，尾巴比较短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T68 |
| 四个部位都差不多，体型中等。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T234 |
| 四个部位都比较平均。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T130 |
| 头、腿、尾巴差不多长。 | 1 | 0.000 | equality_range:头+腿+尾巴 = | S2T118 |
| 头、腿和尾巴差不多长。 | 1 | 0.000 | equality_range:头+腿+尾巴 = | S2T226 |
| 头和尾巴差不多，体型偏小。 | 1 | 0.000 | equality_range:头+尾巴 = | S2T168 |
| 头和脖子一样。 | 1 | 0.000 | equality_range:头+脖子 = | S2T175 |
| 头和腿，尾巴和腿差不多一样长。 | 1 | 0.000 | equality_range:尾巴+腿 = | S2T47 |
| 头比尾巴长一点。 | 1 | 0.000 | comparison:头 > 尾巴 | S3T16 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S2T36 |
| 头长，体型小。 | 1 | 0.000 | absolute_long:头 > 0.50 | S2T165 |
| 头长，尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S3T117 |
| 尾巴和腿差不多一样长。 | 1 | 0.000 | equality_range:尾巴+腿 = | S2T22 |
| 尾巴最长，脖子最短。 | 1 | 0.000 | superlative:尾巴 > 脖子; superlative:尾巴 > 头; superlative:尾巴 > 腿; superlative:脖子 < 头 | S3T35 |
| 脖子长，体型偏小。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T167 |
| 腿、尾巴和腿差不多长。 | 1 | 0.000 | equality_range:腿+尾巴 = | S2T122 |
| 腿和尾巴差不多一样。 | 1 | 0.000 | equality_range:腿+尾巴 = | S2T203 |
| 腿和尾巴差不多一样长。 | 1 | 0.000 | equality_range:腿+尾巴 = | S2T24 |
| 腿和尾巴差不多同样长。 | 1 | 0.000 | equality_range:腿+尾巴 = | S2T12 |
| 腿和尾巴差不多长。 | 1 | 0.000 | equality_range:腿+尾巴 = | S2T269 |
| 腿短，尾巴长。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_long:尾巴 > 0.50 | S2T88 |
| 除了腿最短，其他差不多都长。 | 1 | 0.000 | exclusion:脖子 < 0.50; exclusion:头 < 0.50; exclusion:尾巴 < 0.50; exclusion:腿 > 0.50 | S1T94 |
| 面朝右边，四个部位差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T38 |
| 面朝左边，脖子和腿一样长，头比较长，尾巴比较短。 | 1 | 0.000 | equality_range:脖子+腿 =; absolute_long:头 > 0.50; absolute_short:尾巴 < 0.50 | S1T29 |
| 面朝左边，尾巴比较短，其他三个部位比较长。 | 1 | 0.250 | complement:脖子 > 0.50; complement:头 > 0.50; complement:腿 > 0.50 | S1T14 |
| 头比脖子长，腿和尾巴比较短。 | 1 | 0.333 | absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T135 |
| 腿最短，头和脖子比较长。 | 1 | 0.400 | superlative:腿 < 尾巴; absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T290 |

### S223

- trial 数: 1280; 非空文本: 1278; fidelity 可评分率: 0.998; 平均 fidelity: 0.910; 完全忠实率: 0.673; 低 fidelity 率: 0.009.
- 旧版 region 覆盖率: 0.998; 旧版 region 有未处理片段率: 0.002.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1278 | 0.998 |
| superlative | 125 | 0.098 |
| comparison | 39 | 0.030 |
| empty | 2 | 0.002 |
| equality | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿短，尾巴短。 | 60 |
| 腿长，尾巴短。 | 53 |
| 头短，尾巴长。 | 44 |
| 头长，脖子长，腿短，尾巴长。 | 43 |
| 头长，脖子长，腿短，尾巴短。 | 41 |
| 头长，尾巴长。 | 34 |
| 头长，脖子短，腿长，尾巴长。 | 27 |
| 头短，脖子长，腿短，尾巴长。 | 27 |
| 头短，脖子长，腿适中，尾巴长。 | 25 |
| 头长，脖子长，腿适中，尾巴适中。 | 24 |
| 头长，脖子短，腿适中，尾巴短。 | 22 |
| 尾巴最长。 | 22 |
| 脖子最长。 | 21 |
| 头最长。 | 21 |
| 头短，脖子长，腿适中，尾巴短。 | 20 |
| 头长，脖子长，腿适中，尾巴长。 | 18 |
| 头长，脖子短，腿短，尾巴长。 | 18 |
| 头短，脖子适中，腿长，尾巴长。 | 18 |
| 头长，脖子长，腿适中，尾巴短。 | 17 |
| 腿最长。 | 16 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头短，脖子长，腿长，尾巴长，头比脖子短，脖子跟腿差不多长。 | 1 | equality | S2T29 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短，尾巴短。 | 3 | 0.000 | absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S4T158, S4T200, S4T245 |
| 头长，脖子长，腿短，尾巴长。 | 2 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50; absolute_short:腿 < 0.50 | S1T161, S1T281 |
| 头长，脖子短，腿短，尾巴长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_short:腿 < 0.50; absolute_long:尾巴 > 0.50 | S2T22 |
| 尾巴最长。 | 1 | 0.333 | superlative:尾巴 > 头; superlative:尾巴 > 腿 | S4T22 |
| 头短，脖子较长，腿中等，尾巴长。 | 1 | 0.400 | absolute_long:脖子 > 0.50; absolute:腿 middle_lower; absolute_long:尾巴 > 0.50 | S3T276 |
| 头长，脖子短，腿长，尾巴适中。 | 1 | 0.400 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50; absolute:尾巴 middle_upper | S1T15 |
| 头长，脖子长，腿长，尾巴适中。 | 1 | 0.400 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S1T99 |
| 脖子长，腿长，头短，尾巴适中。 | 1 | 0.400 | absolute_long:腿 > 0.50; absolute_short:头 < 0.50; absolute:尾巴 middle_lower | S1T6 |

### S224

- trial 数: 1664; 非空文本: 1660; fidelity 可评分率: 0.995; 平均 fidelity: 0.917; 完全忠实率: 0.785; 低 fidelity 率: 0.028.
- 旧版 region 覆盖率: 0.995; 旧版 region 有未处理片段率: 0.017.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1657 | 0.996 |
| superlative | 319 | 0.192 |
| comparison | 290 | 0.174 |
| count_abstract | 104 | 0.062 |
| equality | 76 | 0.046 |
| ranking | 26 | 0.016 |
| empty | 4 | 0.002 |
| other | 1 | 0.001 |
| negation | 1 | 0.001 |
| group_sum | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，尾巴短。 | 77 |
| 头短，尾巴短。 | 68 |
| 尾巴长，腿短。 | 60 |
| 尾巴长，腿长。 | 54 |
| 腿比其他三个部位短。 | 36 |
| 腿短，尾巴长。 | 30 |
| 尾巴短，头短。 | 25 |
| 腿最短。 | 22 |
| 尾巴短，头长。 | 18 |
| 头最长。 | 14 |
| 腿极短。 | 13 |
| 只有脖子最短。 | 12 |
| 腿比较短。 | 11 |
| 腿长，尾巴长。 | 11 |
| 腿非常短。 | 11 |
| 四个部位长度均适中。 | 11 |
| 头比尾巴长。 | 10 |
| 头非常长。 | 10 |
| 头特别长。 | 10 |
| 只有脖子较短。 | 10 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿比其他三个部位短。 | 36 | count_abstract | S1T9, S1T50, S1T54, S1T56, S1T64, S1T144, S1T195, S1T208 |
| 四个部位长度比较均匀。 | 5 | equality | S1T2, S1T37, S1T42, S1T46, S1T151 |
| 脖子比其他三个部位短。 | 5 | count_abstract | S1T58, S1T59, S1T124, S2T16, S2T147 |
| 四个部位长度差不多。 | 4 | equality | S1T156, S1T227, S2T11, S3T27 |
| 头比其他三个部位长。 | 4 | count_abstract | S1T143, S2T66, S2T119, S2T129 |
| 尾巴比其他三个部位短。 | 4 | count_abstract | S1T57, S1T61, S1T63, S1T117 |
| 腿比其他三个部位明显短。 | 4 | count_abstract | S2T18, S3T9, S3T14, S3T15 |
| 四个部位都差不多长。 | 3 | equality | S1T80, S1T260, S1T263 |
| 腿比其他三个部位长。 | 3 | count_abstract | S1T26, S1T31, S1T33 |
| 四个部位比较均匀。 | 2 | equality | S1T27, S1T100 |
| 四个部位长度均差不多。 | 2 | equality | S2T34, S3T11 |
| 头和脖子比其他两个部位长。 | 2 | count_abstract | S1T18, S4T3 |
| 头比其他三个部位短。 | 2 | count_abstract | S1T51, S2T148 |
| 尾巴最短，其他部位差不多。 | 2 | equality | S1T232, S1T285 |
| 脖子最短，其他三个部位适中。 | 2 | count_abstract | S1T184, S1T185 |
| 脖子最长，其他三个差不多。 | 2 | equality | S1T150, S1T281 |
| 腿最短，脖子次之。 | 2 | ranking | S3T83, S3T178 |
| 腿最长，四个部位都差不多长。 | 2 | equality | S1T176, S1T178 |
| 腿最长，脖子和尾巴次之，头最短。 | 2 | ranking | S3T176, S4T176 |
| 腿比其他三个部位短，头较短。 | 2 | count_abstract | S2T44, S2T102 |
| 腿比其他三个部位都短。 | 2 | count_abstract | S1T240, S2T15 |
| 三个部位均较短，腿和脖子较长。 | 1 | count_abstract | S2T196 |
| 三个部位均较长，脖子最短。 | 1 | count_abstract | S2T123 |
| 三个部位长度适中，尾巴稍微短一些。 | 1 | count_abstract | S3T1 |
| 只有头较短，脖子相当长。 | 1 | equality | S2T13 |
| 只有尾巴较短，差不多。 | 1 | equality | S2T82 |
| 只有脖子较短，其他三个部位差不多等长。 | 1 | equality, count_abstract | S4T184 |
| 只有脖子较短，其余部位差不多长。 | 1 | equality | S3T289 |
| 四个部位差不多长。 | 1 | equality | S3T277 |
| 四个部位差不多，脖子最长。 | 1 | equality | S2T204 |
| 四个部位都差不多长，尾巴最长。 | 1 | equality | S1T189 |
| 四个部位都差不多长，腿最长。 | 1 | equality | S1T257 |
| 四个部位都差不多长，腿最长，长度都适中。 | 1 | equality | S1T259 |
| 四个部位长度均匀。 | 1 | equality | S1T32 |
| 四个部位长度均差不多，头和脖子较长一些。 | 1 | equality | S3T139 |
| 四个部位长度均差不多，脖子短。 | 1 | equality | S2T94 |
| 四个部位长度均差不多，腿稍短一些。 | 1 | equality | S3T2 |
| 四个部位长度均衡。 | 1 | equality | S2T155 |
| 四个部位长度均较均匀，腿较短。 | 1 | equality | S3T198 |
| 四个部位长度差不多，头和脖子比腿和尾巴长一些。 | 1 | equality | S2T90 |
| 四个部位长度差不多，头和脖子略长一些。 | 1 | equality | S2T60 |
| 四个部位长度差不多，头和脖子较短一些。 | 1 | equality | S2T62 |
| 四个部位长度差不多，头较短。 | 1 | equality | S2T17 |
| 四个部位长度差不多，头较短一些。 | 1 | equality | S3T17 |
| 四个部位长度差不多，尾巴较短。 | 1 | equality | S2T77 |
| 四个部位长度差不多，脖子比头长一些。 | 1 | equality | S1T250 |
| 四个部位长度差不多，脖子较短。 | 1 | equality | S2T49 |
| 四个部位长度差不多，腿较长。 | 1 | equality | S2T76 |
| 四个部位长度比较均匀，脖子略短一些。 | 1 | equality | S1T296 |
| 四个部位长度较均匀，腿比较短。 | 1 | equality | S1T320 |
| 四个部位长度都差不多居中。 | 1 | equality | S3T310 |
| 四个部位长度都比较均匀，头最短。 | 1 | equality | S1T183 |
| 四个部位长度都非常均匀。 | 1 | equality | S1T62 |
| 头、腿和尾巴差不多，脖子较长。 | 1 | equality | S2T166 |
| 头、腿极长，脖子次之、尾巴极短。 | 1 | ranking | S3T234 |
| 头和脖子、腿差不多长，尾巴极短。 | 1 | equality | S2T61 |
| 头和脖子比其他两个部位明显较长。 | 1 | count_abstract | S4T8 |
| 头和脖子比其他两个部位短。 | 1 | count_abstract | S1T19 |
| 头和脖子相对。 | 1 | other | S1T112 |
| 头和脖子较短，其余两个部位较长。 | 1 | count_abstract | S4T53 |
| 头和脖子较短，比例较为协调。 | 1 | group_sum | S2T242 |
| 头和脖子较长，其他两个部位较短。 | 1 | count_abstract | S4T14 |
| 头和腿差不多长，尾巴极短。 | 1 | equality | S2T55 |
| 头最短，尾巴次之。 | 1 | ranking | S3T267 |
| 头最短，脖子次之。 | 1 | ranking | S3T214 |
| 头最短，腿、脖子、尾巴差不多长。 | 1 | equality | S1T233 |
| 头最短，腿次之。 | 1 | ranking | S3T102 |
| 头最长，脖子次之。 | 1 | ranking | S3T181 |
| 头最长，脖子次之，腿、尾巴极短。 | 1 | ranking | S3T242 |
| 头最长，腿次之。 | 1 | ranking | S3T182 |
| 头最长，腿次之，脖子和尾巴很短。 | 1 | ranking | S1T181 |
| 头比其他三个部位显著长。 | 1 | count_abstract | S2T36 |
| 头比其他三个部位都长。 | 1 | count_abstract | S1T210 |
| 头比其他三个部位长，头比尾巴长。 | 1 | count_abstract | S3T21 |
| 头比较短，其余部位差不多。 | 1 | equality | S2T226 |
| 头比较短，四个部位都差不多长。 | 1 | equality | S1T201 |
| 头短，其他三个部位都比较长。 | 1 | count_abstract | S1T23 |
| 头短，尾巴短，腿长，脖子长，四个部位都差不多长。 | 1 | equality | S1T99 |
| 头较短，其余三个差不多长。 | 1 | equality | S2T106 |
| 头较短，其余三个部位均较长。 | 1 | count_abstract | S2T179 |
| 头较短，其余三个部位差不多。 | 1 | equality, count_abstract | S2T79 |
| 头较短，其余三个部位较长。 | 1 | count_abstract | S2T99 |
| 头较短，腿较长，脖子、尾巴和腿差不多长。 | 1 | equality | S2T51 |
| 尾巴最短，其他三个差不多。 | 1 | equality | S1T278 |
| 尾巴最短，其次是脖子，头和腿很长。 | 1 | ranking | S1T234 |
| 尾巴最短，头次之，脖子和腿较长。 | 1 | ranking | S3T77 |
| 尾巴最短，脖子次之，头和腿较长。 | 1 | ranking | S2T121 |
| 尾巴最长，四个部位长度差不多。 | 1 | equality | S1T308 |
| 尾巴极短，其他三个部位都很长。 | 1 | count_abstract | S3T28 |
| 尾巴比其他三个部位短，头较短。 | 1 | count_abstract | S2T141 |
| 尾巴比其他三个部位较长。 | 1 | count_abstract | S2T149 |
| 尾巴比其他三个部位都短。 | 1 | count_abstract | S1T1 |
| 脖子、腿、尾巴差不多长，头最短。 | 1 | equality | S1T306 |
| 脖子、腿、尾巴差不多长，头较短。 | 1 | equality | S2T142 |
| 脖子和尾巴明显长，头最短，腿次之。 | 1 | ranking | S3T135 |
| 脖子和腿差不多长，尾巴和头较短。 | 1 | equality | S2T86 |
| 脖子最短，其他三个部位很长。 | 1 | count_abstract | S1T200 |
| 脖子最短，其他都差不多长。 | 1 | equality | S2T10 |
| 脖子最短，尾巴次之。 | 1 | ranking | S3T294 |
| 脖子最短，腿、头、尾巴差不多长。 | 1 | equality | S1T288 |
| 脖子最短，腿次之，其余两个差不多长。 | 1 | equality, ranking | S2T107 |
| 脖子最长，头次之，腿和尾巴较短。 | 1 | ranking | S3T167 |
| 脖子极短，其他三个部位较长。 | 1 | count_abstract | S2T67 |
| 脖子比其他三个部位明显短。 | 1 | count_abstract | S3T10 |
| 脖子比其他三个部位明显长，其他三部位差不多长。 | 1 | equality, count_abstract | S3T20 |
| 脖子比其他三个部位来说较短，也适中长。 | 1 | count_abstract | S3T93 |
| 脖子比其他三个部位较长。 | 1 | count_abstract | S2T161 |
| 脖子较短，其余差不多中等。 | 1 | equality | S3T43 |
| 腿和脖子比其他两个部位明显短。 | 1 | count_abstract | S3T247 |
| 腿很短，其他三个差不多。 | 1 | equality | S1T166 |
| 腿很短，其余三个部位都很长。 | 1 | count_abstract | S4T178 |
| 腿明显短于其他三个部位。 | 1 | count_abstract | S2T50 |
| 腿最短，头次之。 | 1 | ranking | S3T252 |
| 腿最短，头次之，脖子和尾巴极长。 | 1 | ranking | S4T44 |
| 腿最短，尾巴次之，头比脖子长。 | 1 | ranking | S3T65 |
| 腿最长，其他三个部位中等。 | 1 | count_abstract | S4T117 |
| 腿最长，其他三个部位长度差不多。 | 1 | equality, count_abstract | S1T228 |
| 腿最长，其余差不多。 | 1 | equality | S1T315 |
| 腿最长，头次之，尾巴极短。 | 1 | ranking | S3T207 |
| 腿最长，头次之，脖子和尾巴比较短。 | 1 | ranking | S1T300 |
| 腿最长，脖子次之，尾巴和头较短。 | 1 | ranking | S3T76 |
| 腿极短，其他三个部位都很长。 | 1 | count_abstract | S4T197 |
| 腿极短，其余三个部位均适中。 | 1 | count_abstract | S5T9 |
| 腿比其他三个部位明显较短。 | 1 | count_abstract | S3T18 |
| 腿较其他三个部位很长。 | 1 | count_abstract | S3T31 |
| 腿较其他三个部位极短。 | 1 | count_abstract | S3T160 |
| 腿较其他三个部位较短。 | 1 | count_abstract | S2T313 |
| 腿较短，其他部位差不多。 | 1 | equality | S2T272 |
| 腿较短，其余三个部位适中。 | 1 | count_abstract | S2T177 |
| 腿较短，四个部位差不多。 | 1 | equality | S2T170 |
| 腿长最长，脖子次之。 | 1 | ranking | S3T156 |
| 腿长，其余三个部位都不长。 | 1 | count_abstract, negation | S1T256 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度比较均匀。 | 5 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T2, S1T37, S1T42, S1T46, S1T151 |
| 四个部位长度差不多。 | 4 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T156, S1T227, S2T11, S3T27 |
| 四个部位都差不多长。 | 3 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T80, S1T260, S1T263 |
| 四个部位比较均匀。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T27, S1T100 |
| 四个部位长度均差不多。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T34, S3T11 |
| 四个部位差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T277 |
| 四个部位长度均匀。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T32 |
| 四个部位长度均差不多，腿稍短一些。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 =; absolute_short:腿 < 0.50 | S3T2 |
| 四个部位长度均衡。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T155 |
| 四个部位长度均较均匀，腿较短。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 =; absolute_short:腿 < 0.50 | S3T198 |
| 四个部位长度都非常均匀。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T62 |
| 头和脖子比其他两个部位明显较长。 | 1 | 0.000 | complement:腿 > 0.50; complement:尾巴 > 0.50 | S4T8 |
| 头很长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T90 |
| 头明显较长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S3T24 |
| 头较长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T275 |
| 脖子较短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S4T93 |
| 腿和头较短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S4T264 |
| 腿极短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S4T141 |
| 腿较其他三个部位很长。 | 1 | 0.000 | complement:脖子 > 0.50; complement:头 > 0.50; complement:尾巴 > 0.50 | S3T31 |
| 腿较其他三个部位极短。 | 1 | 0.000 | complement:脖子 < 0.50; complement:头 < 0.50; complement:尾巴 < 0.50 | S3T160 |
| 腿较短，四个部位差不多。 | 1 | 0.000 | absolute_short:腿 < 0.50; equality_range:脖子+头+腿+尾巴 = | S2T170 |
| 腿较短，头较短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S3T266 |
| 只有尾巴较短。 | 1 | 0.250 | exclusive_case:脖子 > 0.50; exclusive_case:头 > 0.50; exclusive_case:腿 > 0.50 | S3T232 |
| 只有脖子极短。 | 1 | 0.250 | exclusive_case:头 > 0.50; exclusive_case:腿 > 0.50; exclusive_case:尾巴 > 0.50 | S4T240 |
| 只有腿最短，脖子较短。 | 1 | 0.250 | superlative:腿 < 脖子; superlative:腿 < 头; superlative:腿 < 尾巴; exclusive_case:腿 < 0.50 | S4T103 |
| 头和脖子较腿、尾巴及长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S2T246 |
| 头较短，其余三个差不多长。 | 1 | 0.250 | complement:脖子 < 0.50; complement:腿 < 0.50; complement:尾巴 < 0.50 | S2T106 |
| 四个部位长度差不多，头和脖子略长一些。 | 1 | 0.333 | equality_range:脖子+头+腿+尾巴 =; absolute_long:脖子 > 0.50 | S2T60 |
| 四个部位长度差不多，头和脖子较短一些。 | 1 | 0.333 | equality_range:脖子+头+腿+尾巴 =; absolute_short:头 < 0.50 | S2T62 |
| 头和脖子最长。 | 1 | 0.333 | superlative:脖子 > 头; superlative:脖子 > 腿 | S1T191 |
| 腿、头、尾巴较长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S5T109 |
| 腿较其他三个部位较短。 | 1 | 0.333 | complement:脖子 < 0.50; complement:头 < 0.50 | S2T313 |
| 腿较，尾巴和脖子较短，头较短。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S2T95 |
| 只有腿适中长，其余部位都很长。 | 1 | 0.429 | exclusive_case:腿 > 0.50; exclusive_case:脖子 < 0.50; exclusive_case:头 < 0.50; exclusive_case:尾巴 < 0.50 | S3T72 |
| 腿较长，其余部位均适中。 | 1 | 0.429 | absolute_long:腿 > 0.50; complement:脖子 middle_lower; complement:头 middle_lower; complement:尾巴 middle_lower | S4T209 |

### S225

- trial 数: 704; 非空文本: 701; fidelity 可评分率: 0.993; 平均 fidelity: 0.908; 完全忠实率: 0.795; 低 fidelity 率: 0.028.
- 旧版 region 覆盖率: 0.993; 旧版 region 有未处理片段率: 0.003.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 701 | 0.996 |
| superlative | 59 | 0.084 |
| comparison | 42 | 0.060 |
| body_ref | 24 | 0.034 |
| equality | 8 | 0.011 |
| empty | 3 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，腿短。 | 90 |
| 头长，腿长。 | 69 |
| 脖子长，腿短。 | 43 |
| 头短，脖子短。 | 38 |
| 脖子长，头短。 | 28 |
| 头长。 | 26 |
| 脖子长。 | 23 |
| 脖子最长。 | 23 |
| 各部位都较短。 | 15 |
| 腿长。 | 14 |
| 各部位都较长。 | 14 |
| 头比脖子长。 | 13 |
| 脖子长，腿长。 | 13 |
| 头短，脖子长。 | 13 |
| 身体各部位都较长。 | 10 |
| 脖子比头长。 | 9 |
| 腿短。 | 9 |
| 腿最长。 | 8 |
| 头长，脖子短。 | 8 |
| 除腿外，其余较短。 | 8 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 身体各部位都较长。 | 10 | body_ref | S1T210, S1T211, S1T218, S1T236, S1T257, S1T270, S2T1, S2T8 |
| 身体各部位都较短。 | 6 | body_ref | S1T224, S1T280, S1T287, S2T2, S2T13, S2T15 |
| 除腿外，身体各部位都较短。 | 3 | body_ref | S1T284, S1T285, S2T9 |
| 除尾巴外，身体各部位都较短。 | 2 | body_ref | S1T283, S2T10 |
| 各部位都长度差不多。 | 1 | equality | S1T188 |
| 头、脖子、腿、尾巴一样长。 | 1 | equality | S1T128 |
| 头、脖子、腿、尾巴长度差不多。 | 1 | equality | S1T68 |
| 头和尾巴一样长，腿短。 | 1 | equality | S1T85 |
| 头和脖子一样长。 | 1 | equality | S1T179 |
| 头和脖子长度差不多。 | 1 | equality | S1T181 |
| 尾巴和腿长度差不多。 | 1 | equality | S1T176 |
| 脖子和头一样长。 | 1 | equality | S2T303 |
| 身体各部位都适中。 | 1 | body_ref | S1T228 |
| 除尾巴外，身体各部位都较长。 | 1 | body_ref | S1T219 |
| 除腿外，身体各部位都较长。 | 1 | body_ref | S1T279 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 除腿外，各部位都较短。 | 2 | 0.000 | exclusion:脖子 < 0.50; exclusion:头 < 0.50; exclusion:尾巴 < 0.50; exclusion:腿 > 0.50 | S2T23, S2T57 |
| 身体各部位都较短。 | 2 | 0.125 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50 | S1T280, S2T13 |
| 各部位都长度差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T188 |
| 头、脖子、腿、尾巴一样长。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T128 |
| 头、脖子、腿、尾巴长度差不多。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T68 |
| 头短。 | 1 | 0.000 | absolute_short:头 < 0.50 | S1T17 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T52 |
| 头长，脖子短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_short:脖子 < 0.50 | S2T116 |
| 头长，腿短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_short:腿 < 0.50 | S2T290 |
| 尾巴和腿长度差不多。 | 1 | 0.000 | equality_range:尾巴+腿 = | S1T176 |
| 尾巴最长。 | 1 | 0.000 | superlative:尾巴 > 脖子; superlative:尾巴 > 头; superlative:尾巴 > 腿 | S1T177 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T136 |
| 腿长，其余较短。 | 1 | 0.000 | absolute_long:腿 > 0.50; complement:脖子 < 0.50; complement:头 < 0.50; complement:尾巴 < 0.50 | S1T259 |
| 除头外，各部位都较短。 | 1 | 0.000 | exclusion:脖子 < 0.50; exclusion:腿 < 0.50; exclusion:尾巴 < 0.50; exclusion:头 > 0.50 | S1T317 |
| 除腿外，脖子最长。 | 1 | 0.000 | exclusion:脖子 > 0.50; exclusion:头 > 0.50; exclusion:尾巴 > 0.50; exclusion:腿 < 0.50 | S2T195 |
| 各部位都较长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T36 |
| 脖子长，其余各部位都较短。 | 1 | 0.250 | complement:头 < 0.50; complement:腿 < 0.50; complement:尾巴 < 0.50 | S1T226 |
| 除腿外，别的较短。 | 1 | 0.250 | exclusion:脖子 < 0.50; exclusion:头 < 0.50; exclusion:尾巴 < 0.50 | S2T68 |

### S226

- trial 数: 640; 非空文本: 636; fidelity 可评分率: 0.906; 平均 fidelity: 0.867; 完全忠实率: 0.709; 低 fidelity 率: 0.072.
- 旧版 region 覆盖率: 0.906; 旧版 region 有未处理片段率: 0.169.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 612 | 0.956 |
| comparison | 88 | 0.138 |
| equality | 61 | 0.095 |
| count_abstract | 28 | 0.044 |
| superlative | 13 | 0.020 |
| group_sum | 7 | 0.011 |
| body_ref | 6 | 0.009 |
| empty | 4 | 0.006 |
| other | 3 | 0.005 |
| meta | 1 | 0.002 |
| negation | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，尾巴短。 | 41 |
| 头短，腿短。 | 37 |
| 头短，腿长。 | 31 |
| 头长，尾巴长。 | 28 |
| 四个差不多。 | 12 |
| 两长两短。 | 12 |
| 三长一短。 | 11 |
| 尾巴短。 | 10 |
| 脖子短。 | 10 |
| 四个差不多长。 | 8 |
| 头短，尾巴长。 | 8 |
| 腿长。 | 7 |
| 脖子长。 | 6 |
| 腿短。 | 6 |
| 头短，脖子长。 | 5 |
| 头短，尾巴短。 | 5 |
| 头短。 | 4 |
| 尾巴长。 | 4 |
| 四个都长。 | 4 |
| 四个都短。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 两长两短。 | 12 | count_abstract | S1T46, S1T48, S1T49, S1T52, S1T56, S1T58, S1T61, S1T65 |
| 四个差不多。 | 12 | equality | S1T31, S1T80, S1T84, S1T190, S1T196, S1T257, S1T261, S1T265 |
| 三长一短。 | 11 | count_abstract | S1T26, S1T32, S1T45, S1T50, S1T53, S1T57, S1T66, S1T74 |
| 四个差不多长。 | 8 | equality | S1T5, S1T6, S1T19, S1T70, S1T197, S1T230, S1T233, S2T6 |
| 三个差不多长。 | 4 | equality | S1T67, S1T72, S1T78, S1T151 |
| 腿和头差不多长。 | 3 | equality | S1T133, S1T144, S1T153 |
| 三长一短，尾巴短。 | 2 | count_abstract | S1T27, S1T29 |
| 四个差不多，像马。 | 2 | equality | S1T91, S1T132 |
| 四个差不多，都挺长。 | 2 | equality | S1T268, S1T269 |
| 腿和尾巴差不多长。 | 2 | equality | S1T143, S1T152 |
| 腿短，另外三个差不多。 | 2 | equality | S1T173, S2T109 |
| 三个差不多。 | 1 | equality | S1T82 |
| 三长一短，头短。 | 1 | count_abstract | S1T35 |
| 不知道什么规律，随便选的。 | 1 | meta | S1T127 |
| 两个差不多长。 | 1 | equality | S1T69 |
| 两长两短，尾巴短。 | 1 | count_abstract | S1T76 |
| 像马，四个都差不多。 | 1 | equality | S1T90 |
| 四个依次变化。 | 1 | other | S1T42 |
| 四个加起来有点短。 | 1 | group_sum | S1T242 |
| 四个加起来比较长。 | 1 | group_sum | S1T241 |
| 四个差不多都挺长。 | 1 | equality | S1T166 |
| 四个差不多长，像马。 | 1 | equality | S1T107 |
| 四个差不多长，都挺长。 | 1 | equality | S1T188 |
| 四个差不多，加起来都挺长。 | 1 | equality, group_sum | S1T245 |
| 四个差不多，都长。 | 1 | equality | S1T270 |
| 头、腿、尾巴差不多。 | 1 | equality | S2T84 |
| 头和尾巴差不多长。 | 1 | equality | S1T136 |
| 头比躯干短，尾巴短。 | 1 | body_ref | S2T177 |
| 头短，其他都长，比躯干长。 | 1 | body_ref | S2T204 |
| 头短，尾巴比躯干略短一点。 | 1 | body_ref | S2T180 |
| 小马。 | 1 | other | S1T110 |
| 尾巴短，加起来短。 | 1 | group_sum | S1T254 |
| 尾巴长，头比躯干短点。 | 1 | body_ref | S2T182 |
| 差不多都挺长。 | 1 | equality | S1T234 |
| 差不多长。 | 1 | equality | S1T77 |
| 差不多，头有点短。 | 1 | equality | S1T276 |
| 点错了。 | 1 | other | S1T96 |
| 腿和另外三个不一样。 | 1 | equality, negation | S1T139 |
| 腿和头差不多长，另外两个差不多长。 | 1 | equality | S1T158 |
| 腿和尾巴差不多。 | 1 | equality | S1T140 |
| 腿和尾巴差不多长，另外两个很长。 | 1 | equality | S1T145 |
| 腿比较长，四个都差不多。 | 1 | equality | S1T134 |
| 腿短三个长，加起来还挺长。 | 1 | group_sum | S1T247 |
| 腿短，其他三个差不多。 | 1 | equality | S2T83 |
| 腿短，和尾巴差不多，另外两个很长。 | 1 | equality | S1T137 |
| 腿短，脖子短，加起来一般。 | 1 | group_sum | S1T253 |
| 腿长，其他三个差不多。 | 1 | equality | S2T82 |
| 腿长，其他两长一短。 | 1 | count_abstract | S1T207 |
| 腿长，另外三个差不多，像马。 | 1 | equality | S1T126 |
| 腿长，四个差不多。 | 1 | equality | S1T208 |
| 这个差不多。 | 1 | equality | S1T81 |
| 这个差不多长，像马。 | 1 | equality | S1T106 |
| 都挺长，加起来挺长。 | 1 | group_sum | S1T248 |
| 都比身子短。 | 1 | body_ref | S2T201 |
| 都跟躯干差不多长。 | 1 | equality, body_ref | S2T210 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个差不多。 | 12 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T31, S1T80, S1T84, S1T190, S1T196, S1T257, S1T261, S1T265 |
| 四个差不多长。 | 8 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T5, S1T6, S1T19, S1T70, S1T197, S1T230, S1T233, S2T6 |
| 四个差不多，像马。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T91, S1T132 |
| 四个差不多，都挺长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T268, S1T269 |
| 腿和尾巴差不多长。 | 2 | 0.000 | equality_range:腿+尾巴 = | S1T143, S1T152 |
| 腿长两个短。 | 2 | 0.000 | absolute_short:腿 < 0.50 | S1T203, S1T204 |
| 三短一长，脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T34 |
| 像马，四个都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T90 |
| 四个差不多都挺长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T166 |
| 四个差不多长，像马。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T107 |
| 四个差不多长，都挺长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T188 |
| 四个差不多，加起来都挺长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T245 |
| 四个差不多，都长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T270 |
| 头、腿、尾巴差不多。 | 1 | 0.000 | equality_range:头+腿+尾巴 = | S2T84 |
| 头和尾巴差不多长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T136 |
| 头长，尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T171 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T219 |
| 尾巴长，两个长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T292 |
| 尾巴长，有一个短。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T287 |
| 腿和另外三个不一样。 | 1 | 0.000 | equality_range:脖子+头+尾巴 = | S1T139 |
| 腿和头差不多长。 | 1 | 0.000 | equality_range:腿+头 = | S1T153 |
| 腿和头差不多长，另外两个差不多长。 | 1 | 0.000 | equality_range:腿+头 =; complement:脖子 < 0.50; complement:尾巴 < 0.50 | S1T158 |
| 腿比脖子长。 | 1 | 0.000 | comparison:腿 > 脖子 | S1T62 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T313 |

### S227

- trial 数: 448; 非空文本: 448; fidelity 可评分率: 0.960; 平均 fidelity: 0.929; 完全忠实率: 0.824; 低 fidelity 率: 0.036.
- 旧版 region 覆盖率: 0.960; 旧版 region 有未处理片段率: 0.047.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 448 | 1.000 |
| equality | 53 | 0.118 |
| superlative | 11 | 0.025 |
| comparison | 9 | 0.020 |
| count_abstract | 2 | 0.004 |
| body_ref | 2 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长。 | 34 |
| 尾巴长。 | 33 |
| 脖子长。 | 19 |
| 尾巴长，脖子短。 | 16 |
| 脖子长，尾巴长。 | 15 |
| 脖子、腿、尾巴短。 | 13 |
| 脖子、尾巴长。 | 13 |
| 脖子、腿、尾巴长。 | 11 |
| 脖子较长。 | 10 |
| 脖子长，腿长。 | 10 |
| 尾巴短，腿长。 | 10 |
| 四个部位都较短。 | 9 |
| 头长。 | 9 |
| 四个部位都较长。 | 8 |
| 四个部位长度各不相同。 | 7 |
| 腿长，尾巴长。 | 6 |
| 尾巴长，腿长。 | 6 |
| 脖子、腿、尾巴都短。 | 6 |
| 腿较长。 | 5 |
| 脖子、腿长。 | 5 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子和尾巴长度相似。 | 5 | equality | S1T82, S1T84, S1T85, S1T87, S1T89 |
| 头和尾巴差不多长。 | 4 | equality | S1T120, S1T122, S1T153, S1T158 |
| 头和脖子一样长。 | 4 | equality | S1T18, S1T29, S1T113, S1T117 |
| 头、尾巴一样长。 | 3 | equality | S1T55, S1T56, S1T149 |
| 头和尾巴一样长。 | 3 | equality | S1T116, S1T137, S1T138 |
| 头、尾巴差不多长。 | 2 | equality | S1T47, S1T146 |
| 头、尾巴长度相似。 | 2 | equality | S1T134, S1T160 |
| 脖子和尾巴一样长。 | 2 | equality | S1T118, S1T131 |
| 四个部位长度相似。 | 1 | equality | S1T62 |
| 四个部位长度相近。 | 1 | equality | S1T74 |
| 头、尾差不多长。 | 1 | equality | S1T145 |
| 头、尾长度相近。 | 1 | equality | S1T73 |
| 头、脖子、尾巴、腿差不多长。 | 1 | equality | S1T21 |
| 头、脖子、尾长度相近，腿短。 | 1 | equality | S1T143 |
| 头、脖子和腿差不多长，尾巴短。 | 1 | equality | S1T121 |
| 头、脖子差不多长。 | 1 | equality | S1T44 |
| 头、脖子差不多长，腿和尾巴较短。 | 1 | equality | S1T23 |
| 头、腿长度相近。 | 1 | equality | S1T75 |
| 头和脖子一样长，腿较长，尾巴最长。 | 1 | equality | S1T16 |
| 头和脖子差不多长。 | 1 | equality | S1T156 |
| 头和脖子差不多长，腿和尾巴差不多长。 | 1 | equality | S1T51 |
| 尾巴和腿差不多长。 | 1 | equality | S1T155 |
| 尾巴短，头和脖子一样长。 | 1 | equality | S1T13 |
| 有两个部位的长度超过了躯干。 | 1 | body_ref, count_abstract | S1T107 |
| 脖子、尾巴一样长。 | 1 | equality | S1T144 |
| 脖子、腿、尾巴差不多长。 | 1 | equality | S1T31 |
| 脖子、腿长度相似。 | 1 | equality | S1T64 |
| 脖子、腿长度相近。 | 1 | equality | S1T78 |
| 脖子和尾巴一样长，头短。 | 1 | equality | S1T170 |
| 脖子和尾巴差不多长。 | 1 | equality | S1T154 |
| 脖子和腿一样长。 | 1 | equality | S1T119 |
| 脖子和腿差不多长。 | 1 | equality | S1T69 |
| 脖子和腿长度相似。 | 1 | equality | S1T83 |
| 脖子和腿长度相近，头最长。 | 1 | equality | S1T142 |
| 脖子最长，头和尾巴差不多长。 | 1 | equality | S1T46 |
| 脖子短，其余三个部位较长，且差不多长。 | 1 | equality, count_abstract | S1T53 |
| 脖子长度和躯干近似。 | 1 | body_ref | S1T93 |
| 腿短，脖子和尾巴差不多长。 | 1 | equality | S1T45 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度相似。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T62 |
| 四个部位长度相近。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T74 |
| 头、尾差不多长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T145 |
| 头、尾巴差不多长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T47 |
| 头、尾长度相近。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T73 |
| 头、脖子、尾巴、腿差不多长。 | 1 | 0.000 | equality_range:头+脖子+尾巴+腿 = | S1T21 |
| 头、腿长度相近。 | 1 | 0.000 | equality_range:头+腿 = | S1T75 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S2T72 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T198 |
| 脖子、腿、尾巴差不多长。 | 1 | 0.000 | equality_range:脖子+腿+尾巴 = | S1T31 |
| 脖子和尾巴长度相似。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T85 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T40 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T52 |
| 脖子、尾巴长，头短。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T61 |
| 脖子、腿、尾巴短。 | 1 | 0.333 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T311 |
| 脖子、腿、尾巴长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T281 |

### S228

- trial 数: 640; 非空文本: 609; fidelity 可评分率: 0.920; 平均 fidelity: 0.864; 完全忠实率: 0.680; 低 fidelity 率: 0.056.
- 旧版 region 覆盖率: 0.920; 旧版 region 有未处理片段率: 0.031.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 587 | 0.917 |
| comparison | 117 | 0.183 |
| superlative | 35 | 0.055 |
| empty | 31 | 0.048 |
| equality | 24 | 0.037 |
| group_sum | 22 | 0.034 |
| count_abstract | 1 | 0.002 |
| negation | 1 | 0.002 |
| body_ref | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子短。 | 34 |
| 脖子短，腿短。 | 30 |
| 腿短，其他长。 | 25 |
| 腿短。 | 24 |
| 脖子短，腿长。 | 23 |
| 脖子长，尾巴长。 | 22 |
| 头和脖子总和长，尾巴短。 | 16 |
| 脖子长，尾巴短。 | 16 |
| 头和脖子长。 | 15 |
| 脖子短，其他长。 | 13 |
| 尾巴长。 | 12 |
| 比较均匀。 | 10 |
| 脖子长。 | 9 |
| 腿最短。 | 8 |
| 脖子短，尾巴长。 | 8 |
| 脖子短，头长。 | 7 |
| 头、尾巴长。 | 7 |
| 尾巴短。 | 7 |
| 脖子最短。 | 7 |
| 头长，脖子长。 | 7 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头和脖子总和长，尾巴短。 | 16 | group_sum | S1T206, S1T219, S1T220, S1T221, S1T224, S1T226, S1T238, S1T239 |
| 比较均匀。 | 10 | equality | S1T146, S2T79, S2T82, S2T88, S2T96, S2T98, S2T102, S2T125 |
| 四个部位比较均匀。 | 6 | equality | S1T195, S1T199, S1T211, S1T232, S1T250, S2T50 |
| 均匀。 | 5 | equality | S2T104, S2T107, S2T110, S2T112, S2T116 |
| 腿长，其他均匀。 | 2 | equality | S2T117, S2T118 |
| 三个都在躯干上面。 | 1 | body_ref | S1T270 |
| 三长一短。 | 1 | count_abstract | S1T77 |
| 头、脖子、腿一样长。 | 1 | equality | S1T30 |
| 头和脖子总和比较长。 | 1 | group_sum | S2T1 |
| 头和脖子总和比较长，尾巴比较短。 | 1 | group_sum | S2T3 |
| 头和脖子总和短，尾巴长。 | 1 | group_sum | S1T207 |
| 头和脖子总和长，腿长。 | 1 | group_sum | S1T223 |
| 脖子不是最短。 | 1 | negation | S1T87 |
| 腿短，头和脖子总和长。 | 1 | group_sum | S2T4 |
| 腿短，头和脖子总和长，尾巴短。 | 1 | group_sum | S1T320 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子短。 | 8 | 0.000 | absolute_short:脖子 < 0.50 | S1T84, S1T86, S1T121, S1T156, S1T276, S1T295, S2T157, S2T171 |
| 四个部位比较均匀。 | 6 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T195, S1T199, S1T211, S1T232, S1T250, S2T50 |
| 脖子短，其他长。 | 2 | 0.250 | complement:头 > 0.50; complement:尾巴 > 0.50; complement:腿 > 0.50; absolute_short:脖子 < 0.50 | S1T53, S1T290 |
| 头和脖子总和长，尾巴短。 | 2 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_short:尾巴 < 0.50 | S1T238, S1T249 |
| 头、尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T136 |
| 头比脖子长。 | 1 | 0.000 | comparison:头 > 脖子 | S2T190 |
| 头短，腿长。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_long:腿 > 0.50 | S1T175 |
| 头长，脖子短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_short:脖子 < 0.50 | S1T142 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T173 |
| 脖子、头比较长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T131 |
| 脖子、腿长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S1T157 |
| 脖子和腿比较短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T35 |
| 脖子比尾巴长。 | 1 | 0.000 | comparison:脖子 > 尾巴 | S1T96 |
| 脖子比较短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S1T200 |
| 脖子长，尾巴短。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_short:尾巴 < 0.50 | S1T89 |
| 腿短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S2T103 |
| 除了腿，都比较长。 | 1 | 0.000 | exclusion:脖子 > 0.50; exclusion:头 > 0.50; exclusion:尾巴 > 0.50; exclusion:腿 < 0.50 | S2T41 |
| 只有尾巴短。 | 1 | 0.250 | exclusive_case:脖子 > 0.50; exclusive_case:头 > 0.50; exclusive_case:腿 > 0.50 | S1T254 |
| 尾巴短，其他长。 | 1 | 0.250 | absolute_short:尾巴 < 0.50; complement:脖子 > 0.50; complement:腿 > 0.50 | S1T193 |
| 腿比较长，其他比较短。 | 1 | 0.250 | absolute_long:腿 > 0.50; complement:脖子 < 0.50; complement:尾巴 < 0.50 | S1T188 |
| 脖子最短。 | 1 | 0.333 | superlative:脖子 < 头; superlative:脖子 < 尾巴 | S1T41 |
| 腿最短。 | 1 | 0.333 | superlative:腿 < 头; superlative:腿 < 尾巴 | S1T70 |

### S229

- trial 数: 1088; 非空文本: 1086; fidelity 可评分率: 0.998; 平均 fidelity: 0.932; 完全忠实率: 0.903; 低 fidelity 率: 0.040.
- 旧版 region 覆盖率: 0.998; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1086 | 0.998 |
| equality | 22 | 0.020 |
| superlative | 22 | 0.020 |
| empty | 2 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿短。 | 105 |
| 尾巴短。 | 97 |
| 头短。 | 86 |
| 腿长。 | 80 |
| 脖子短。 | 80 |
| 腿长，尾巴短。 | 63 |
| 腿长，尾巴长。 | 56 |
| 脖子长。 | 39 |
| 头短，腿短。 | 38 |
| 腿短，头长。 | 34 |
| 头长，腿短。 | 33 |
| 头长，腿长。 | 30 |
| 头短，腿长。 | 29 |
| 头长。 | 27 |
| 腿短，头短。 | 24 |
| 腿短，尾巴短。 | 20 |
| 尾巴长。 | 17 |
| 腿长，脖子短。 | 17 |
| 头长，尾巴短。 | 11 |
| 头短，尾巴短。 | 11 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子和腿一样长。 | 8 | equality | S1T1, S1T2, S1T4, S1T13, S1T38, S1T44, S1T53, S1T65 |
| 头和腿一样长。 | 5 | equality | S1T28, S1T31, S1T42, S1T47, S1T57 |
| 头和尾巴一样长。 | 3 | equality | S1T3, S1T7, S1T8 |
| 头和脖子一样长。 | 3 | equality | S1T12, S1T16, S1T48 |
| 脖子和尾巴一样长。 | 1 | equality | S1T30 |
| 腿和尾巴一样长。 | 1 | equality | S1T14 |
| 腿和脖子一样短。 | 1 | equality | S1T52 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头短。 | 7 | 0.000 | absolute_short:头 < 0.50 | S2T81, S2T102, S2T136, S2T156, S2T188, S2T211, S2T252 |
| 尾巴短。 | 7 | 0.000 | absolute_short:尾巴 < 0.50 | S1T193, S2T44, S2T148, S2T170, S2T192, S2T223, S3T1 |
| 脖子短。 | 6 | 0.000 | absolute_short:脖子 < 0.50 | S2T51, S2T87, S2T127, S2T180, S2T298, S3T4 |
| 腿短。 | 6 | 0.000 | absolute_short:腿 < 0.50 | S1T34, S2T185, S3T15, S3T21, S3T23, S3T34 |
| 头短，尾巴短。 | 2 | 0.000 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S3T73, S3T81 |
| 头短，腿短。 | 2 | 0.000 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50 | S3T120, S3T136 |
| 头长，尾巴短。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_short:尾巴 < 0.50 | S1T141, S3T69 |
| 腿长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S2T19, S3T231 |
| 头和尾巴一样长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T7 |
| 头和尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T167 |
| 头和腿一样长。 | 1 | 0.000 | equality_range:头+腿 = | S1T42 |
| 头短，脖子短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50 | S2T282 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S2T320 |
| 脖子、腿短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T80 |
| 脖子和尾巴长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T183 |
| 腿短，尾巴短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S4T23 |
| 腿短，脖子短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50 | S3T95 |
| 腿长，尾巴长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S3T316 |

### S230

- trial 数: 640; 非空文本: 638; fidelity 可评分率: 0.997; 平均 fidelity: 0.894; 完全忠实率: 0.816; 低 fidelity 率: 0.045.
- 旧版 region 覆盖率: 0.997; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 630 | 0.984 |
| superlative | 74 | 0.116 |
| equality | 15 | 0.023 |
| comparison | 7 | 0.011 |
| empty | 2 | 0.003 |
| body_ref | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子长，头短。 | 52 |
| 脖子短，尾巴短。 | 43 |
| 脖子长，头长。 | 40 |
| 头长，脖子长。 | 37 |
| 脖子短，尾巴长。 | 35 |
| 头长，尾巴长。 | 31 |
| 头长。 | 22 |
| 脖子长。 | 21 |
| 头长，腿长。 | 21 |
| 脖子最长。 | 16 |
| 腿最长。 | 15 |
| 腿长。 | 15 |
| 头长，脖子短。 | 14 |
| 头和脖子都长。 | 13 |
| 头短，脖子短。 | 13 |
| 头最长。 | 8 |
| 腿最短。 | 8 |
| 四个部位差不多长。 | 7 |
| 脖子最短。 | 7 |
| 腿短。 | 7 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位差不多长。 | 7 | equality | S1T26, S1T43, S1T68, S1T72, S1T95, S1T119, S1T198 |
| 四个部位差不多。 | 5 | equality | S1T104, S1T116, S1T214, S1T226, S1T231 |
| 四个部位都差不多。 | 1 | equality | S1T84 |
| 头和脖子差不多。 | 1 | equality | S2T167 |
| 脖子和头差不多。 | 1 | equality | S1T167 |
| 脖子比躯干长。 | 1 | body_ref | S1T244 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位差不多长。 | 7 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T26, S1T43, S1T68, S1T72, S1T95, S1T119, S1T198 |
| 四个部位差不多。 | 4 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T104, S1T214, S1T226, S1T231 |
| 头长，脖子长。 | 4 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T161, S1T251, S2T94, S2T173 |
| 头长。 | 3 | 0.000 | absolute_long:头 > 0.50 | S1T175, S1T229, S2T20 |
| 脖子长。 | 2 | 0.000 | absolute_long:脖子 > 0.50 | S1T173, S2T135 |
| 四个部位都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T84 |
| 头和脖子长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T177 |
| 头长，尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T71 |
| 脖子和头长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S2T19 |
| 腿最短。 | 1 | 0.000 | superlative:腿 < 脖子; superlative:腿 < 头; superlative:腿 < 尾巴 | S1T127 |
| 腿比其他长。 | 1 | 0.000 | comparison:腿 > 脖子+头+尾巴 | S1T265 |
| 腿短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S1T260 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T222 |
| 腿短，头长，脖子长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T51 |

### S231

- trial 数: 1344; 非空文本: 1342; fidelity 可评分率: 0.995; 平均 fidelity: 0.957; 完全忠实率: 0.902; 低 fidelity 率: 0.012.
- 旧版 region 覆盖率: 0.995; 旧版 region 有未处理片段率: 0.017.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1340 | 0.997 |
| comparison | 300 | 0.223 |
| superlative | 238 | 0.177 |
| equality | 96 | 0.071 |
| ranking | 6 | 0.004 |
| empty | 2 | 0.001 |
| body_ref | 2 | 0.001 |
| count_abstract | 1 | 0.001 |
| other | 1 | 0.001 |
| negation | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴短，头长。 | 45 |
| 腿长，脖子短。 | 33 |
| 尾巴长，脖子短。 | 33 |
| 头比腿长，脖子比尾巴长。 | 31 |
| 腿比头长，尾巴比脖子长。 | 28 |
| 腿比头长，脖子比尾巴长。 | 26 |
| 尾巴最短。 | 25 |
| 头比腿长，尾巴比脖子长。 | 23 |
| 腿最短。 | 22 |
| 头短，尾巴长。 | 21 |
| 脖子短。 | 18 |
| 尾巴短，头短。 | 15 |
| 尾巴长，头短。 | 15 |
| 尾巴长，脖子长。 | 14 |
| 尾巴比头长，脖子比腿长。 | 14 |
| 脖子最短。 | 14 |
| 脖子短，头长。 | 14 |
| 腿长，尾巴短。 | 14 |
| 腿短，尾巴长。 | 13 |
| 头长，尾巴短。 | 13 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿比头长，脖子和尾巴差不多。 | 8 | equality | S3T161, S3T165, S3T192, S3T206, S3T213, S3T228, S3T246, S3T290 |
| 头比腿长，脖子和尾巴差不多。 | 7 | equality | S3T182, S3T191, S3T202, S3T214, S3T292, S3T295, S3T306 |
| 头比尾巴长，脖子和腿差不多。 | 3 | equality | S3T87, S3T102, S3T154 |
| 腿比头长，尾巴和脖子差不多。 | 3 | equality | S3T164, S3T186, S3T239 |
| 各个部位差不多长。 | 2 | equality | S1T17, S1T18 |
| 各个部分差不多长。 | 2 | equality | S1T302, S3T125 |
| 四个部位差不多长。 | 2 | equality | S1T192, S2T232 |
| 尾巴比脖子长，头和腿差不多。 | 2 | equality | S3T189, S3T200 |
| 腿比脖子长，头和腿差不多。 | 2 | equality | S3T215, S3T296 |
| 腿长，脖子短，头和尾巴差不多。 | 2 | equality | S3T67, S3T97 |
| 各个部分差不多长，脖子、尾巴一样长。 | 1 | equality | S1T286 |
| 各个部分看上去差不多，腿和尾巴略短。 | 1 | equality | S1T160 |
| 各部位差不多一样长。 | 1 | equality | S1T307 |
| 各部位差不多都比较长。 | 1 | equality | S3T54 |
| 各部位差不多长。 | 1 | equality | S2T18 |
| 四个部位差不多都比较长。 | 1 | equality | S2T54 |
| 四个部位差不多长，脖子稍长一些。 | 1 | equality | S1T271 |
| 头、尾巴、脖子差不多，腿略长一些。 | 1 | equality | S1T152 |
| 头、脖子、尾巴差不多长。 | 1 | equality | S1T251 |
| 头、脖子、尾巴短且差不多，腿最长。 | 1 | equality | S1T186 |
| 头和尾巴差不多长。 | 1 | equality | S1T8 |
| 头和脖子是最长的且差不多，尾巴最短。 | 1 | equality | S1T39 |
| 头和脖子最长且差不多，尾巴和腿都比较短且差不多。 | 1 | equality | S1T82 |
| 头和腿差不多长，尾巴和脖子也差不多长。 | 1 | equality | S3T286 |
| 头和腿差不多长，尾巴比脖子长。 | 1 | equality | S3T289 |
| 头和腿最短且差不多。 | 1 | equality | S1T53 |
| 头最短，其他部位差不多。 | 1 | equality | S1T34 |
| 头最短，尾巴和脖子差不多长。 | 1 | equality | S1T48 |
| 头最短，脖子和尾巴差不多长，都长。 | 1 | equality | S1T73 |
| 头最短，脖子第二短。 | 1 | ranking | S1T88 |
| 头最长，其他三个部分差不多，比较短。 | 1 | equality | S1T64 |
| 头最长，其他部位稍短、长度差不多。 | 1 | equality | S1T75 |
| 头最长，尾巴和脖子差不多长。 | 1 | equality | S1T20 |
| 头最长，尾巴次之。 | 1 | ranking | S1T65 |
| 头最长，腿第二长。 | 1 | ranking | S1T66 |
| 头比尾巴长，腿和脖子差不多。 | 1 | equality | S3T86 |
| 头比腿长，脖子、尾巴差不多。 | 1 | equality | S3T243 |
| 头短，其他部位也短，但是没有头短。 | 1 | negation | S2T34 |
| 头短，其他部位差不多长。 | 1 | equality | S2T2 |
| 头长，尾巴短，脖子和腿差不多。 | 1 | equality | S3T101 |
| 头长，尾巴长，脖子长，腿应该也相当于一半的长度。 | 1 | equality, body_ref | S4T10 |
| 尾巴和头差不多长，脖子和腿差不多长。 | 1 | equality | S3T77 |
| 尾巴和脖子。 | 1 | other | S1T140 |
| 尾巴和脖子差不多。 | 1 | equality | S3T318 |
| 尾巴和腿差不多长，头和脖子差不多长。 | 1 | equality | S3T40 |
| 尾巴和腿最短并且差不多，头最长。 | 1 | equality | S1T51 |
| 尾巴最短，其他部位差不多。 | 1 | equality | S1T43 |
| 尾巴最短，头和脖子差不多长。 | 1 | equality | S1T49 |
| 尾巴最短，腿第二短。 | 1 | ranking | S1T100 |
| 尾巴最长，其他部位稍短，并且长度差不多。 | 1 | equality | S1T30 |
| 尾巴最长，然后是脖子，其他两个部位差不多。 | 1 | equality, ranking, count_abstract | S1T22 |
| 尾巴比头长，腿和脖子差不多。 | 1 | equality | S3T121 |
| 差不多长，腿略短一点。 | 1 | equality | S2T143 |
| 整个身体各个部位差不多长。 | 1 | equality, body_ref | S1T125 |
| 脖子、尾巴差不多长，腿最短。 | 1 | equality | S1T10 |
| 脖子和尾巴差不多短，腿和头差不多长。 | 1 | equality | S1T55 |
| 脖子和腿差不多长。 | 1 | equality | S1T46 |
| 脖子最短，其他部位差不多、略长。 | 1 | equality | S1T155 |
| 脖子最短，其他部位差不多。 | 1 | equality | S1T237 |
| 脖子最短，其他部位差不多长。 | 1 | equality | S1T62 |
| 脖子最短，头和尾巴差不多长。 | 1 | equality | S1T44 |
| 脖子最短，尾巴和脖子差不多、都短。 | 1 | equality | S1T143 |
| 脖子最短，腿和头差不多长。 | 1 | equality | S1T38 |
| 脖子最短，腿和尾巴差不多长。 | 1 | equality | S1T72 |
| 脖子最短，腿和尾巴最长，并且差不多。 | 1 | equality | S1T31 |
| 脖子最长，腿可能和腿差不多。 | 1 | equality | S1T60 |
| 脖子比尾巴长，头和腿差不多。 | 1 | equality | S3T208 |
| 脖子比腿长，头和尾巴差不多。 | 1 | equality | S3T157 |
| 脖子短，尾巴、头差不多长。 | 1 | equality | S2T67 |
| 腿和头差不多长，脖子和尾巴也差不多长。 | 1 | equality | S3T277 |
| 腿和头都是最短的并且差不多。 | 1 | equality | S1T37 |
| 腿和尾巴最长，然后是头，最后是脖子短。 | 1 | ranking | S1T78 |
| 腿最长，其他部分差不多长。 | 1 | equality | S1T127 |
| 腿最长，头和脖子都短，并且差不多。 | 1 | equality | S1T35 |
| 腿比头长，脖子和尾巴差不多长。 | 1 | equality | S3T280 |
| 腿比尾巴长，脖子和头差不多。 | 1 | equality | S3T35 |
| 腿比脖子长一点，头和尾巴差不多。 | 1 | equality | S3T69 |
| 腿比脖子长，尾巴和头差不多。 | 1 | equality | S3T74 |
| 腿短，脖子和尾巴差不多。 | 1 | equality | S4T107 |
| 都差不多，头略短一些。 | 1 | equality | S1T156 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 各个部位差不多长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T17, S1T18 |
| 四个部位差不多长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T192, S2T232 |
| 各部位差不多一样长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T307 |
| 各部位差不多都比较长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T54 |
| 各部位差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T18 |
| 四个部位差不多都比较长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T54 |
| 头、脖子、尾巴差不多长。 | 1 | 0.000 | equality_range:头+脖子+尾巴 = | S1T251 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S2T238 |
| 脖子较短，各个部分都比较长。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S1T214 |
| 都差不多，头略短一些。 | 1 | 0.000 | absolute_short:头 < 0.50 | S1T156 |
| 头、尾巴、腿、脖子长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50 | S4T78 |
| 头短，其他部位差不多长。 | 1 | 0.250 | complement:脖子 < 0.50; complement:腿 < 0.50; complement:尾巴 < 0.50 | S2T2 |
| 尾巴脖子短，头、腿长。 | 1 | 0.250 | absolute_short:尾巴 < 0.50; absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T85 |
| 腿长，尾巴、脖子、头较短。 | 1 | 0.250 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S1T215 |

### S232

- trial 数: 512; 非空文本: 504; fidelity 可评分率: 0.984; 平均 fidelity: 0.931; 完全忠实率: 0.758; 低 fidelity 率: 0.004.
- 旧版 region 覆盖率: 0.984; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 504 | 0.984 |
| comparison | 222 | 0.434 |
| empty | 8 | 0.016 |
| equality | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，脖子短。 | 20 |
| 腿长，脖子长。 | 19 |
| 腿长，尾巴长，脖子长，头长。 | 18 |
| 腿短，尾巴长。 | 15 |
| 腿短，尾巴长，脖子长，头长。 | 14 |
| 腿短，尾巴短。 | 14 |
| 腿短，尾巴短，脖子长，头长。 | 14 |
| 腿短，尾巴长，脖子长，头短。 | 13 |
| 腿短，尾巴短，脖子短，头长。 | 12 |
| 腿长，尾巴短，脖子短，头长。 | 11 |
| 腿长，尾巴长，脖子短，头短。 | 11 |
| 腿长，尾巴长，脖子短，头长。 | 10 |
| 腿长，尾巴短，脖子短，头短。 | 10 |
| 腿比较短，尾巴比较长。 | 10 |
| 腿短，尾巴长，脖子短，头短。 | 10 |
| 腿短，尾巴长，脖子短，头长。 | 10 |
| 腿长，尾巴短，脖子长，头长。 | 10 |
| 腿短，尾巴短，脖子长，头短。 | 9 |
| 腿长，尾巴长，脖子长，头短。 | 9 |
| 腿长，尾巴短，脖子长，头短。 | 9 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿特别长，尾巴比较短，头和脖子差不多。 | 1 | equality | S1T38 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿比较短，脖子比较短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50 | S1T172 |
| 头比较短，脖子比较长，腿比较短。 | 1 | 0.333 | absolute_short:头 < 0.50; absolute_short:腿 < 0.50 | S1T175 |

### S301

- trial 数: 640; 非空文本: 640; fidelity 可评分率: 1.000; 平均 fidelity: 0.895; 完全忠实率: 0.759; 低 fidelity 率: 0.022.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.052.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 640 | 1.000 |
| comparison | 156 | 0.244 |
| equality | 37 | 0.058 |
| count_abstract | 13 | 0.020 |
| superlative | 6 | 0.009 |
| body_ref | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子长，头短。 | 72 |
| 脖子短，腿长。 | 67 |
| 脖子短，腿短。 | 53 |
| 脖子长，头长。 | 51 |
| 头长，脖子短。 | 17 |
| 头和脖子都很长。 | 6 |
| 脖子短，腿比脖子长。 | 4 |
| 头和脖子非常长。 | 4 |
| 头和脖子都非常长。 | 3 |
| 头、脖子和腿较长，尾巴中等。 | 3 |
| 脖子比腿长，头短。 | 3 |
| 脖子短，头长。 | 3 |
| 脖子长，头比脖子短。 | 3 |
| 脖子长，头比腿短。 | 3 |
| 脖子和头都很长，长度相近。 | 3 |
| 脖子长，头短，腿较长，尾巴较短。 | 2 |
| 脖子和头都很长。 | 2 |
| 脖子和头都很长，头比脖子长。 | 2 |
| 头长，脖子短，腿长。 | 2 |
| 脖子比腿长，头长。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子和头都很长，长度相近。 | 3 | equality | S1T309, S2T10, S2T11 |
| 四个部位长度比较均衡。 | 2 | equality | S1T289, S1T298 |
| 尾巴非常长，其他部位比较均衡。 | 2 | equality | S1T200, S1T211 |
| 各部位长度均衡，尾巴比较长。 | 1 | equality | S1T271 |
| 四个部位长度均衡。 | 1 | equality | S1T232 |
| 四个部位长度相近。 | 1 | equality | S1T31 |
| 四个部位长度相近，四个部位都比较长。 | 1 | equality | S1T11 |
| 头和尾巴相近，四个部位都长，脖子很短，腿较长。 | 1 | equality | S1T157 |
| 头和尾巴长度相近、都非常长，腿和脖子较短。 | 1 | equality | S1T99 |
| 头和脖子很长，腿也很长，三者长度相近。 | 1 | equality | S1T110 |
| 头和脖子都极长，其余两个部位较短。 | 1 | count_abstract | S1T32 |
| 头和脖子非常长，比其余两个部位长。 | 1 | count_abstract | S1T247 |
| 头较短，其余三个部位长度均衡。 | 1 | equality, count_abstract | S1T135 |
| 头长，脖子非常短，腿与头长度相近。 | 1 | equality | S1T89 |
| 尾巴和头相近，四个部位都长。 | 1 | equality | S1T154 |
| 尾巴和头长度相近，四个部位都比较长，其余部位较短。 | 1 | equality | S1T128 |
| 尾巴很长，头很长，两者长度相近，腿很短。 | 1 | equality | S1T138 |
| 尾巴极长，其他部位均衡。 | 1 | equality | S1T156 |
| 尾巴较长，其余三个部位偏短。 | 1 | count_abstract | S1T13 |
| 尾巴非常长，其他三个部位长度相对均衡。 | 1 | equality, count_abstract | S1T130 |
| 尾巴非常长，其余部位比较均衡。 | 1 | equality | S1T127 |
| 脖子和头都较短，长度相近，尾巴较长。 | 1 | equality | S2T20 |
| 脖子和头长度相近，都偏短。 | 1 | equality | S2T13 |
| 脖子很短，其余三个部位都很长。 | 1 | count_abstract | S1T27 |
| 脖子比躯干短，头更短。 | 1 | body_ref | S2T47 |
| 脖子非常长，其他三个部位长度比较短。 | 1 | count_abstract | S1T144 |
| 脖子非常长，其余三个部位较长，长度相近。 | 1 | equality, count_abstract | S1T114 |
| 脖子非常长，比其他三个部位长。 | 1 | count_abstract | S1T246 |
| 腿和头极长，脖子和尾巴相近，四个部位都偏短。 | 1 | equality | S1T35 |
| 腿很短，其余三个部位非常长。 | 1 | count_abstract | S1T10 |
| 腿很长，其他三个部位差不多长。 | 1 | equality, count_abstract | S1T276 |
| 腿很长，其余三个部位与腿相近。 | 1 | equality, count_abstract | S1T66 |
| 腿很长，其余三个部位较长，长度相近。 | 1 | equality, count_abstract | S1T94 |
| 腿很长，头和脖子相近，也都比较长。 | 1 | equality | S1T72 |
| 腿很长，尾巴、脖子和头长度相近。 | 1 | equality | S1T280 |
| 腿极短，头、脖子、尾巴都较长，长度相近。 | 1 | equality | S1T42 |
| 腿极长，头、脖子和尾巴相近，稍短一些。 | 1 | equality | S1T36 |
| 腿极长，头和尾巴都偏长、长度相近，脖子相对较短。 | 1 | equality | S1T41 |
| 腿较长，头和尾巴也较长、长度相近，脖子相对短一些。 | 1 | equality | S1T44 |
| 腿较长，头和脖子比较均衡，尾巴较长。 | 1 | equality | S1T131 |
| 长度比较均衡，相对来说脖子长一些。 | 1 | equality | S1T270 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度比较均衡。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T289, S1T298 |
| 四个部位长度均衡。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T232 |
| 四个部位长度相近。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T31 |
| 头和脖子相对比较长，脖子更长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T220 |
| 脖子和头长度相近，都偏短。 | 1 | 0.000 | equality_range:脖子+头 = | S2T13 |
| 脖子短，腿长。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_long:腿 > 0.50 | S2T200 |
| 尾巴很长，腿、头和脖子均较长。 | 1 | 0.250 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T104 |
| 尾巴较长，头和脖子都比较长，腿较短。 | 1 | 0.250 | absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T145 |
| 腿极短，头和脖子很长，尾巴也较长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T60 |
| 头和脖子都很长，腿长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T36 |
| 头长，脖子非常短，腿与头长度相近。 | 1 | 0.333 | absolute_long:头 > 0.50; equality_range:腿+头 = | S1T89 |
| 脖子长，头也长，相对来说脖子比头长的较少。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S2T29 |
| 尾巴和头长度相近，四个部位都比较长，其余部位较短。 | 1 | 0.429 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T128 |

### S302

- trial 数: 448; 非空文本: 446; fidelity 可评分率: 0.993; 平均 fidelity: 0.940; 完全忠实率: 0.846; 低 fidelity 率: 0.007.
- 旧版 region 覆盖率: 0.993; 旧版 region 有未处理片段率: 0.029.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 446 | 0.996 |
| comparison | 41 | 0.092 |
| negation | 9 | 0.020 |
| equality | 2 | 0.004 |
| empty | 2 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子短，头短。 | 73 |
| 脖子短，头长。 | 71 |
| 脖子长，腿短。 | 63 |
| 脖子长，腿长。 | 50 |
| 头长，脖子短。 | 12 |
| 脖子长，尾巴短。 | 7 |
| 脖子短。 | 5 |
| 头长于脖子。 | 4 |
| 脖子比头长很多。 | 4 |
| 头比脖子长。 | 4 |
| 脖子长，头长，腿短，尾巴短。 | 4 |
| 脖子长，头长，腿长，尾巴短。 | 4 |
| 脖子长，腿短，尾巴短。 | 4 |
| 四个部位都比较长。 | 3 |
| 脖子长，尾巴长，头短，腿短。 | 3 |
| 腿长，脖子长，头长，尾巴短。 | 3 |
| 脖子长，腿长，尾巴长。 | 3 |
| 脖子长。 | 3 |
| 四个部位都短。 | 3 |
| 脖子长，头短，尾巴短，腿短。 | 3 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿长，其他部位都不是很长，比较匀称。 | 2 | negation | S1T109, S1T136 |
| 头、腿、脖子、尾巴长度差不多。 | 1 | equality | S1T71 |
| 头相对长，其他部位不是很长，比较匀称。 | 1 | negation | S1T106 |
| 头长，脖子长，腿长，尾巴不是很短。 | 1 | negation | S1T95 |
| 尾巴长，脖子、头、腿都不是很长，整体比较匀称。 | 1 | negation | S1T93 |
| 脖子比头长，腿短，尾巴不是很短。 | 1 | negation | S1T140 |
| 脖子长，头短，尾巴长，腿不是很短。 | 1 | negation | S1T99 |
| 脖子长，腿长，尾巴长，头不短。 | 1 | negation | S1T177 |
| 腿长，脖子、头和尾巴差不多中等。 | 1 | equality | S1T155 |
| 都比较中等，都不是很长。 | 1 | negation | S1T146 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头、腿、脖子、尾巴长度差不多。 | 1 | 0.000 | equality_range:头+腿+脖子+尾巴 = | S1T71 |
| 头长，脖子长，尾巴长，腿短。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T53 |
| 腿长，脖子长，头长，尾巴短。 | 1 | 0.250 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T23 |

### S303

- trial 数: 192; 非空文本: 191; fidelity 可评分率: 0.995; 平均 fidelity: 0.932; 完全忠实率: 0.755; 低 fidelity 率: 0.000.
- 旧版 region 覆盖率: 0.995; 旧版 region 有未处理片段率: 0.005.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 191 | 0.995 |
| comparison | 6 | 0.031 |
| superlative | 1 | 0.005 |
| empty | 1 | 0.005 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头短，脖子短，尾巴短，腿长。 | 7 |
| 头长，脖子长，尾巴长，腿短。 | 7 |
| 头短，脖子长，腿长，尾巴短。 | 7 |
| 头长，腿长，脖子短，尾巴短。 | 5 |
| 头和腿长，脖子和尾巴短。 | 4 |
| 头长，脖子短，腿短，尾巴短。 | 4 |
| 头短，脖子短，腿长，尾巴短。 | 3 |
| 头长，脖子短，尾巴短，腿长。 | 3 |
| 头短，脖子长，尾巴短，腿短。 | 3 |
| 头长，脖子短，腿长，尾巴长。 | 3 |
| 头长，脖子长，腿长，尾巴短。 | 3 |
| 头短，脖子、腿和尾巴中等偏长。 | 2 |
| 头、脖子和尾巴中等偏长，腿中等偏短。 | 2 |
| 头短，脖子长，尾巴长，腿长。 | 2 |
| 头短，脖子短，腿短，尾巴长。 | 2 |
| 头和尾巴长，脖子和腿短。 | 2 |
| 头短，脖子长，腿短，尾巴短。 | 2 |
| 四个部位都中等偏长。 | 2 |
| 头短，脖子短，腿长，尾巴长。 | 2 |
| 头长，脖子短，腿长，尾巴短。 | 2 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
无。

### S304

- trial 数: 320; 非空文本: 319; fidelity 可评分率: 0.988; 平均 fidelity: 0.931; 完全忠实率: 0.903; 低 fidelity 率: 0.059.
- 旧版 region 覆盖率: 0.988; 旧版 region 有未处理片段率: 0.009.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 318 | 0.994 |
| equality | 16 | 0.050 |
| body_ref | 14 | 0.044 |
| empty | 1 | 0.003 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头偏长。 | 42 |
| 腿偏短。 | 41 |
| 腿偏长。 | 35 |
| 头偏短。 | 31 |
| 脖子偏长。 | 17 |
| 头短，脖子长。 | 16 |
| 头短，脖子短。 | 15 |
| 尾巴偏短。 | 13 |
| 尾巴偏长。 | 11 |
| 头长，腿短。 | 11 |
| 脖子偏短。 | 9 |
| 头偏长，腿偏长。 | 9 |
| 头偏短，脖子偏长。 | 8 |
| 头偏短，脖子偏短。 | 8 |
| 头偏长，腿偏短。 | 6 |
| 头长，腿长。 | 6 |
| 腿较短。 | 3 |
| 头、脖子、尾巴长度接近。 | 3 |
| 脖子与躯干长度接近。 | 2 |
| 尾巴和躯干长度接近。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头、脖子、尾巴长度接近。 | 3 | equality | S1T24, S1T26, S1T28 |
| 头和脖子长度接近。 | 2 | equality | S1T66, S1T67 |
| 尾巴和躯干长度接近。 | 2 | equality, body_ref | S1T34, S1T37 |
| 脖子与躯干长度接近。 | 2 | equality, body_ref | S1T75, S1T77 |
| 头和躯干长度相同，其他部位长度是躯干的0.7倍。 | 1 | body_ref | S1T6 |
| 头是躯干的1/3，其他部位是躯干的0.7倍。 | 1 | body_ref | S1T2 |
| 尾巴、躯干、腿长度接近。 | 1 | equality, body_ref | S1T33 |
| 尾巴与躯干长度接近。 | 1 | equality, body_ref | S1T42 |
| 尾巴较长，其他部位是躯干的0.7倍。 | 1 | body_ref | S1T7 |
| 尾巴长度是躯干的1.5倍，其他部位长度是躯干的0.5倍。 | 1 | body_ref | S1T1 |
| 所有部位的长度和躯干较为接近。 | 1 | equality, body_ref | S1T10 |
| 所有部位的长度和躯干长度较为接近。 | 1 | equality, body_ref | S1T12 |
| 脖子和躯干长度接近。 | 1 | equality, body_ref | S1T73 |
| 腿和尾巴长度接近。 | 1 | equality | S1T39 |
| 躯干和脖子长度接近。 | 1 | equality, body_ref | S1T40 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、尾巴长度接近。 | 2 | 0.000 | equality_range:头+脖子+尾巴 = | S1T26, S1T28 |
| 头偏短。 | 2 | 0.000 | absolute_short:头 < 0.50 | S1T99, S1T124 |
| 脖子与躯干长度接近。 | 2 | 0.000 | body_ref:脖子 = 0.50 | S1T75, S1T77 |
| 头偏长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T140 |
| 尾巴、躯干、腿长度接近。 | 1 | 0.000 | body_ref:尾巴 = 0.50; body_ref:腿 = 0.50; equality_range:尾巴+腿 = | S1T33 |
| 尾巴与躯干长度接近。 | 1 | 0.000 | body_ref:尾巴 = 0.50 | S1T42 |
| 尾巴偏短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T185 |
| 尾巴和躯干长度接近。 | 1 | 0.000 | body_ref:尾巴 = 0.50 | S1T34 |
| 尾巴长度是躯干的1.5倍，其他部位长度是躯干的0.5倍。 | 1 | 0.000 | body_ref:尾巴 = 0.50; body_ref:脖子 = 0.50; body_ref:头 = 0.50; body_ref:腿 = 0.50 | S1T1 |
| 所有部位的长度和躯干长度较为接近。 | 1 | 0.000 | body_ref:脖子 = 0.50; body_ref:头 = 0.50; body_ref:腿 = 0.50; body_ref:尾巴 = 0.50 | S1T12 |
| 脖子偏短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S1T49 |
| 脖子偏长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T228 |
| 腿偏短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S1T196 |
| 躯干和脖子长度接近。 | 1 | 0.000 | body_ref:脖子 = 0.50 | S1T40 |
| 所有部位的长度和躯干较为接近。 | 1 | 0.200 | body_ref:脖子 = 0.50; body_ref:头 = 0.50; body_ref:尾巴 = 0.50; equality_range:脖子+头+腿+尾巴 = | S1T10 |
| 头是躯干的1/3，其他部位是躯干的0.7倍。 | 1 | 0.250 | body_ref:头 = 0.50; body_ref:腿 = 0.50; body_ref:尾巴 = 0.50 | S1T2 |

### S305

- trial 数: 192; 非空文本: 192; fidelity 可评分率: 0.875; 平均 fidelity: 0.908; 完全忠实率: 0.667; 低 fidelity 率: 0.010.
- 旧版 region 覆盖率: 0.875; 旧版 region 有未处理片段率: 0.172.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 168 | 0.875 |
| other | 22 | 0.115 |
| comparison | 2 | 0.010 |
| negation | 1 | 0.005 |
| count_abstract | 1 | 0.005 |
| meta | 1 | 0.005 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，脖子短。 | 24 |
| 腿中等，头长。 | 21 |
| 腿长，脖子长。 | 17 |
| 腿短，头长。 | 16 |
| 腿短，头短。 | 12 |
| 挺高大。 | 8 |
| 腿中等，头短。 | 8 |
| 腿适中。 | 7 |
| 身材高大。 | 7 |
| 腿中等，头中等。 | 6 |
| 很高大。 | 6 |
| 腿短，头中等。 | 5 |
| 挺高大，脖子短。 | 5 |
| 腿适中，头长。 | 3 |
| 腿短。 | 3 |
| 腿适中，脖子长。 | 2 |
| 腿短，其他挺长。 | 2 |
| 腿中等，头很长。 | 2 |
| 腿长，脖子中等。 | 2 |
| 腿中等长度，头和脖子很长。 | 1 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 挺高大。 | 8 | other | S1T50, S1T51, S1T52, S1T54, S1T55, S1T56, S1T59, S1T61 |
| 身材高大。 | 7 | other | S1T70, S1T80, S1T82, S1T89, S1T92, S1T94, S1T96 |
| 很高大。 | 6 | other | S1T1, S1T12, S1T23, S1T24, S1T31, S1T46 |
| 很高。 | 1 | other | S1T3 |
| 腿很短，其他很长，头也不长。 | 1 | negation | S1T15 |
| 腿短，尾巴一般，其他两个部位很长。 | 1 | count_abstract | S1T21 |
| 选错了。 | 1 | meta | S1T163 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿中等，头长。 | 1 | 0.333 | absolute:腿 middle_lower; absolute_long:头 > 0.50 | S1T69 |
| 腿适中，头长。 | 1 | 0.333 | absolute:腿 middle_lower; absolute_long:头 > 0.50 | S1T28 |

### S306

- trial 数: 768; 非空文本: 768; fidelity 可评分率: 1.000; 平均 fidelity: 0.916; 完全忠实率: 0.809; 低 fidelity 率: 0.016.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.047.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 768 | 1.000 |
| comparison | 197 | 0.257 |
| equality | 81 | 0.105 |
| superlative | 19 | 0.025 |
| body_ref | 14 | 0.018 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿短，脖子长。 | 98 |
| 腿长，头短。 | 74 |
| 腿长，头长。 | 71 |
| 腿短，脖子短。 | 66 |
| 腿较长，头较短。 | 13 |
| 腿较长，头较长。 | 12 |
| 腿较短，脖子较长。 | 9 |
| 腿短，头比脖子长。 | 8 |
| 腿长，尾巴短。 | 7 |
| 腿长，头比脖子长。 | 7 |
| 腿短，头长。 | 6 |
| 腿较短，脖子较短。 | 4 |
| 腿较短，脖子短。 | 4 |
| 腿长，尾巴长。 | 4 |
| 腿短，尾巴长，脖子比头长。 | 4 |
| 腿短，尾巴长，头比脖子长。 | 4 |
| 腿长，头较短。 | 3 |
| 腿短，脖子很长。 | 3 |
| 腿短，头比脖子长，尾巴长。 | 3 |
| 腿较短，头较长。 | 3 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿短，头比脖子长，尾巴和脖子差不多一样长。 | 2 | equality | S1T271, S1T272 |
| 腿短，尾巴、脖子、头差不多一样长。 | 2 | equality | S1T139, S1T163 |
| 腿短，脖子、头、尾巴差不多一样长。 | 2 | equality | S1T235, S1T311 |
| 腿短，脖子和头差不多一样长，尾巴短。 | 2 | equality | S1T225, S1T226 |
| 腿短，脖子短，头和尾巴一样长。 | 2 | equality | S1T189, S1T190 |
| 腿长，头比脖子长，脖子和尾巴差不多长。 | 2 | equality | S1T78, S1T93 |
| 四个部位都差不多、较短，其中头更短一些。 | 1 | equality | S1T5 |
| 四个部位都差不多一样长、都较长。 | 1 | equality | S1T12 |
| 头和尾巴一样长、较短，腿适中，脖子较长。 | 1 | equality | S1T2 |
| 头和尾巴一样长，腿比较短，脖子略短。 | 1 | equality | S1T1 |
| 头比脖子短，其他部位差不多一样长、长度适中。 | 1 | equality | S1T41 |
| 脖子和腿差不多一样长，头更长，尾巴短。 | 1 | equality | S1T46 |
| 脖子和腿很长、一样长，头短，尾巴稍长一些。 | 1 | equality | S1T49 |
| 脖子比头长，和腿差不多，比腿稍长，尾巴最短。 | 1 | equality | S1T44 |
| 腿和头差不多一样长，长度较长，尾巴和脖子相对来说较短。 | 1 | equality | S1T42 |
| 腿和尾巴一样长，脖子较短，头较长。 | 1 | equality | S1T10 |
| 腿和尾巴一样长，较长，脖子非常长，头较短。 | 1 | equality | S1T11 |
| 腿很短，尾巴较长，头比脖子长，头和尾巴差不多一样长。 | 1 | equality | S2T26 |
| 腿很长，头、脖子、尾巴长度接近，四个部位都短。 | 1 | equality | S2T56 |
| 腿比躯干长，其他部位均较短。 | 1 | body_ref | S1T56 |
| 腿短，其他部位差不多。 | 1 | equality | S1T161 |
| 腿短，其他部位长度接近。 | 1 | equality | S1T92 |
| 腿短，在其他躯干中，脖子最短，头和尾巴差不多一样长。 | 1 | equality, body_ref | S1T164 |
| 腿短，头与脖子接近。 | 1 | equality | S1T282 |
| 腿短，头和尾巴脖接近。 | 1 | equality | S1T275 |
| 腿短，头和脖子接近。 | 1 | equality | S1T300 |
| 腿短，头比脖子长，和尾巴差不多长，是尾巴和脖子差不多长。 | 1 | equality | S1T302 |
| 腿短，头长，尾巴和脖子差不多一样短。 | 1 | equality | S1T102 |
| 腿短，尾巴、脖子、头差不多长，其中脖子最长。 | 1 | equality | S1T131 |
| 腿短，尾巴和脖子差不多长，脖子比头长。 | 1 | equality | S1T81 |
| 腿短，尾巴短，尾巴较短，头和脖子差不多一样长。 | 1 | equality | S1T151 |
| 腿短，尾巴短，脖子和头差不多一样长。 | 1 | equality | S1T136 |
| 腿短，尾巴长，头和腿差不多一样长，脖子最短。 | 1 | equality | S1T140 |
| 腿短，尾巴长，脖子和头差不多长。 | 1 | equality | S1T128 |
| 腿短，脖子、头和尾巴接近。 | 1 | equality | S1T279 |
| 腿短，脖子和头差不多一样长。 | 1 | equality | S1T314 |
| 腿短，脖子和头差不多一样长，尾巴略短。 | 1 | equality | S1T268 |
| 腿短，脖子和头差不多一样长，脖子比尾巴长。 | 1 | equality | S1T89 |
| 腿短，脖子和头很长，脖子和尾巴差不多一样短。 | 1 | equality | S1T165 |
| 腿短，脖子和头接近，长度较长，尾巴更长一些。 | 1 | equality | S2T17 |
| 腿短，脖子和尾巴一样长，比头短。 | 1 | equality | S1T155 |
| 腿短，脖子和尾巴一样长，脖子比头长。 | 1 | equality | S1T243 |
| 腿短，脖子和尾巴差不多一样长，头很长。 | 1 | equality | S1T90 |
| 腿短，脖子和尾巴差不多一样长，脖子比头长。 | 1 | equality | S1T176 |
| 腿短，脖子比头长，和尾巴差不多一样长。 | 1 | equality | S1T209 |
| 腿短，脖子短，头和尾巴差不多一样长，四个部位都长。 | 1 | equality | S1T207 |
| 腿短，脖子较短，尾巴和头差不多长。 | 1 | equality | S1T198 |
| 腿较短，其他三者较长，且长度接近。 | 1 | equality | S2T36 |
| 腿较短，头、尾巴、脖子差不多一样长。 | 1 | equality | S2T27 |
| 腿较短，头、脖子、尾巴长度接近。 | 1 | equality | S2T16 |
| 腿较短，头较短，尾巴和脖子长度接近，其中尾巴要更长一些。 | 1 | equality | S2T34 |
| 腿较短，尾巴和头长度接近，脖子更长。 | 1 | equality | S2T39 |
| 腿较短，尾巴较短，头和脖子长度接近，其中头的长度更长一些。 | 1 | equality | S2T33 |
| 腿较短，尾巴长，头和脖子长度接近、略短。 | 1 | equality | S2T47 |
| 腿较短，脖子比头子，脖子比头长，尾巴和脖子差不多一样长。 | 1 | equality | S1T220 |
| 腿较短，脖子比头长，脖子和尾巴差不多一样长。 | 1 | equality | S1T240 |
| 腿较短，脖子短，头和尾巴差不多一样长。 | 1 | equality | S1T172 |
| 腿较短，脖子较短，头和尾巴长度接近，较长。 | 1 | equality | S2T37 |
| 腿较短，脖子较长，头和尾巴差不多一样长，比脖子短。 | 1 | equality | S1T174 |
| 腿较躯干来说适中，脖子适中，尾巴短，头比脖子短。 | 1 | body_ref | S1T55 |
| 腿较长，头、脖子、尾巴较为接近。 | 1 | equality | S2T12 |
| 腿较长，头和尾巴差不多一样长，脖子短。 | 1 | equality | S1T201 |
| 腿较长，头和脖子差不多一样长，脖子和尾巴较长。 | 1 | equality | S1T274 |
| 腿较长，尾巴较短，头和脖子长度接近，较长。 | 1 | equality | S2T18 |
| 腿较长，尾巴较长，头和脖子长度接近，脖子比头长一些、都较短。 | 1 | equality | S2T31 |
| 腿较长，脖子较短，头和尾巴长度接近、都较长。 | 1 | equality | S2T59 |
| 腿长，尾巴、头、脖子差不多一样长。 | 1 | equality | S1T312 |
| 腿长，尾巴很长，脖子和头差不多一样长。 | 1 | equality | S1T224 |
| 腿长，尾巴比脖和脖子差不多一样长。 | 1 | equality | S1T149 |
| 腿长，尾巴长，头和脖子差不多长。 | 1 | equality | S1T98 |
| 腿长，脖子、尾巴、头差不多一样长。 | 1 | equality | S1T241 |
| 腿长，脖子和头差不多一样长。 | 1 | equality | S1T317 |
| 腿长，脖子和头接近。 | 1 | equality | S1T278 |
| 腿长，脖子和尾巴一样长，比头长。 | 1 | equality | S1T72 |
| 腿长，脖子和尾巴差不多长，头最短。 | 1 | equality | S1T86 |
| 腿长，脖子和尾巴差不多长，比头长一些。 | 1 | equality | S2T14 |
| 腿长，脖子和尾巴还有头差不多一样长。 | 1 | equality | S1T91 |
| 较躯干来说，腿短，尾巴长。 | 1 | body_ref | S1T65 |
| 较躯干来说，腿短，尾巴长，头比脖子稍长。 | 1 | body_ref | S1T67 |
| 较躯干来说，腿较短，其他部位均较长，脖子比头长。 | 1 | body_ref | S1T57 |
| 较躯干来说，腿较长，尾巴短，脖子和头都较长。 | 1 | body_ref | S1T63 |
| 较躯干来说，腿较长，脖子比头长，是头比脖子长，尾巴短。 | 1 | body_ref | S1T64 |
| 较躯干来说，腿适中，其他部位较长一些。 | 1 | body_ref | S1T58 |
| 较躯干而言，腿的适中，脖子较短，头较长，尾巴短。 | 1 | body_ref | S1T60 |
| 较躯干而言，腿短，头比脖子稍长一些，四个部位都短，尾巴最长。 | 1 | body_ref | S1T62 |
| 较躯干而言，腿适中，脖子比头长。 | 1 | body_ref | S1T61 |
| 较躯干而言，腿适中，脖子较短，其他部位适中。 | 1 | body_ref | S1T59 |
| 较躯干而言，腿长，尾巴短。 | 1 | body_ref | S1T66 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿较长，头较长。 | 3 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S2T140, S2T158, S2T164 |
| 四个部位都差不多一样长、都较长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T12 |
| 尾巴和头较其他部位都很短。 | 1 | 0.000 | complement:脖子 < 0.50; complement:腿 < 0.50 | S1T32 |
| 脖子和头明显比其他部位短。 | 1 | 0.000 | complement:腿 < 0.50; complement:尾巴 < 0.50 | S1T28 |
| 腿短，尾巴、脖子、头差不多一样长。 | 1 | 0.000 | absolute_short:腿 < 0.50; equality_range:尾巴+脖子+头 = | S1T163 |
| 腿较短，头、脖子、尾巴长度接近。 | 1 | 0.000 | absolute_short:腿 < 0.50; equality_range:头+脖子+尾巴 = | S2T16 |
| 腿较长，尾巴较长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T195 |
| 腿短，头、尾巴、脖子越来越短。 | 1 | 0.250 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T99 |
| 头比脖子短，其他部位差不多一样长、长度适中。 | 1 | 0.333 | complement:腿 < 0.50; complement:尾巴 < 0.50 | S1T41 |
| 腿短，头长，尾巴长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T281 |

### S307

- trial 数: 704; 非空文本: 702; fidelity 可评分率: 0.996; 平均 fidelity: 0.928; 完全忠实率: 0.795; 低 fidelity 率: 0.011.
- 旧版 region 覆盖率: 0.996; 旧版 region 有未处理片段率: 0.013.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 700 | 0.994 |
| comparison | 416 | 0.591 |
| body_ref | 173 | 0.246 |
| equality | 42 | 0.060 |
| empty | 2 | 0.003 |
| other | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子比躯干短，尾巴长。 | 35 |
| 脖子比躯干长，头长。 | 28 |
| 脖子比躯干长，头短。 | 22 |
| 脖子比躯干短，尾巴短。 | 19 |
| 脖子比躯干短，尾巴比较长。 | 19 |
| 脖子比躯干长，头比较短。 | 16 |
| 脖子比躯干长，头比较长。 | 15 |
| 脖子比躯干短，尾巴比较短。 | 14 |
| 脖子比头长，尾巴比腿长。 | 13 |
| 尾巴短于头、短于脖子。 | 10 |
| 头比脖子长，尾巴较长。 | 9 |
| 头比脖子长，尾巴比脖子长。 | 9 |
| 头比脖子长很多。 | 9 |
| 脖子比头长很多。 | 7 |
| 头比脖子长，尾巴较短。 | 7 |
| 头短于脖子、短于尾巴。 | 6 |
| 脖子短于头、短于尾巴。 | 6 |
| 头比脖子长。 | 6 |
| 脖子比头长，尾巴比头长。 | 6 |
| 尾巴短于脖子、短于头。 | 6 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子比躯干短，尾巴长。 | 35 | body_ref | S2T217, S2T220, S2T226, S2T227, S2T230, S2T232, S2T233, S2T234 |
| 脖子比躯干长，头长。 | 28 | body_ref | S2T221, S2T225, S2T229, S2T239, S2T243, S2T245, S2T259, S2T262 |
| 脖子比躯干长，头短。 | 22 | body_ref | S2T231, S2T238, S2T252, S2T253, S2T254, S2T256, S2T260, S2T264 |
| 脖子比躯干短，尾巴比较长。 | 19 | body_ref | S3T3, S3T4, S3T5, S3T9, S3T11, S3T12, S3T15, S3T19 |
| 脖子比躯干短，尾巴短。 | 19 | body_ref | S2T216, S2T218, S2T219, S2T222, S2T223, S2T224, S2T228, S2T235 |
| 脖子比躯干长，头比较短。 | 16 | body_ref | S3T8, S3T13, S3T16, S3T17, S3T18, S3T24, S3T33, S3T36 |
| 脖子比躯干长，头比较长。 | 15 | body_ref | S3T2, S3T21, S3T23, S3T25, S3T27, S3T30, S3T32, S3T34 |
| 脖子比躯干短，尾巴比较短。 | 14 | body_ref | S3T1, S3T6, S3T7, S3T10, S3T14, S3T20, S3T28, S3T29 |
| 脖子和头差不多长。 | 4 | equality | S1T34, S1T35, S1T38, S1T41 |
| 头和脖子差不多长。 | 3 | equality | S1T51, S1T54, S1T55 |
| 尾巴等于头、短于脖子。 | 3 | equality | S1T296, S1T306, S1T313 |
| 脖子短于头、等于尾巴。 | 3 | equality | S1T288, S1T291, S1T316 |
| 头和脖子比躯干短，尾巴短。 | 2 | body_ref | S2T214, S2T215 |
| 尾巴等于脖子、短于头。 | 2 | equality | S1T283, S1T298 |
| 脖子和头差不多长，尾巴很短。 | 2 | equality | S1T185, S1T186 |
| 脖子比头长，尾巴跟脖子差不多长。 | 2 | equality | S1T239, S1T242 |
| 头、脖子、腿差不多长，尾巴较长。 | 1 | equality | S1T61 |
| 头和脖子差不多长、都是中等，腿和尾巴较长。 | 1 | equality | S1T59 |
| 头和脖子差不多长，头、脖子、尾巴、腿都是中等。 | 1 | equality | S1T106 |
| 头和脖子差不多长，头、脖子、尾巴、腿都较长。 | 1 | equality | S1T104 |
| 头和脖子差不多长，头、脖子、尾巴都很长，腿中等。 | 1 | equality | S1T105 |
| 头和脖子差不多长，头、脖子、尾巴都较长，腿较短。 | 1 | equality | S1T84 |
| 头和脖子差不多长，头、脖子、腿、尾巴都较长。 | 1 | equality | S1T83 |
| 头和脖子差不多长，头较短，尾巴较长，腿较长。 | 1 | equality | S1T88 |
| 头和脖子差不多长，它们都是中等，腿较长，尾巴较长。 | 1 | equality | S1T86 |
| 头和脖子差不多长，尾巴、腿都较长。 | 1 | equality | S1T103 |
| 头和脖子差不多长，尾巴很短，腿长中等。 | 1 | equality | S1T69 |
| 头和脖子差不多长，脖子较长，腿较长，尾巴较短。 | 1 | equality | S1T109 |
| 头和脖子差不多长，腿中等，尾巴中等。 | 1 | equality | S1T155 |
| 头和脖子差不多长，腿中等，尾巴很短。 | 1 | equality | S1T100 |
| 头和脖子差不多长，腿很长，尾巴很长，头和脖子较短。 | 1 | equality | S1T107 |
| 头和脖子差不多长，腿很长，尾巴较长，头和脖子较长。 | 1 | equality | S1T111 |
| 头和脖子差不多长，腿长中等，尾巴较长。 | 1 | equality | S1T70 |
| 头和脖子差不多长，都是较长，尾巴中等，腿较长。 | 1 | equality | S1T66 |
| 头和脖子比躯干长。 | 1 | body_ref | S2T213 |
| 头比脖子长，尾巴跟脖子差不多长。 | 1 | equality | S1T243 |
| 头等于脖子、等于尾巴。 | 1 | equality | S1T295 |
| 尾巴较长，脖子比躯干短。 | 1 | body_ref | S2T212 |
| 脖子和头差不多长，尾巴较长，腿中等。 | 1 | equality | S1T73 |
| 脖子和头差不多长，腿很长，尾巴较短。 | 1 | equality | S1T80 |
| 脖子是头的两倍。 | 1 | other | S1T141 |
| 脖子比头长很多，脖子、腿、尾巴一样长。 | 1 | equality | S1T62 |
| 脖子比躯干长，尾巴长。 | 1 | body_ref | S2T263 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子和头差不多长。 | 2 | 0.000 | equality_range:脖子+头 = | S1T34, S1T35 |
| 脖子短于头、等于尾巴。 | 2 | 0.400 | chained_comparison:脖子 = 尾巴; absolute_short:头 < 0.50; comparison:脖子+头 = 尾巴 | S1T291, S1T316 |
| 头和脖子差不多长。 | 1 | 0.000 | equality_range:头+脖子 = | S1T51 |
| 头比脖子长一点。 | 1 | 0.000 | comparison:头 > 脖子 | S1T142 |
| 尾巴短于脖子、短于头。 | 1 | 0.400 | comparison:尾巴 < 脖子; absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T319 |
| 脖子中等，头较长，尾巴较长，腿较长。 | 1 | 0.400 | absolute:脖子 middle_lower; absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T190 |

### S308

- trial 数: 896; 非空文本: 889; fidelity 可评分率: 0.978; 平均 fidelity: 0.904; 完全忠实率: 0.767; 低 fidelity 率: 0.028.
- 旧版 region 覆盖率: 0.978; 旧版 region 有未处理片段率: 0.030.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 889 | 0.992 |
| comparison | 560 | 0.625 |
| superlative | 292 | 0.326 |
| body_ref | 160 | 0.179 |
| equality | 156 | 0.174 |
| negation | 61 | 0.068 |
| empty | 7 | 0.008 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 脖子长于一半，尾巴短于一半。 | 44 |
| 脖子短于一半，头长于一半。 | 41 |
| 脖子最长。 | 34 |
| 脖子比头长。 | 31 |
| 脖子和尾巴均长于一半。 | 30 |
| 脖子短，头长。 | 26 |
| 脖子长，尾巴长。 | 26 |
| 脖子比头长，脖子比尾巴长。 | 26 |
| 脖子长，尾巴短。 | 24 |
| 脖子和头均短于一半。 | 23 |
| 脖子比尾巴长。 | 22 |
| 脖子短，头短。 | 21 |
| 腿最长。 | 19 |
| 脖子没有尾巴长。 | 15 |
| 头比脖子长。 | 14 |
| 脖子不是最长。 | 12 |
| 尾巴最长。 | 12 |
| 头最长。 | 11 |
| 头比脖子长，头比腿长。 | 11 |
| 头比脖子长，腿比头长。 | 11 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 脖子长于一半，尾巴短于一半。 | 44 | body_ref | S3T16, S3T17, S3T24, S3T25, S3T33, S3T35, S3T42, S3T44 |
| 脖子短于一半，头长于一半。 | 41 | body_ref | S3T10, S3T12, S3T18, S3T26, S3T29, S3T30, S3T32, S3T34 |
| 脖子和尾巴均长于一半。 | 30 | body_ref | S3T22, S3T27, S3T28, S3T38, S3T40, S3T46, S3T62, S3T69 |
| 脖子和头均短于一半。 | 23 | body_ref | S3T23, S3T31, S3T39, S3T45, S3T47, S3T49, S3T58, S3T74 |
| 脖子没有尾巴长。 | 15 | negation | S2T23, S2T31, S2T185, S2T190, S2T191, S2T193, S2T195, S2T196 |
| 脖子不是最长。 | 12 | negation | S2T235, S2T242, S2T243, S2T244, S2T257, S2T258, S2T273, S2T280 |
| 脖子比头长，脖子没有尾巴长。 | 10 | negation | S2T13, S2T33, S2T38, S2T55, S2T60, S2T72, S2T78, S2T85 |
| 头比脖子长，脖子没有尾巴长。 | 9 | negation | S2T27, S2T32, S2T34, S2T37, S2T43, S2T52, S2T63, S2T118 |
| 脖子长于一半，尾巴长于一半。 | 8 | body_ref | S3T13, S3T14, S3T20, S3T52, S3T55, S3T60, S3T70, S3T85 |
| 脖子比头长，腿和尾巴长度相似。 | 5 | equality | S1T168, S1T169, S1T188, S1T189, S1T221 |
| 头比脖子长，腿和尾巴长度相似。 | 4 | equality | S1T158, S1T159, S1T178, S1T190 |
| 头、脖子、腿、尾巴长度相似。 | 3 | equality | S1T179, S1T230, S1T232 |
| 头、脖子和尾巴长度相似，腿最短。 | 3 | equality | S1T193, S1T196, S1T218 |
| 头比脖子长，腿与尾巴长度相似。 | 3 | equality | S1T18, S1T141, S1T173 |
| 腿与尾巴长度相似，头比脖子长。 | 3 | equality | S1T118, S1T119, S1T151 |
| 腿最长，头与尾巴长度相似。 | 3 | equality | S1T106, S1T149, S1T174 |
| 头与尾巴长度相似，腿最短。 | 2 | equality | S1T58, S1T150 |
| 头比脖子长，头和尾巴长度相似。 | 2 | equality | S1T167, S1T219 |
| 尾巴最长，头、脖子、腿长度相似。 | 2 | equality | S1T71, S1T72 |
| 脖子比头长，头与腿等长。 | 2 | equality | S1T46, S1T47 |
| 脖子比头长，尾巴与腿等长。 | 2 | equality | S1T33, S1T44 |
| 脖子比头长，腿、尾巴长度相似。 | 2 | equality | S1T192, S1T195 |
| 脖子没有尾巴长，尾巴比腿长。 | 2 | negation | S2T206, S2T207 |
| 脖子没有尾巴长，腿比头长。 | 2 | negation | S2T209, S2T210 |
| 脖子短于一半，头短于一半。 | 2 | body_ref | S3T19, S3T57 |
| 脖子短于一半，腿长。 | 2 | body_ref | S3T3, S3T6 |
| 脖子短于一半，腿长于一半。 | 2 | body_ref | S3T11, S3T15 |
| 腿和尾巴长度相似，头比脖子长。 | 2 | equality | S1T144, S1T191 |
| 腿最短，头、脖子、尾巴长度相似。 | 2 | equality | S1T95, S1T164 |
| 腿最长，头、脖子、尾巴长度相似。 | 2 | equality | S1T83, S1T186 |
| 腿最长，脖子比头长，头与尾巴长度相似。 | 2 | equality | S1T104, S1T153 |
| 哪个最长，头和脖子长度相似，腿最短。 | 1 | equality | S1T160 |
| 头、尾巴、脖子、腿长度相近。 | 1 | equality | S1T246 |
| 头、脖子、尾巴长度相似，腿最短。 | 1 | equality | S1T113 |
| 头、脖子、腿、尾巴长度均相似。 | 1 | equality | S1T199 |
| 头、脖子、腿长度相似，尾巴最短。 | 1 | equality | S1T139 |
| 头、脖子和腿长度相似，尾巴最短。 | 1 | equality | S1T161 |
| 头与尾巴长度相似。 | 1 | equality | S1T69 |
| 头与尾巴长度相似，头比脖子长，腿最短。 | 1 | equality | S1T56 |
| 头与尾巴长度相似，比脖子长。 | 1 | equality | S1T92 |
| 头与尾巴长度相似，脖子最短，腿最长。 | 1 | equality | S1T65 |
| 头与尾巴长度相似，脖子比头长。 | 1 | equality | S1T240 |
| 头与尾巴长度相似，腿比脖子长。 | 1 | equality | S1T67 |
| 头与脖子长度相似，尾巴最短。 | 1 | equality | S1T171 |
| 头与脖子长度相似，尾巴最长。 | 1 | equality | S1T121 |
| 头与脖子长度相似，腿最长。 | 1 | equality | S1T216 |
| 头与腿长度相似，尾巴最短。 | 1 | equality | S1T70 |
| 头和尾巴长度相似，脖子最短。 | 1 | equality | S1T146 |
| 头和脖子长于一半。 | 1 | body_ref | S3T8 |
| 头和脖子长于一半，尾巴短于一半。 | 1 | body_ref | S3T21 |
| 头和脖子长度相似，尾巴最长。 | 1 | equality | S1T245 |
| 头和脖子长度相似，腿和尾巴长度相似，腿和尾巴长度更短。 | 1 | equality | S1T166 |
| 头和腿长度相似，头比脖子长，尾巴最短。 | 1 | equality | S1T140 |
| 头最短，脖子、腿、尾巴长度相似。 | 1 | equality | S1T77 |
| 头最短，脖子最长，尾巴与躯干等长。 | 1 | equality, body_ref | S1T4 |
| 头最短，腿与尾巴等长，且更长。 | 1 | equality | S1T112 |
| 头最长，头比脖子长，脖子与尾巴长度相似。 | 1 | equality | S1T123 |
| 头最长，尾巴、腿、脖子长度相似。 | 1 | equality | S1T84 |
| 头最长，尾巴最短，脖子和腿长度相似。 | 1 | equality | S1T91 |
| 头最长，脖子、腿、尾巴长度相似。 | 1 | equality | S1T242 |
| 头最长，脖子与尾巴长度相似。 | 1 | equality | S1T86 |
| 头最长，脖子与尾巴长度相似，腿最短。 | 1 | equality | S1T64 |
| 头最长，脖子最短，尾巴与躯干等长。 | 1 | equality, body_ref | S1T5 |
| 头最长，脖子最短，腿与尾巴长度相似。 | 1 | equality | S1T122 |
| 头比脖子长，头与尾巴长度相似，腿最短。 | 1 | equality | S1T185 |
| 头比脖子长，头与腿等长，尾巴最短。 | 1 | equality | S1T137 |
| 头比脖子长，头与腿长度相似。 | 1 | equality | S1T225 |
| 头比脖子长，头与腿长度相似，尾巴最短。 | 1 | equality | S1T133 |
| 头比脖子长，头与腿长度相似，腿略长，头与尾巴长度相似。 | 1 | equality | S1T148 |
| 头比脖子长，头和腿长度相似，尾巴最短。 | 1 | equality | S1T138 |
| 头比脖子长，头和腿长度相似，尾巴最长。 | 1 | equality | S1T177 |
| 头比脖子长，尾巴与腿等长。 | 1 | equality | S1T34 |
| 头比脖子长，尾巴与腿长度相似。 | 1 | equality | S1T54 |
| 头比脖子长，尾巴和腿等长。 | 1 | equality | S1T32 |
| 头比脖子长，尾巴和腿长度相似。 | 1 | equality | S1T50 |
| 头比脖子长，尾巴最短，头与腿长的相似。 | 1 | equality | S1T156 |
| 头比脖子长，脖子与尾巴长度相似。 | 1 | equality | S1T231 |
| 头比脖子长，脖子与尾巴长度相似，头与腿长度相似。 | 1 | equality | S1T213 |
| 头比脖子长，脖子与尾巴长度相似，腿最短。 | 1 | equality | S1T170 |
| 头比脖子长，脖子与腿等长。 | 1 | equality | S1T45 |
| 头比脖子长，脖子最短，腿与尾巴长度相似。 | 1 | equality | S1T198 |
| 头比脖子长，腿、尾巴和头长度相似。 | 1 | equality | S1T127 |
| 头比脖子长，腿与尾巴等长。 | 1 | equality | S1T116 |
| 头比脖子长，腿与尾巴长度相似，脖子最短。 | 1 | equality | S1T128 |
| 头比脖子长，腿最短，头与尾巴长度相似。 | 1 | equality | S1T129 |
| 头没有脖子长，脖子比尾巴长。 | 1 | negation | S2T133 |
| 头没有脖子长，脖子没有尾巴长。 | 1 | negation | S2T116 |
| 尾巴和腿长度相似，脖子比头长。 | 1 | equality | S1T162 |
| 尾巴最短，头、脖子、腿长度相似。 | 1 | equality | S1T157 |
| 尾巴最长，其，头、脖子、腿长度相似。 | 1 | equality | S1T89 |
| 尾巴最长，头、腿、脖子长度相似。 | 1 | equality | S1T79 |
| 尾巴最长，头与腿长度相似，脖子最短。 | 1 | equality | S1T105 |
| 尾巴最长，头最短，腿与脖子长度相似。 | 1 | equality | S1T126 |
| 尾巴最长，头比脖子长，头与腿长度相似。 | 1 | equality | S1T130 |
| 尾巴最长，尾巴与头长度相似，头比脖子长。 | 1 | equality | S1T103 |
| 尾巴最长，尾巴与腿长度相似。 | 1 | equality | S1T228 |
| 尾巴最长，尾巴与腿长度相似，头和脖子短，头与脖子长度相似。 | 1 | equality | S1T107 |
| 尾巴最长，尾巴和头等长，脖子最短。 | 1 | equality | S1T82 |
| 脖子不是最长，尾巴长。 | 1 | negation | S2T241 |
| 脖子不是最长，腿长。 | 1 | negation | S2T239 |
| 脖子与头长度相似，尾巴长。 | 1 | equality | S1T31 |
| 脖子与尾巴长度相似。 | 1 | equality | S1T68 |
| 脖子与尾巴长度相似，且最长。 | 1 | equality | S1T63 |
| 脖子与尾巴长度相似，脖子比头长，尾巴最短。 | 1 | equality | S1T147 |
| 脖子与尾巴长度相似，腿略长。 | 1 | equality | S1T114 |
| 脖子与尾巴长度相似，腿短。 | 1 | equality | S1T20 |
| 脖子与腿长度相似，且长度长于头。 | 1 | equality | S1T55 |
| 脖子与腿长度相似，尾巴最长。 | 1 | equality | S1T57 |
| 脖子和头长于一半。 | 1 | body_ref | S3T9 |
| 脖子和头长度相似，尾巴最短。 | 1 | equality | S1T244 |
| 脖子和尾巴长度相似，腿最短，脖子比头长。 | 1 | equality | S1T85 |
| 脖子最短，头、尾巴、腿长度相似。 | 1 | equality | S1T75 |
| 脖子最短，头、腿、尾巴长度相似。 | 1 | equality | S1T78 |
| 脖子最长，头、腿、尾巴长度相似。 | 1 | equality | S1T73 |
| 脖子最长，头、腿长度相似，尾巴最短。 | 1 | equality | S1T125 |
| 脖子最长，头与尾巴长度相似。 | 1 | equality | S1T90 |
| 脖子最长，腿最短，脖子与尾巴长度相似。 | 1 | equality | S1T154 |
| 脖子比头长，头与尾巴等长。 | 1 | equality | S1T93 |
| 脖子比头长，头最短，头、腿、尾巴等长。 | 1 | equality | S1T136 |
| 脖子比头长，头最短，脖子和尾巴长度相似。 | 1 | equality | S1T97 |
| 脖子比头长，尾巴和腿等长。 | 1 | equality | S1T40 |
| 脖子比头长，尾巴和腿长度相似。 | 1 | equality | S1T49 |
| 脖子比头长，尾巴最短，头跟腿长度相似。 | 1 | equality | S1T163 |
| 脖子比头长，脖子、腿、尾巴长度相似。 | 1 | equality | S1T155 |
| 脖子比头长，脖子与尾巴长度相似。 | 1 | equality | S1T19 |
| 脖子比头长，脖子与腿长度相似。 | 1 | equality | S1T237 |
| 脖子比头长，脖子没有尾巴长，腿比脖子长。 | 1 | negation | S2T49 |
| 脖子比头长，脖子没有腿长。 | 1 | negation | S2T73 |
| 脖子比头长，腿与尾巴长度相似。 | 1 | equality | S1T204 |
| 脖子比头长，腿与尾巴长度相似，且比且比脖子更长。 | 1 | equality | S1T210 |
| 脖子比头长，腿和尾巴长度相似，且比脖子更长。 | 1 | equality | S1T194 |
| 脖子没有头长，脖子比尾巴长。 | 1 | negation | S2T56 |
| 脖子没有尾巴长，头最长。 | 1 | negation | S2T30 |
| 脖子没有尾巴长，头比脖子长。 | 1 | negation | S2T29 |
| 脖子没有尾巴长，头比腿长。 | 1 | negation | S2T200 |
| 脖子没有尾巴长，脖子比头长。 | 1 | negation | S2T28 |
| 脖子短于一半头长。 | 1 | body_ref | S3T5 |
| 脖子长于一半。 | 1 | body_ref | S3T7 |
| 脖子长于一半，腿短。 | 1 | body_ref | S3T4 |
| 腿和头、脖子长度相似，尾巴最短。 | 1 | equality | S1T117 |
| 腿最短，头和脖子长度相似。 | 1 | equality | S1T241 |
| 腿最短，尾巴、脖子、头长度相似。 | 1 | equality | S1T183 |
| 腿最短，脖子与头相似。 | 1 | equality | S1T9 |
| 腿最短，脖子与头等长。 | 1 | equality | S1T35 |
| 腿最短，脖子与头长度相似。 | 1 | equality | S1T8 |
| 腿最长，头与脖子长度相似，尾巴最短。 | 1 | equality | S1T152 |
| 腿最长，头比脖子略长，尾巴与脖子长度相似。 | 1 | equality | S1T124 |
| 腿最长，头比脖子长，头与尾巴长度相似。 | 1 | equality | S1T102 |
| 腿最长，尾巴和脖子长度相似，脖子比头长。 | 1 | equality | S1T120 |
| 腿最长，尾巴最短，脖子与头长度相似。 | 1 | equality | S1T6 |
| 腿最长，脖子、头、尾巴长度相似。 | 1 | equality | S1T74 |
| 腿最长，脖子与头长度相似，尾巴最短。 | 1 | equality | S1T96 |
| 腿最长，脖子最短，头与尾巴长度相似。 | 1 | equality | S1T94 |
| 腿最长，脖子比头长，脖子与尾巴长度相似。 | 1 | equality | S1T134 |
| 腿最长，腿与尾巴长度相似。 | 1 | equality | S1T88 |
| 腿最长，腿与尾巴长度相似，头比脖子长。 | 1 | equality | S1T115 |
| 腿最长，腿与尾巴长度相似，脖子最短。 | 1 | equality | S1T99 |
| 腿最长，臀与尾巴、脖子长度相似，头最短。 | 1 | equality | S1T98 |
| 腿短，尾巴短，脖子与头相似。 | 1 | equality | S1T12 |
| 腿长，尾巴长，脖子与头相似。 | 1 | equality | S1T11 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子没有尾巴长。 | 4 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T193, S2T196, S2T198, S2T214 |
| 头、脖子、腿、尾巴长度相似。 | 3 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T179, S1T230, S1T232 |
| 脖子最长。 | 2 | 0.000 | superlative:脖子 > 头; superlative:脖子 > 腿; superlative:脖子 > 尾巴 | S2T251, S2T284 |
| 脖子比尾巴长。 | 2 | 0.000 | comparison:脖子 > 尾巴 | S2T187, S2T194 |
| 头、尾巴、脖子、腿长度相近。 | 1 | 0.000 | equality_range:头+尾巴+脖子+腿 = | S1T246 |
| 头、脖子、腿、尾巴长度均相似。 | 1 | 0.000 | equality_range:头+脖子+腿+尾巴 = | S1T199 |
| 头与尾巴长度相似。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T69 |
| 头比脖子长。 | 1 | 0.000 | comparison:头 > 脖子 | S1T293 |
| 尾巴比腿，腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T296 |
| 脖子与尾巴长度相似。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T68 |
| 脖子与尾巴长度相似，且最长。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T63 |
| 脖子比头长。 | 1 | 0.000 | comparison:脖子 > 头 | S1T209 |
| 脖子长于一半。 | 1 | 0.000 | body_ref:脖子 > 0.50 | S3T7 |
| 腿比头长。 | 1 | 0.000 | comparison:腿 > 头 | S2T236 |
| 脖子与尾巴长度相似，脖子比头长，尾巴最短。 | 1 | 0.200 | equality_range:脖子+尾巴 =; superlative:尾巴 < 脖子; superlative:尾巴 < 头; superlative:尾巴 < 腿 | S1T147 |
| 脖子没有头长，脖子比尾巴长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; comparison:脖子 > 尾巴 | S2T56 |
| 脖子没有尾巴长，头比脖子长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S2T29 |
| 腿最短。 | 1 | 0.333 | superlative:腿 < 脖子; superlative:腿 < 尾巴 | S2T277 |

### S309

- trial 数: 768; 非空文本: 768; fidelity 可评分率: 0.990; 平均 fidelity: 0.927; 完全忠实率: 0.833; 低 fidelity 率: 0.009.
- 旧版 region 覆盖率: 0.990; 旧版 region 有未处理片段率: 0.013.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 755 | 0.983 |
| comparison | 84 | 0.109 |
| superlative | 38 | 0.049 |
| equality | 18 | 0.023 |
| meta | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头短，尾巴长。 | 104 |
| 头长，脖子长。 | 94 |
| 头长，脖子短。 | 85 |
| 头短，尾巴短。 | 78 |
| 腿长，头短。 | 14 |
| 腿长，尾巴长。 | 13 |
| 腿最长。 | 11 |
| 头长，腿短。 | 11 |
| 腿短，尾巴长。 | 10 |
| 头长，腿长。 | 9 |
| 腿长，头长。 | 8 |
| 头比脖子长。 | 8 |
| 腿长，尾巴短。 | 7 |
| 尾巴最长。 | 6 |
| 腿长，尾巴长，脖子短。 | 6 |
| 腿长，头长，尾巴短。 | 5 |
| 腿短，脖子长。 | 5 |
| 头长，尾巴短。 | 5 |
| 头短，腿长。 | 5 |
| 腿非常长。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 整体都较为均匀。 | 2 | equality | S1T26, S1T27 |
| 比较均衡。 | 2 | equality | S1T258, S1T264 |
| 腿和头差不多。 | 2 | equality | S1T234, S1T237 |
| 四个部位较为均匀。 | 1 | equality | S1T202 |
| 四个部位都比较均匀。 | 1 | equality | S1T46 |
| 基本都很均衡。 | 1 | equality | S1T317 |
| 头和脖子差不多。 | 1 | equality | S2T63 |
| 头和腿差不多。 | 1 | equality | S1T238 |
| 尾巴和头一样长。 | 1 | equality | S2T18 |
| 尾巴非常短，其他比较均匀。 | 1 | equality | S1T31 |
| 脖子和头差不多长。 | 1 | equality | S2T28 |
| 脖子缀得最长，尾巴最短，腿和头差不多。 | 1 | equality | S1T231 |
| 腿和尾巴差不多长。 | 1 | equality | S2T22 |
| 腿和脖子一样长，尾巴短。 | 1 | equality | S1T283 |
| 选错了。 | 1 | meta | S3T9 |
| 非常均衡。 | 1 | equality | S1T295 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位较为均匀。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T202 |
| 四个部位都比较均匀。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T46 |
| 头和腿差不多。 | 1 | 0.000 | equality_range:头+腿 = | S1T238 |
| 腿和头差不多。 | 1 | 0.000 | equality_range:腿+头 = | S1T234 |
| 腿长，头长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S1T185 |
| 头长，尾巴长，脖子长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T97 |
| 尾巴比较短，头适中。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute:头 middle_upper | S2T42 |

### S310

- trial 数: 1088; 非空文本: 1085; fidelity 可评分率: 0.996; 平均 fidelity: 0.913; 完全忠实率: 0.756; 低 fidelity 率: 0.023.
- 旧版 region 覆盖率: 0.996; 旧版 region 有未处理片段率: 0.034.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1081 | 0.994 |
| superlative | 544 | 0.500 |
| equality | 133 | 0.122 |
| comparison | 125 | 0.115 |
| body_ref | 100 | 0.092 |
| ranking | 65 | 0.060 |
| count_abstract | 4 | 0.004 |
| empty | 3 | 0.003 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，尾巴短。 | 75 |
| 头长，尾巴长。 | 58 |
| 头短，脖子长。 | 34 |
| 头最长。 | 27 |
| 脖子最长。 | 21 |
| 头最长，尾巴最短。 | 18 |
| 头短，尾巴长。 | 18 |
| 尾巴最长。 | 17 |
| 头短，脖子短。 | 15 |
| 腿最长。 | 11 |
| 尾巴最短。 | 10 |
| 脖子最短。 | 8 |
| 头最长，腿最短。 | 7 |
| 头最长，脖子比腿长。 | 6 |
| 脖子最长，腿最短。 | 6 |
| 腿最长，尾巴最短。 | 6 |
| 头比躯干长，尾巴长。 | 6 |
| 头比躯干长，尾巴短。 | 6 |
| 脖子和腿较长，头和尾巴较短。 | 6 |
| 脖子最长，头最短。 | 5 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头比躯干长，尾巴短。 | 6 | body_ref | S3T90, S4T7, S4T8, S4T9, S4T10, S4T14 |
| 头比躯干长，尾巴长。 | 6 | body_ref | S3T91, S3T122, S3T125, S4T11, S4T19, S4T20 |
| 头比躯干短，脖子比躯干短。 | 4 | body_ref | S4T75, S4T84, S4T85, S4T87 |
| 脖子最长，头第二，尾巴第三，腿最短。 | 4 | ranking | S1T33, S1T48, S2T11, S2T51 |
| 腿最长，尾巴第二，头第三，脖子最短。 | 4 | ranking | S2T5, S2T13, S2T65, S2T75 |
| 四个部位差不多长。 | 3 | equality | S1T275, S1T299, S2T30 |
| 头最长，脖子第二，腿第三，尾巴最短。 | 3 | ranking | S1T141, S1T260, S2T114 |
| 头比躯干短，尾巴短。 | 3 | body_ref | S3T123, S3T126, S3T151 |
| 头比躯干短，脖子比躯干长。 | 3 | body_ref | S4T80, S4T81, S4T83 |
| 头比躯干长，尾巴比躯干短。 | 3 | body_ref | S4T78, S4T82, S4T86 |
| 四个部位较均衡。 | 2 | equality | S2T61, S2T101 |
| 头最长，其他部位差不多。 | 2 | equality | S1T268, S2T49 |
| 头最长，尾巴第二，脖子和腿最短。 | 2 | ranking | S1T109, S2T283 |
| 头最长，尾巴第二，腿第三，脖子最短。 | 2 | ranking | S2T91, S2T96 |
| 头最长，脖子、尾巴差不多，腿最短。 | 2 | equality | S2T6, S2T63 |
| 头最长，脖子和腿差不多。 | 2 | equality | S1T94, S1T98 |
| 头最长，脖子第二，腿和尾巴最短。 | 2 | ranking | S2T98, S2T100 |
| 头短，脖子和尾巴差不多长。 | 2 | equality | S3T161, S3T214 |
| 头短，脖子最长，其余各部位相等。 | 2 | equality | S3T253, S3T257 |
| 头短，躯干长。 | 2 | body_ref | S3T300, S3T319 |
| 尾巴最长，其余部分差不多。 | 2 | equality | S2T285, S2T290 |
| 尾巴最长，头第二，脖子和腿最短。 | 2 | ranking | S1T122, S1T198 |
| 其他差不多长，腿最短。 | 1 | equality | S1T306 |
| 各部位较均衡。 | 1 | equality | S3T37 |
| 各部位长度差不多。 | 1 | equality | S3T21 |
| 四个部位差不多长，脖子稍微短一些。 | 1 | equality | S2T162 |
| 四个部位相对较均衡，头最长。 | 1 | equality | S2T46 |
| 四个部位较均衡，头较长。 | 1 | equality | S2T21 |
| 四个部位都较短，四个部位长度差不多。 | 1 | equality | S2T40 |
| 头、尾巴和躯干差不多长，脖子、腿很短。 | 1 | equality, body_ref | S3T35 |
| 头、脖子、腿和尾巴长度差不多，是躯干的一半。 | 1 | equality, body_ref | S3T40 |
| 头、脖子、腿最长，躯干次之，尾巴最短。 | 1 | ranking, body_ref | S3T68 |
| 头、脖子、腿的长度差不多，尾巴最短。 | 1 | equality | S2T45 |
| 头、脖子、腿长度差不多、且比躯干短，尾巴最短。 | 1 | equality, body_ref | S3T18 |
| 头、脖子、躯干、腿长度相当，尾巴稍短。 | 1 | equality, body_ref | S3T45 |
| 头、脖子和尾巴长度差不多，腿稍短。 | 1 | equality | S3T102 |
| 头、脖子和腿长度差不多，尾巴最短。 | 1 | equality | S1T226 |
| 头、脖子和腿长度差不多，尾巴要稍短一些，和躯干的长度相当，全身比较均衡。 | 1 | equality, body_ref | S2T37 |
| 头、腿较短，脖子、尾巴最长，比躯干长。 | 1 | body_ref | S3T105 |
| 头、躯干、腿长度差不多，脖子、尾巴长度差不多且较短。 | 1 | equality, body_ref | S3T50 |
| 头和尾巴差不多长，脖子和腿很短。 | 1 | equality | S3T206 |
| 头和尾巴很短，腿最长，比躯干长，脖子稍短。 | 1 | body_ref | S3T31 |
| 头和尾巴最长且相等，脖子最短，腿稍长。 | 1 | equality | S3T29 |
| 头和尾巴最长，脖子、躯干、腿长度差不多。 | 1 | equality, body_ref | S3T72 |
| 头和尾巴最长，腿次之，脖子最短。 | 1 | ranking | S1T12 |
| 头和尾巴相当，脖子和腿相当。 | 1 | equality | S3T30 |
| 头和尾巴长度差不多、比脖子短，腿最短，且各部位都较短。 | 1 | equality | S3T48 |
| 头和尾巴长度差不多，腿最长。 | 1 | equality | S1T73 |
| 头和尾巴长度相当，脖子和躯干长度相当，腿最短。 | 1 | equality, body_ref | S3T61 |
| 头和尾巴，相当躯干，差不多，脖子最长，腿最短。 | 1 | equality, body_ref | S3T69 |
| 头和脖子差不多长，腿最短，尾巴中间。 | 1 | equality | S1T202 |
| 头和脖子差不多长，腿短，尾巴最短。 | 1 | equality | S1T140 |
| 头和脖子差不多，尾巴最短。 | 1 | equality | S2T267 |
| 头和脖子比躯干更长，腿和尾巴最短。 | 1 | body_ref | S3T26 |
| 头和脖子比躯干短。 | 1 | body_ref | S4T76 |
| 头和腿差不多长，脖子第二，尾巴最短。 | 1 | equality, ranking | S1T313 |
| 头和腿最长，尾巴次之，脖子最短。 | 1 | ranking | S2T2 |
| 头和腿比躯干更长，脖子和尾巴比躯干稍短。 | 1 | body_ref | S3T24 |
| 头和腿比躯干短，脖子和尾巴比躯干长。 | 1 | body_ref | S3T15 |
| 头很长，尾巴很短，脖子、腿和头差不多长。 | 1 | equality | S2T119 |
| 头很长，尾巴第二，脖子和腿较短。 | 1 | ranking | S2T128 |
| 头很长，脖子和腿，长度差不多。 | 1 | equality | S1T160 |
| 头最短，其余部分较均衡。 | 1 | equality | S2T62 |
| 头最短，尾巴、脖子、腿差不多长。 | 1 | equality | S1T188 |
| 头最短，脖子最长，其余部位长度相当。 | 1 | equality | S3T62 |
| 头最短，脖子第二短，腿和尾巴很长。 | 1 | ranking | S1T280 |
| 头最长、比躯干长，脖子、腿、尾巴长度相当，且与躯干长度差不多。 | 1 | equality, body_ref | S3T49 |
| 头最长且和脖子差不多长。 | 1 | equality | S1T316 |
| 头最长，其他部位相差不多。 | 1 | equality | S1T93 |
| 头最长，其余三个部位长度相当。 | 1 | equality, count_abstract | S1T46 |
| 头最长，其余各部位相当。 | 1 | equality | S3T67 |
| 头最长，其余各部位长度差不多。 | 1 | equality | S3T46 |
| 头最长，其余各部分差不多。 | 1 | equality | S2T276 |
| 头最长，其余四个部位长度差不多。 | 1 | equality | S3T20 |
| 头最长，尾巴最短，脖子和腿差不多。 | 1 | equality | S2T22 |
| 头最长，尾巴最短，脖子和腿差不多长。 | 1 | equality | S1T273 |
| 头最长，尾巴次之，腿较长，脖子最短。 | 1 | ranking | S2T140 |
| 头最长，尾巴第二。 | 1 | ranking | S2T57 |
| 头最长，尾巴第二，脖子、腿最短。 | 1 | ranking | S1T190 |
| 头最长，尾巴第二，脖子和腿中间。 | 1 | ranking | S1T207 |
| 头最长，尾巴第二，脖子和腿差不多。 | 1 | equality, ranking | S2T94 |
| 头最长，尾巴第二，脖子和腿短。 | 1 | ranking | S1T129 |
| 头最长，尾巴第二，脖子和腿稍短。 | 1 | ranking | S2T72 |
| 头最长，脖子、腿、躯干差不多，尾巴最短。 | 1 | equality, body_ref | S3T86 |
| 头最长，脖子、腿、躯干长度相当，尾巴最短。 | 1 | equality, body_ref | S3T22 |
| 头最长，脖子、腿，差不多长，尾巴很短。 | 1 | equality | S2T82 |
| 头最长，脖子、躯干、尾巴差不多长，腿最短。 | 1 | equality, body_ref | S3T41 |
| 头最长，脖子、躯干、尾巴长度相当，腿稍短。 | 1 | equality, body_ref | S3T56 |
| 头最长，脖子、躯干、腿，长度差不多，尾巴最短。 | 1 | equality, body_ref | S3T44 |
| 头最长，脖子和腿差不多，尾巴最短。 | 1 | equality | S1T110 |
| 头最长，脖子和腿第二，尾巴最短。 | 1 | ranking | S1T82 |
| 头最长，脖子次之，尾巴次较长，腿最短。 | 1 | ranking | S2T268 |
| 头最长，脖子次之，腿最短。 | 1 | ranking | S1T258 |
| 头最长，腿也长，脖子次之，尾巴最短。 | 1 | ranking | S1T187 |
| 头最长，腿第二长，脖子和尾巴最短。 | 1 | ranking | S1T10 |
| 头最长，腿第二，脖子第三，尾巴最短。 | 1 | ranking | S2T1 |
| 头最长，腿，第二脖子和尾巴最短。 | 1 | ranking | S2T97 |
| 头最长，躯干第二，脖子和尾巴稍短，腿最短。 | 1 | ranking, body_ref | S3T63 |
| 头比躯干短，其余各部位长度比头长，且长度相当。 | 1 | equality, body_ref | S4T17 |
| 头比躯干短，尾巴最短，腿很长，脖子长。 | 1 | body_ref | S4T4 |
| 头比躯干短，尾巴最短，腿最长。 | 1 | body_ref | S3T200 |
| 头比躯干短，尾巴比脖子长。 | 1 | body_ref | S4T30 |
| 头比躯干短，尾巴短，躯干最长，且明显长于其他部位。 | 1 | body_ref | S4T16 |
| 头比躯干短，尾巴稍长，尾巴比脖子长。 | 1 | body_ref | S3T121 |
| 头比躯干短，尾巴长。 | 1 | body_ref | S3T124 |
| 头比躯干短，脖子、躯干、尾巴都很长，腿较短。 | 1 | body_ref | S4T15 |
| 头比躯干短，脖子比躯干、尾巴、腿稍短。 | 1 | body_ref | S4T5 |
| 头比躯干短，脖子长。 | 1 | body_ref | S3T127 |
| 头比躯干短，腿比其他部位都长，且最长。 | 1 | body_ref | S4T13 |
| 头比躯干短，躯干较长，其余各部分较短。 | 1 | body_ref | S4T18 |
| 头比躯干长，尾巴比躯干长。 | 1 | body_ref | S4T77 |
| 头比躯干长，尾巴稍短。 | 1 | body_ref | S3T93 |
| 头比躯干长，尾巴稍长。 | 1 | body_ref | S3T96 |
| 头短，其余各部位都较短且差不多长。 | 1 | equality | S3T250 |
| 头短，其余各部位长度均衡。 | 1 | equality | S4T45 |
| 头短，其余都差不多长。 | 1 | equality | S3T320 |
| 头短，尾巴、躯干较长，腿，脖子较短。 | 1 | body_ref | S3T262 |
| 头短，尾巴更长，其余各部分相等。 | 1 | equality | S3T259 |
| 头短，尾巴短，其余各部分差不多长。 | 1 | equality | S4T23 |
| 头短，尾巴短，脖子、躯干长。 | 1 | body_ref | S3T263 |
| 头短，脖子、躯干差不多长。 | 1 | equality, body_ref | S3T316 |
| 头短，脖子、躯干长。 | 1 | body_ref | S3T309 |
| 头短，脖子和尾巴短，腿、躯干长。 | 1 | body_ref | S4T38 |
| 头短，脖子比躯干长。 | 1 | body_ref | S4T74 |
| 头短，脖子躯干长。 | 1 | body_ref | S3T311 |
| 头短，脖子长，尾巴长，躯干长，腿最短。 | 1 | body_ref | S4T36 |
| 头短，腿最短，其余各部位差不多长。 | 1 | equality | S3T315 |
| 头短，躯干最长，其余各部位都较短且差不多长。 | 1 | equality, body_ref | S4T40 |
| 头稍短，脖子、躯干、尾巴、腿长度差不多，比头长。 | 1 | equality, body_ref | S3T42 |
| 头稍长，躯干最长，脖子、尾巴、腿较短。 | 1 | body_ref | S3T92 |
| 头长，尾巴和躯干差不多长。 | 1 | equality, body_ref | S3T148 |
| 尾巴和腿很长，脖子次之，头最短。 | 1 | ranking | S2T280 |
| 尾巴和腿，长度比躯干更长，头和脖子很短。 | 1 | body_ref | S3T75 |
| 尾巴很长，头很短，脖子和腿差不多。 | 1 | equality | S2T124 |
| 尾巴最短，其他差不多长。 | 1 | equality | S1T305 |
| 尾巴最长、比躯干长，头次之、比躯干短，尾巴、脖子、腿都很短。 | 1 | ranking, body_ref | S3T43 |
| 尾巴最长，其余三个部位差不多。 | 1 | equality, count_abstract | S2T104 |
| 尾巴最长，其余各部位都稍短且差不多。 | 1 | equality | S3T115 |
| 尾巴最长，头第二，脖子和腿很短。 | 1 | ranking | S2T3 |
| 尾巴最长，脖子第二，头第三，尾巴和腿最短。 | 1 | ranking | S1T271 |
| 尾巴最长，脖子第二，腿第三，头最短。 | 1 | ranking | S1T83 |
| 尾巴最长，躯干次之，头、脖子、腿较短。 | 1 | ranking, body_ref | S3T52 |
| 尾巴比头长，脖子和腿差不多。 | 1 | equality | S1T108 |
| 尾巴比头长，腿和脖子长度差不多。 | 1 | equality | S1T103 |
| 尾巴较长，其余差不多。 | 1 | equality | S2T319 |
| 尾巴较长，头最短，其余部位长度差不多。 | 1 | equality | S3T64 |
| 尾巴较长，脖子和腿差不多长。 | 1 | equality | S1T283 |
| 尾巴长，腿第二，头第三，脖子，次。 | 1 | ranking | S2T76 |
| 差不多长。 | 1 | equality | S2T117 |
| 脖子、尾巴和躯干差不多长，头、腿很短。 | 1 | equality, body_ref | S3T36 |
| 脖子、腿很长，头次之，尾巴最短。 | 1 | ranking | S2T277 |
| 脖子、躯干、尾巴、腿长度相当，头最短。 | 1 | equality, body_ref | S3T53 |
| 脖子、躯干、腿长度相当，头、尾巴长度较短。 | 1 | equality, body_ref | S3T23 |
| 脖子和尾巴一样长，腿最短。 | 1 | equality | S1T135 |
| 脖子和尾巴最长，腿第二，头最短。 | 1 | ranking | S1T298 |
| 脖子和腿差不多长。 | 1 | equality | S1T311 |
| 脖子和腿差不多，尾巴最长。 | 1 | equality | S2T43 |
| 脖子和腿差不多，尾巴比脖子和腿要长一点，头最短。 | 1 | equality | S2T106 |
| 脖子和腿差不多，腿最短，头和尾巴较长。 | 1 | equality | S1T262 |
| 脖子和腿很长，头次之，尾巴最短。 | 1 | ranking | S1T276 |
| 脖子和腿比躯干长，头和尾巴长度差不多。 | 1 | equality, body_ref | S3T74 |
| 脖子很长，其他，差不多。 | 1 | equality | S2T155 |
| 脖子最短，其余各部位长度差不多。 | 1 | equality | S3T39 |
| 脖子最长，其他部位长度差不多。 | 1 | equality | S1T211 |
| 脖子最长，其他部位长度相当。 | 1 | equality | S1T186 |
| 脖子最长，头、尾巴和腿差不多。 | 1 | equality | S1T138 |
| 脖子最长，头、尾巴和腿都差不多长。 | 1 | equality | S2T69 |
| 脖子最长，头、尾巴长度也较长且相等，腿稍短。 | 1 | equality | S3T101 |
| 脖子最长，头、尾巴长度相当，腿稍短。 | 1 | equality | S3T66 |
| 脖子最长，头、躯干、腿长度相当，尾巴最短。 | 1 | equality, body_ref | S3T59 |
| 脖子最长，头、躯干长度相当，腿、尾巴最短。 | 1 | equality, body_ref | S3T51 |
| 脖子最长，头和尾巴第二、第三长，腿最短。 | 1 | ranking, count_abstract | S1T11 |
| 脖子最长，头和腿差不多，尾巴最短。 | 1 | equality | S1T66 |
| 脖子最长，头和腿的长度差不多，尾巴最短。 | 1 | equality | S1T99 |
| 脖子最长，头比躯干短，尾巴长。 | 1 | body_ref | S3T120 |
| 脖子最长，头，尾巴稍短，比躯干短，腿最短。 | 1 | body_ref | S3T60 |
| 脖子最长，尾巴和腿差不多，头最短。 | 1 | equality | S1T269 |
| 脖子最长，尾巴最短，头也短，腿第二长。 | 1 | ranking | S1T267 |
| 脖子最长，尾巴较短，腿和头长度在中间且差不多。 | 1 | equality | S2T66 |
| 脖子最长，腿第二。 | 1 | ranking | S1T274 |
| 脖子最长，腿第二，头和尾巴较短。 | 1 | ranking | S1T227 |
| 脖子第一长，尾巴第二长，头第三，腿最短。 | 1 | ranking | S2T95 |
| 腿和尾巴和躯干差不多长，头稍短，脖子最短。 | 1 | equality, body_ref | S3T17 |
| 腿和尾巴差不多长。 | 1 | equality | S1T282 |
| 腿和脖子差不多长。 | 1 | equality | S1T285 |
| 腿和脖子很长，比躯干长，头和尾巴也较长。 | 1 | body_ref | S3T55 |
| 腿和脖子长，头次之，尾巴最短。 | 1 | ranking | S1T14 |
| 腿很长，脖子第二，头和尾巴都很短。 | 1 | ranking | S2T31 |
| 腿最短，其他三个部分同样差不多长。 | 1 | equality | S2T261 |
| 腿最短，其余各部分差不多。 | 1 | equality | S2T272 |
| 腿最短，头、脖子、尾巴差不多长。 | 1 | equality | S2T71 |
| 腿最短，躯干最长，头、脖子、尾巴长度中间且差不多。 | 1 | equality, body_ref | S3T71 |
| 腿最长、比躯干长，头、尾巴和躯干长度差不多，脖子最短。 | 1 | equality, body_ref | S3T47 |
| 腿最长，其他部位相当。 | 1 | equality | S1T200 |
| 腿最长，其他部位都相对较短，且长度差不多。 | 1 | equality | S1T251 |
| 腿最长，其余差不多。 | 1 | equality | S2T320 |
| 腿最长，其余部分差不多。 | 1 | equality | S2T289 |
| 腿最长，头、尾巴、脖子较短，且长度差不多。 | 1 | equality | S1T236 |
| 腿最长，头、尾巴长度差不多，脖子稍短。 | 1 | equality | S3T70 |
| 腿最长，头、脖子长度稍短且相当，尾巴较长。 | 1 | equality | S3T27 |
| 腿最长，头和脖子差不多，尾巴最短。 | 1 | equality | S2T265 |
| 腿最长，头和躯干差不多，脖子最短，尾巴稍短。 | 1 | equality, body_ref | S3T78 |
| 腿最长，头比躯干短，尾巴最短。 | 1 | body_ref | S3T119 |
| 腿最长，头比躯干短，尾巴长。 | 1 | body_ref | S3T136 |
| 腿最长，头第二，脖子第三，尾巴最短。 | 1 | ranking | S1T181 |
| 腿最长，尾巴与长躯干差不多，头和脖子稍短。 | 1 | equality, body_ref | S3T54 |
| 腿最长，尾巴比头要稍稍长一点，脖子和头差不多长。 | 1 | equality | S2T70 |
| 腿最长，尾巴第二长，头第三长，脖子最短。 | 1 | ranking, count_abstract | S1T13 |
| 腿最长，尾巴第二，脖子第三，头最短。 | 1 | ranking | S1T265 |
| 腿最长，比躯干长，头、脖子较短，尾巴最短。 | 1 | body_ref | S3T58 |
| 腿最长，脖子和头次之，尾巴最短。 | 1 | ranking | S1T38 |
| 腿最长，脖子第二，头和尾巴较短。 | 1 | ranking | S1T217 |
| 腿最长，脖子第二，头第三，尾巴最短。 | 1 | ranking | S2T4 |
| 腿最长，脖子第二，尾巴第三，头最短。 | 1 | ranking | S1T81 |
| 腿最长，躯干第二，头、尾巴长度稍短，脖子最短。 | 1 | ranking, body_ref | S3T65 |
| 腿比躯干长，脖子和尾巴非常短，头比脖子和尾巴稍长。 | 1 | body_ref | S3T38 |
| 腿稍微短一些，头、脖子、尾巴差不多。 | 1 | equality | S2T102 |
| 腿长，头、脖子、尾巴差不多长。 | 1 | equality | S2T251 |
| 躯干比脖子短，头和尾巴稍短，腿最短。 | 1 | body_ref | S3T19 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位差不多长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T275, S2T30 |
| 四个部位较均衡。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T61, S2T101 |
| 各部位较均衡。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T37 |
| 各部位长度差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T21 |
| 四个部位差不多长，脖子稍微短一些。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 =; absolute_short:脖子 < 0.50 | S2T162 |
| 四个部位较均衡，头较长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 =; absolute_long:头 > 0.50 | S2T21 |
| 头、脖子和腿长度差不多，尾巴要稍短一些，和躯干的长度相当，全身比较均衡。 | 1 | 0.000 | equality_range:头+脖子+腿 =; absolute_short:尾巴 < 0.50; body_ref:尾巴 = 0.50 | S2T37 |
| 头和脖子同样长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T235 |
| 头和脖子长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T316 |
| 头和腿差不多长，脖子第二，尾巴最短。 | 1 | 0.000 | comparison:脖子 > 头+腿+尾巴 | S1T313 |
| 头短，其余各部位长度均衡。 | 1 | 0.000 | absolute_short:头 < 0.50; equality_range:脖子+腿+尾巴 = | S4T45 |
| 头短，其余都差不多长。 | 1 | 0.000 | absolute_short:头 < 0.50; complement:脖子 < 0.50; complement:腿 < 0.50; complement:尾巴 < 0.50 | S3T320 |
| 头较长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T21 |
| 头长，尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S4T21 |
| 尾巴较长，其余差不多。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; equality_range:脖子+头+腿 = | S2T319 |
| 脖子和腿差不多长。 | 1 | 0.000 | equality_range:脖子+腿 = | S1T311 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T250 |
| 头、脖子、腿和尾巴长度差不多，是躯干的一半。 | 1 | 0.200 | equality_range:头+脖子+腿+尾巴 =; body_ref:头 = 0.25; body_ref:脖子 = 0.25; body_ref:尾巴 = 0.25 | S3T40 |
| 头和脖子长，头和腿长，脖子和尾巴短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S2T200 |
| 头和腿长，脖子和腿，和尾巴短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T300 |
| 头长，腿长，尾巴更长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T247 |
| 头和脖子较长，尾巴较短，腿中间。 | 1 | 0.400 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute:腿 middle_lower | S1T111 |
| 脖子最长，头、尾巴长度也较长且相等，腿稍短。 | 1 | 0.400 | superlative:脖子 > 尾巴; equality_range:头+尾巴 =; absolute_short:腿 < 0.50 | S3T101 |

### S311

- trial 数: 1024; 非空文本: 1023; fidelity 可评分率: 0.998; 平均 fidelity: 0.923; 完全忠实率: 0.848; 低 fidelity 率: 0.032.
- 旧版 region 覆盖率: 0.998; 旧版 region 有未处理片段率: 0.043.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1021 | 0.997 |
| comparison | 369 | 0.360 |
| superlative | 306 | 0.299 |
| ranking | 99 | 0.097 |
| equality | 16 | 0.016 |
| empty | 1 | 0.001 |
| other | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴很长。 | 94 |
| 头很长，尾巴很短。 | 72 |
| 各部位都比较长。 | 54 |
| 头和尾巴都比较短。 | 46 |
| 头很长。 | 41 |
| 尾巴很长，脖子很短。 | 37 |
| 脖子和尾巴都比较长。 | 34 |
| 头和尾巴都很短。 | 29 |
| 尾巴和脖子都比较长。 | 19 |
| 尾巴很长，脖子比较短。 | 19 |
| 头很长，尾巴比较短。 | 17 |
| 头比较长，尾巴很短。 | 16 |
| 腿很长。 | 11 |
| 尾巴最长，脖子和头，腿最短。 | 11 |
| 尾巴比较长，脖子很短。 | 11 |
| 尾巴比较长，脖子比较短。 | 10 |
| 尾巴和脖子都很长。 | 8 |
| 头、尾巴和腿都比较长。 | 8 |
| 脖子最长，头和腿，尾巴最短。 | 7 |
| 尾巴很长，头比较短。 | 7 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 各部位长度接近。 | 3 | equality | S2T125, S2T159, S2T231 |
| 尾巴最长，脖子，再是头和腿。 | 3 | ranking | S1T146, S1T170, S1T199 |
| 脖子最长，腿，再是头，尾巴最短。 | 3 | ranking | S1T266, S1T267, S1T297 |
| 头和腿最长，其次是脖子，尾巴最短。 | 2 | ranking | S1T42, S1T43 |
| 头最长，其次是尾巴、脖子和腿。 | 2 | ranking | S1T57, S1T66 |
| 头最长，其次是尾巴、脖子，腿最短。 | 2 | ranking | S1T67, S1T74 |
| 头最长，其次是尾巴，再是腿，脖子最短。 | 2 | ranking | S1T37, S1T58 |
| 头最长，其次是脖子、腿，尾巴最短。 | 2 | ranking | S1T65, S1T69 |
| 尾巴最长，其次是头和脖子，腿最短。 | 2 | ranking | S1T22, S1T229 |
| 尾巴最长，其次是脖子、腿，头最短。 | 2 | ranking | S1T7, S1T63 |
| 脖子最长，其次是腿，再是头，尾巴最短。 | 2 | ranking | S1T12, S1T227 |
| 脖子最长，头，再是尾巴和腿。 | 2 | ranking | S1T177, S1T259 |
| 腿最长，头，再是尾巴，脖子最短。 | 2 | ranking | S1T23, S1T219 |
| 腿最长，尾巴，再是头，脖子最短。 | 2 | ranking | S1T280, S1T296 |
| 腿最长，脖子，再是尾巴和头。 | 2 | ranking | S1T176, S1T207 |
| 各部位比较均衡。 | 1 | equality | S2T64 |
| 各部位的长度相近。 | 1 | equality | S2T1 |
| 各部位长度比较均衡。 | 1 | equality | S2T68 |
| 头、脖子、尾巴、腿略长，且都比较接近。 | 1 | equality | S1T1 |
| 头和尾巴很长，其次是脖子，腿最短。 | 1 | ranking | S1T45 |
| 头和尾巴最长、且二者长度接近，脖子和腿长度比较接近、且比头和脖子略短。 | 1 | equality | S1T8 |
| 头和尾巴最长，其次是腿，脖子最短。 | 1 | ranking | S1T49 |
| 头和尾巴比较长、且比较接近，脖子和腿比较短、长度接近。 | 1 | equality | S1T4 |
| 头和尾巴长度较长，其次是脖子，腿最短。 | 1 | ranking | S1T14 |
| 头和脖子很长，其次是腿，尾巴最短。 | 1 | ranking | S1T46 |
| 头和脖子最长，其次是尾巴，腿最短。 | 1 | ranking | S1T221 |
| 头和脖子最长，腿，再是尾巴。 | 1 | ranking | S1T258 |
| 头和脖子比较长、且长度接近，腿和尾巴比较短、且二者接近。 | 1 | equality | S1T5 |
| 头和脖子比较长、长度接近，腿和尾巴比较短、长度接近。 | 1 | equality | S1T2 |
| 头和腿很长，其次是尾巴，脖子最短。 | 1 | ranking | S1T113 |
| 头和腿最长，尾巴，再是脖子。 | 1 | ranking | S1T257 |
| 头和腿比较长、且长度接近，尾巴和脖子略短、且长度接近。 | 1 | equality | S1T11 |
| 头最短，其次是腿、脖子，尾巴最长。 | 1 | ranking | S1T15 |
| 头最短，尾巴比头略长，脖子和腿比较长、且比较接近。 | 1 | equality | S1T10 |
| 头最长，其次是尾巴和腿，脖子最短。 | 1 | ranking | S1T225 |
| 头最长，其次是尾巴，再是脖子，腿最短。 | 1 | ranking | S1T16 |
| 头最长，其次是脖子、腿和尾巴。 | 1 | ranking | S1T75 |
| 头最长，其次是脖子，再是尾巴，腿最短。 | 1 | ranking | S1T40 |
| 头最长，其次是脖子，腿和尾巴较短。 | 1 | ranking | S1T82 |
| 头最长，其次是腿、尾巴，脖子最短。 | 1 | ranking | S1T73 |
| 头最长，其次是腿和尾巴，脖子最短。 | 1 | ranking | S1T129 |
| 头最长，其次是腿和脖子，尾巴最短。 | 1 | ranking | S1T140 |
| 头最长，其次是腿，再是脖子，尾巴最短。 | 1 | ranking | S1T53 |
| 头最长，再是尾巴、脖子、腿最短。 | 1 | ranking | S1T59 |
| 头最长，尾巴和脖子，再是腿。 | 1 | ranking | S1T215 |
| 头最长，尾巴，再是脖子和腿。 | 1 | ranking | S1T203 |
| 头最长，尾巴，再是腿，脖子最短。 | 1 | ranking | S1T178 |
| 头最长，脖子和尾巴长度比较接近、且比头短很多，腿最短。 | 1 | equality | S1T9 |
| 头最长，脖子，再是尾巴和腿。 | 1 | ranking | S1T206 |
| 头最长，脖子，再是腿，尾巴最短。 | 1 | ranking | S1T256 |
| 头最长，腿，再是尾巴和脖子。 | 1 | ranking | S1T295 |
| 尾巴和头。 | 1 | other | S2T8 |
| 尾巴最长，其次是头、腿，脖子最短。 | 1 | ranking | S1T84 |
| 尾巴最长，其次是头，再是脖子、腿。 | 1 | ranking | S1T50 |
| 尾巴最长，其次是头，再是脖子和腿。 | 1 | ranking | S1T34 |
| 尾巴最长，其次是头，再是脖子，腿最短。 | 1 | ranking | S1T56 |
| 尾巴最长，其次是头，再是腿，脖子最短。 | 1 | ranking | S1T33 |
| 尾巴最长，其次是头，腿和脖子最短。 | 1 | ranking | S1T62 |
| 尾巴最长，其次是脖子和头，腿最短。 | 1 | ranking | S1T114 |
| 尾巴最长，其次是脖子，再是头，腿最短。 | 1 | ranking | S1T54 |
| 尾巴最长，其次是腿、头和脖子。 | 1 | ranking | S1T38 |
| 尾巴最长，其次是腿、头和脖子最短。 | 1 | ranking | S1T55 |
| 尾巴最长，其次是腿、脖子和头。 | 1 | ranking | S1T83 |
| 尾巴最长，其次是腿、脖子，头最短。 | 1 | ranking | S1T64 |
| 尾巴最长，其次是腿，再是脖子，头最短。 | 1 | ranking | S1T224 |
| 尾巴最长，其次是腿，头，脖子最短。 | 1 | ranking | S1T24 |
| 尾巴最长，头，再是脖子和腿。 | 1 | ranking | S1T159 |
| 尾巴最长，头，再是脖子，腿最短。 | 1 | ranking | S1T254 |
| 尾巴最长，头，再是腿，脖子最短。 | 1 | ranking | S1T255 |
| 尾巴最长，脖子和头，再是腿。 | 1 | ranking | S1T200 |
| 尾巴最长，脖子，再是头，腿最短。 | 1 | ranking | S1T193 |
| 尾巴最长，腿，再是脖子和头。 | 1 | ranking | S1T252 |
| 脖子、腿、尾巴比较长，且比较接近，头最短。 | 1 | equality | S1T6 |
| 脖子、腿、尾巴比较长，其中脖子最长，腿、尾巴其次，头比较短。 | 1 | ranking | S1T3 |
| 脖子和腿比较长、且长度接近，头和尾巴长度略短。 | 1 | equality | S1T19 |
| 脖子最长，其次是头、腿，尾巴最短。 | 1 | ranking | S1T39 |
| 脖子最长，其次是头，腿和尾巴最短。 | 1 | ranking | S1T72 |
| 脖子最长，其次是尾巴、头，和腿。 | 1 | ranking | S1T115 |
| 脖子最长，其次是尾巴和头，腿最短。 | 1 | ranking | S1T232 |
| 脖子最长，其次是尾巴，再是头，腿最短。 | 1 | ranking | S1T41 |
| 脖子最长，其次是尾巴，再是腿，头最短。 | 1 | ranking | S1T48 |
| 脖子最长，其次是腿、头和尾巴很短。 | 1 | ranking | S1T139 |
| 脖子最长，其次是腿、头和尾巴长度较短。 | 1 | ranking | S1T13 |
| 脖子最长，其次是腿，头和尾巴很短。 | 1 | ranking | S1T141 |
| 脖子最长，其次是腿，头和尾巴比较短。 | 1 | ranking | S1T145 |
| 脖子最长，腿，再是尾巴和头。 | 1 | ranking | S1T26 |
| 脖子略长，尾巴，再是腿和头。 | 1 | ranking | S1T270 |
| 腿和尾巴很长，其次是头，脖子最短。 | 1 | ranking | S1T17 |
| 腿和尾巴最长，头，再是脖子。 | 1 | ranking | S1T253 |
| 腿和尾巴最长，脖子，再是头。 | 1 | ranking | S1T218 |
| 腿最长，其次是头和脖子，腿。 | 1 | ranking | S1T222 |
| 腿最长，其次是头，再是尾巴，最后是脖子。 | 1 | ranking | S1T20 |
| 腿最长，其次是头，再是脖子，尾巴最短。 | 1 | ranking | S1T18 |
| 腿最长，其次是尾巴、脖子，头最短。 | 1 | ranking | S1T68 |
| 腿最长，其次是脖子，再是尾巴和头。 | 1 | ranking | S1T51 |
| 腿最长，头，其次是脖子、尾巴和头。 | 1 | ranking | S1T212 |
| 腿最长，尾巴，再是脖子和头。 | 1 | ranking | S1T271 |
| 腿最长，脖子，再是头，尾巴最短。 | 1 | ranking | S1T35 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头很长。 | 5 | 0.000 | absolute_long:头 > 0.50 | S2T74, S2T127, S2T184, S2T236, S3T54 |
| 各部位都比较长。 | 5 | 0.200 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S2T263, S2T265, S2T278, S3T1, S3T125 |
| 尾巴很长。 | 3 | 0.000 | absolute_long:尾巴 > 0.50 | S2T84, S2T197, S2T254 |
| 尾巴比较长。 | 3 | 0.000 | absolute_long:尾巴 > 0.50 | S2T124, S3T14, S3T274 |
| 各部位长度接近。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T125, S2T159 |
| 脖子和尾巴都比较长。 | 2 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S3T281, S4T1 |
| 腿比较长。 | 2 | 0.000 | absolute_long:腿 > 0.50 | S2T51, S2T212 |
| 各部位比较均衡。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T64 |
| 各部位长度比较均衡。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T68 |
| 头、脖子、尾巴、腿略长，且都比较接近。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50 | S1T1 |
| 头很长，尾巴很短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_short:尾巴 < 0.50 | S2T169 |
| 脖子很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T72 |
| 腿很长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S2T75 |
| 头、脖子和尾巴略长，腿略短。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T231 |
| 头、尾巴和腿都比较长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T158 |
| 头、脖子和尾巴都比较长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T301 |
| 头和腿最长，尾巴和脖子。 | 1 | 0.333 | superlative:腿 > 头; superlative:腿 > 尾巴 | S1T286 |
| 脖子很长，头和尾巴比较短。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_short:尾巴 < 0.50 | S2T193 |

### S312

- trial 数: 1344; 非空文本: 1336; fidelity 可评分率: 0.991; 平均 fidelity: 0.895; 完全忠实率: 0.818; 低 fidelity 率: 0.062.
- 旧版 region 覆盖率: 0.991; 旧版 region 有未处理片段率: 0.007.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1335 | 0.993 |
| comparison | 102 | 0.076 |
| equality | 88 | 0.065 |
| superlative | 18 | 0.013 |
| empty | 8 | 0.006 |
| body_ref | 3 | 0.002 |
| meta | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长。 | 42 |
| 脖子长。 | 36 |
| 四个部位一样长。 | 33 |
| 头长，尾巴短。 | 28 |
| 尾巴短，脖子长。 | 26 |
| 头和尾巴长。 | 26 |
| 腿长。 | 22 |
| 尾巴长。 | 21 |
| 头长，尾巴长。 | 20 |
| 头、脖子、尾巴长，腿短。 | 20 |
| 尾巴长，头短。 | 19 |
| 头和腿长。 | 19 |
| 头和脖子长。 | 18 |
| 脖子略长。 | 16 |
| 尾巴长，头长。 | 16 |
| 尾巴短。 | 16 |
| 尾巴短，脖子短。 | 16 |
| 头很长。 | 15 |
| 头长，腿长。 | 15 |
| 尾巴比头长。 | 15 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位一样长。 | 33 | equality | S1T12, S1T16, S1T26, S1T33, S1T37, S1T58, S1T60, S1T73 |
| 脖子、腿、尾巴一样长，头短。 | 7 | equality | S1T112, S1T129, S1T157, S1T172, S1T174, S1T254, S1T259 |
| 四个部位差不多长。 | 5 | equality | S1T45, S1T64, S1T68, S1T70, S2T101 |
| 四个部位长度差不多。 | 4 | equality | S1T194, S1T269, S2T68, S2T80 |
| 头、脖子、尾巴一样长，腿短。 | 4 | equality | S1T100, S1T135, S1T166, S1T169 |
| 头、腿、尾巴一样长，脖子短。 | 3 | equality | S1T101, S1T123, S1T251 |
| 脖子、腿、尾巴一样长。 | 3 | equality | S1T149, S1T152, S1T182 |
| 头、脖子、尾巴一样短，腿长。 | 2 | equality | S1T121, S1T212 |
| 一样长。 | 1 | equality | S1T197 |
| 四个部位差不多长，腿和头略长。 | 1 | equality | S2T109 |
| 头、尾巴、腿一样长，脖子短。 | 1 | equality | S1T117 |
| 头、尾巴一样短，脖子最短，腿很长。 | 1 | equality | S3T213 |
| 头、脖子、尾巴一样长。 | 1 | equality | S1T201 |
| 头、脖子、尾巴一样长，腿更长。 | 1 | equality | S2T70 |
| 头、脖子、尾巴和腿一样长。 | 1 | equality | S1T97 |
| 头、脖子、腿一样长，尾巴最长。 | 1 | equality | S1T143 |
| 头、脖子、腿一样长，尾巴略短。 | 1 | equality | S2T64 |
| 头、脖子、腿一样长，尾巴短。 | 1 | equality | S1T99 |
| 头、脖子和尾巴一样长。 | 1 | equality | S1T231 |
| 头、脖子和尾巴一样长，腿短。 | 1 | equality | S1T167 |
| 头、腿、尾巴一样长，脖子更短。 | 1 | equality | S1T115 |
| 头、腿、尾巴一样长，脖子更长。 | 1 | equality | S1T114 |
| 头、腿、尾巴差不多长，脖子较长。 | 1 | equality | S1T84 |
| 头和尾巴一样长，腿长。 | 1 | equality | S4T3 |
| 头和腿一样长，脖子和尾巴短。 | 1 | equality | S1T122 |
| 头和腿比躯干长。 | 1 | body_ref | S4T68 |
| 头和腿较短，脖子和尾巴一样长。 | 1 | equality | S1T47 |
| 头很短，腿、脖子、尾巴一样长。 | 1 | equality | S1T93 |
| 头短，脖子、腿和尾巴一样长。 | 1 | equality | S1T144 |
| 头长，脖子、腿、尾巴一样长。 | 1 | equality | S1T215 |
| 尾巴很短，头、脖子、腿一样长。 | 1 | equality | S1T46 |
| 尾巴比头长、都比躯干短。 | 1 | body_ref | S4T69 |
| 尾巴比躯干长，头比躯干短。 | 1 | body_ref | S4T65 |
| 尾巴长，腿略短，头和脖子一样长。 | 1 | equality | S4T67 |
| 差不多长。 | 1 | equality | S2T254 |
| 脖子、腿、尾巴一样长，头最长。 | 1 | equality | S1T220 |
| 脖子和尾巴一样长，头和腿一样短。 | 1 | equality | S3T235 |
| 脖子很短，头很短，腿和尾巴差不多长。 | 1 | equality | S1T95 |
| 选错了。 | 1 | meta | S2T140 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位一样长。 | 31 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T12, S1T16, S1T26, S1T33, S1T37, S1T58, S1T60, S1T73 |
| 四个部位差不多长。 | 5 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T45, S1T64, S1T68, S1T70, S2T101 |
| 脖子长。 | 5 | 0.000 | absolute_long:脖子 > 0.50 | S1T256, S1T257, S2T54, S2T258, S4T54 |
| 四个部位长度差不多。 | 4 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T194, S1T269, S2T68, S2T80 |
| 脖子略长。 | 4 | 0.000 | absolute_long:脖子 > 0.50 | S2T173, S3T173, S3T228, S3T286 |
| 头长，尾巴长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T281, S1T288, S1T313 |
| 尾巴略短。 | 3 | 0.000 | absolute_short:尾巴 < 0.50 | S3T221, S3T278, S4T58 |
| 脖子、腿、尾巴一样长。 | 3 | 0.000 | equality_range:脖子+腿+尾巴 = | S1T149, S1T152, S1T182 |
| 都短头和尾巴略长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S3T262, S3T287 |
| 头、脖子、尾巴长。 | 2 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T20, S3T20 |
| 脖子、腿长、尾巴短。 | 2 | 0.333 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S2T143, S3T73 |
| 头、脖子、尾巴一样长。 | 1 | 0.000 | equality_range:头+脖子+尾巴 = | S1T201 |
| 头、脖子、尾巴和腿一样长。 | 1 | 0.000 | equality_range:头+脖子+尾巴+腿 = | S1T97 |
| 头、脖子和尾巴一样长。 | 1 | 0.000 | equality_range:头+脖子+尾巴 = | S1T231 |
| 头和尾巴略长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T76 |
| 头和尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T131 |
| 头和腿长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T18 |
| 头很长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T62 |
| 尾巴很长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T61 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S2T61 |
| 脖子很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T54 |
| 脖子略短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S3T109 |
| 腿和脖子略短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50 | S3T220 |
| 腿很长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T18 |
| 都较短，头略长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S3T253 |
| 头、尾巴长，腿短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T262 |
| 头、脖子长，尾巴短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T300 |
| 头和尾巴长，腿短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T122 |
| 头和脖子长，尾巴短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T301 |
| 头长，尾巴长，腿短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T106 |
| 尾巴长，头长，腿短。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50 | S4T122 |

### S313

- trial 数: 960; 非空文本: 959; fidelity 可评分率: 0.998; 平均 fidelity: 0.862; 完全忠实率: 0.626; 低 fidelity 率: 0.027.
- 旧版 region 覆盖率: 0.998; 旧版 region 有未处理片段率: 0.007.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 928 | 0.967 |
| comparison | 192 | 0.200 |
| body_ref | 142 | 0.148 |
| equality | 76 | 0.079 |
| superlative | 3 | 0.003 |
| empty | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴短，脖子长。 | 27 |
| 尾巴长，头短。 | 20 |
| 尾巴短，脖子短。 | 15 |
| 脖子短，头短，腿长，尾巴短。 | 14 |
| 脖子短，头长，腿短，尾巴长。 | 13 |
| 脖子短，头短，腿短，尾巴长。 | 13 |
| 脖子短，尾巴短。 | 13 |
| 脖子短，头短，腿长，尾巴长。 | 12 |
| 头短，尾巴短，腿短。 | 12 |
| 脖子长，头短，腿长，尾巴短。 | 12 |
| 脖子短，头短，腿短，尾巴短。 | 11 |
| 脖子长，头短，腿短，尾巴短。 | 11 |
| 尾巴短，头短。 | 11 |
| 脖子短，尾巴短，腿长。 | 9 |
| 头、尾巴、腿比躯干短，脖子比躯干长。 | 9 |
| 脖子短，尾巴短，腿短。 | 8 |
| 头短，尾巴短。 | 8 |
| 脖子短，头长，腿短，尾巴短。 | 8 |
| 脖子短，腿短。 | 8 |
| 脖子短，头长，腿长，尾巴短。 | 8 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头、尾巴、腿比躯干短，脖子比躯干长。 | 9 | body_ref | S3T1, S3T22, S3T31, S3T51, S3T74, S3T100, S3T101, S3T106 |
| 头、脖子、尾巴、腿均小于躯干。 | 7 | body_ref | S3T76, S3T85, S3T88, S3T114, S3T120, S3T128, S3T144 |
| 头、脖子、尾巴均小于躯干，腿长于躯干。 | 5 | body_ref | S3T80, S3T122, S3T132, S3T139, S3T140 |
| 头、脖子、尾巴、腿均小于等于躯干。 | 4 | equality, body_ref | S3T39, S3T56, S3T57, S3T133 |
| 头、脖子和腿比躯干短，尾巴比躯干长。 | 4 | body_ref | S3T8, S3T36, S3T38, S3T99 |
| 头、尾巴、腿均小于躯干，脖子长于躯干。 | 3 | body_ref | S3T82, S3T116, S3T131 |
| 头、脖子、尾巴比躯干短，腿比躯干长。 | 3 | body_ref | S3T2, S3T28, S3T34 |
| 头、脖子、腿、尾巴均小于躯干。 | 3 | body_ref | S3T65, S3T78, S3T138 |
| 头、脖子、腿和尾巴比躯干短。 | 3 | body_ref | S3T23, S3T32, S3T35 |
| 头、脖子、腿比躯干短，尾巴比躯干长。 | 3 | body_ref | S3T13, S3T16, S3T97 |
| 头、尾巴、腿短于躯干，脖子长于躯干。 | 2 | body_ref | S3T92, S3T141 |
| 头、脖子、尾巴、腿比躯干短。 | 2 | body_ref | S3T102, S3T103 |
| 头、脖子、尾巴和腿比躯干短。 | 2 | body_ref | S3T30, S3T71 |
| 头、脖子、尾巴和腿都比躯干短。 | 2 | body_ref | S3T3, S3T4 |
| 头、脖子、尾巴短于躯干，腿长于躯干。 | 2 | body_ref | S3T55, S3T90 |
| 头、脖子、腿、尾巴均小于等于躯干。 | 2 | equality, body_ref | S3T58, S3T130 |
| 头、脖子和腿短于等于躯干，尾巴长于躯干。 | 2 | equality, body_ref | S3T52, S3T75 |
| 头、脖子和腿短于躯干，尾巴长于躯干。 | 2 | body_ref | S3T61, S3T125 |
| 头和尾巴短于躯干，脖子和腿长于躯干。 | 2 | body_ref | S3T126, S3T142 |
| 头和腿比躯干短，尾巴和脖子比躯干长。 | 2 | body_ref | S3T29, S3T64 |
| 头长于躯干，尾巴、脖子、腿短于躯干。 | 2 | body_ref | S3T77, S3T96 |
| 尾巴比躯干短，头、脖子、腿比躯干长。 | 2 | body_ref | S3T49, S3T108 |
| 尾巴短于躯干，头、脖子、腿长于躯干。 | 2 | body_ref | S3T91, S3T93 |
| 脖子、尾巴、腿均小于躯干，头长于躯干。 | 2 | body_ref | S3T115, S3T135 |
| 脖子、尾巴、腿比躯干短，头比躯干长。 | 2 | body_ref | S3T7, S3T105 |
| 脖子短，尾巴和头长度差不多。 | 2 | equality | S2T291, S2T294 |
| 四个部位一样长。 | 1 | equality | S1T25 |
| 四个部位差不多长。 | 1 | equality | S2T25 |
| 四个部位长度接近。 | 1 | equality | S1T86 |
| 头、尾巴、脖子、腿均小于等于躯干。 | 1 | equality, body_ref | S3T113 |
| 头、尾巴、脖子、腿均小于躯干。 | 1 | body_ref | S3T95 |
| 头、尾巴、脖子均小于等于躯干，腿长于躯干。 | 1 | equality, body_ref | S3T118 |
| 头、尾巴、腿均小于等于躯干，脖子长于躯干。 | 1 | equality, body_ref | S3T137 |
| 头、尾巴、腿均小于躯干，尾巴长于躯干。 | 1 | body_ref | S3T117 |
| 头、脖子、尾巴、腿均小于或等于，躯干。 | 1 | equality, body_ref | S3T53 |
| 头、脖子、尾巴、腿都小于等于躯干。 | 1 | equality, body_ref | S3T41 |
| 头、脖子、尾巴和腿比躯干长。 | 1 | body_ref | S3T25 |
| 头、脖子、尾巴均小于等于躯干，腿长于躯干。 | 1 | equality, body_ref | S3T121 |
| 头、脖子、尾巴都小于躯干，腿大于躯干。 | 1 | body_ref | S3T45 |
| 头、脖子、尾巴长度差不多，腿长。 | 1 | equality | S2T148 |
| 头、脖子、腿、尾巴都小于躯干。 | 1 | body_ref | S3T40 |
| 头、脖子、腿和尾巴均小于躯干。 | 1 | body_ref | S3T136 |
| 头、脖子、腿和尾巴比躯干都短。 | 1 | body_ref | S3T69 |
| 头、脖子、腿均小于等于躯干，尾巴长于躯干。 | 1 | equality, body_ref | S3T129 |
| 头、脖子、腿均小于躯干，尾巴长于躯干。 | 1 | body_ref | S3T84 |
| 头、脖子和尾巴比躯干长，腿比躯干短。 | 1 | body_ref | S3T104 |
| 头、脖子和腿长度差不多，尾巴很长。 | 1 | equality | S2T142 |
| 头、腿、尾巴、脖子比躯干都短。 | 1 | body_ref | S3T11 |
| 头、腿、尾巴短于躯干，脖子长于躯干。 | 1 | body_ref | S3T67 |
| 头、腿、脖子、尾巴比躯干都短。 | 1 | body_ref | S3T10 |
| 头和尾巴小于躯干，腿大于躯干。 | 1 | body_ref | S3T46 |
| 头和尾巴比躯干短，脖子和腿比躯干长。 | 1 | body_ref | S3T98 |
| 头和尾巴比躯干长，脖子和腿比躯干短。 | 1 | body_ref | S3T70 |
| 头和尾巴长于躯干，脖子和腿短于躯干。 | 1 | body_ref | S3T111 |
| 头和尾巴长度差不多、较短，脖子很长，腿短。 | 1 | equality | S2T140 |
| 头和尾巴长度差不多，四个部位都长，脖子较短，腿短。 | 1 | equality | S2T147 |
| 头和尾巴长度差不多，脖子很短，腿较长。 | 1 | equality | S2T139 |
| 头和尾巴长度差不多，脖子长，腿长。 | 1 | equality | S2T141 |
| 头和脖子均大于躯干，尾巴和腿短于躯干。 | 1 | body_ref | S3T134 |
| 头和脖子短于躯干，尾巴和腿长于躯干。 | 1 | body_ref | S3T123 |
| 头和脖子短于躯干，腿和尾巴长于躯干。 | 1 | body_ref | S3T87 |
| 头和脖子都小于躯干，腿大于躯干，尾巴等于躯干。 | 1 | equality, body_ref | S3T42 |
| 头和腿比躯干短，脖子和尾巴比躯干长。 | 1 | body_ref | S3T5 |
| 头和腿，脖子和尾巴比躯干短。 | 1 | body_ref | S3T37 |
| 头很长，尾巴、腿、脖子差不多长，都比头稍微短一点。 | 1 | equality | S2T70 |
| 头是最长，脖子、尾巴、腿长度差不多，四个部位都短。 | 1 | equality | S2T174 |
| 头比躯干短，脖子、腿、尾巴比躯干长。 | 1 | body_ref | S3T72 |
| 头短于躯干，脖子、腿、尾巴长于躯干。 | 1 | body_ref | S3T143 |
| 头短，尾巴短，头和尾巴长度差不多，脖子较长。 | 1 | equality | S2T144 |
| 头短，尾巴短，腿短，脖子短，尾巴、脖子、腿差不多。 | 1 | equality | S1T107 |
| 头短，脖子和尾巴长度差不多，腿长。 | 1 | equality | S2T168 |
| 头短，脖子短，尾巴短，腿长，头、尾巴和脖子差不多长。 | 1 | equality | S2T71 |
| 尾巴、头和脖子长度差不多。 | 1 | equality | S2T202 |
| 尾巴、脖子、头长度差不多，腿比较短。 | 1 | equality | S2T145 |
| 尾巴、脖子、腿比躯干短，头比躯干长。 | 1 | body_ref | S3T6 |
| 尾巴、脖子、腿短于躯干，头长于躯干。 | 1 | body_ref | S3T94 |
| 尾巴、腿、头比躯干短，脖子比躯干长。 | 1 | body_ref | S3T17 |
| 尾巴、腿、脖子均小于躯干，头长于躯干。 | 1 | body_ref | S3T124 |
| 尾巴、腿、脖子大于躯干，头小于躯干。 | 1 | body_ref | S3T48 |
| 尾巴、腿、脖子短于躯干，头长于躯干。 | 1 | body_ref | S3T62 |
| 尾巴、腿均小于躯干，脖子、头均大于躯干。 | 1 | body_ref | S3T81 |
| 尾巴和头差不多，脖子短。 | 1 | equality | S2T285 |
| 尾巴和头比躯干短腿和脖子比躯干长。 | 1 | body_ref | S3T26 |
| 尾巴和头长度差不多，脖子很短，腿长。 | 1 | equality | S2T276 |
| 尾巴和头长度差不多，脖子很长，腿较短。 | 1 | equality | S2T146 |
| 尾巴和头长度差不多，脖子短。 | 1 | equality | S2T301 |
| 尾巴和头长度差不多，腿和脖子很长，腿较短。 | 1 | equality | S2T204 |
| 尾巴和脖子长于躯干，头和腿短于躯干。 | 1 | body_ref | S3T63 |
| 尾巴和脖子长度差不多，头短。 | 1 | equality | S2T302 |
| 尾巴和脖子长度差不多，头较短。 | 1 | equality | S2T197 |
| 尾巴和脖子长度差不多，头较短，腿很短。 | 1 | equality | S2T198 |
| 尾巴和腿均小于躯干，头和脖子长于躯干。 | 1 | body_ref | S3T89 |
| 尾巴和腿小于躯干，头和脖子大于躯干。 | 1 | body_ref | S3T47 |
| 尾巴和腿比躯干短，头比躯干很长。 | 1 | body_ref | S3T9 |
| 尾巴和腿比躯干短，脖子和头比躯干长。 | 1 | body_ref | S3T12 |
| 尾巴和腿短于等于躯干，脖子和头长于等于躯干。 | 1 | equality, body_ref | S3T54 |
| 尾巴和腿长于躯干，头和脖子短于躯干。 | 1 | body_ref | S3T66 |
| 尾巴和腿长于躯干，脖子和头短于等于躯干。 | 1 | equality, body_ref | S3T79 |
| 尾巴小于躯干，头、脖子、腿大于躯干。 | 1 | body_ref | S3T43 |
| 尾巴比躯干长，头、脖子、腿短于等于躯干。 | 1 | equality, body_ref | S3T109 |
| 尾巴短，头短，腿比较短，腿和脖子差不多长。 | 1 | equality | S2T35 |
| 尾巴短，头长，四个部位差不多。 | 1 | equality | S2T11 |
| 尾巴短，脖子短，头长，尾巴和脖子一样短。 | 1 | equality | S2T312 |
| 尾巴短，脖子长，头短，头和尾巴长度差不多。 | 1 | equality | S2T300 |
| 尾巴短，脖子长，尾巴和头长度差不多。 | 1 | equality | S2T271 |
| 尾巴短，脖子长，尾巴和腿差不多长。 | 1 | equality | S2T187 |
| 尾巴短，腿、脖子、头长度差不多。 | 1 | equality | S2T91 |
| 尾巴短，腿长，脖子短，头短，头比脖子短，头和尾巴长度差不多。 | 1 | equality | S2T90 |
| 尾巴长，头长，四个部位长度差不多。 | 1 | equality | S2T112 |
| 尾巴长，脖子短，头长，头和尾巴差不多长。 | 1 | equality | S2T241 |
| 尾巴长，腿长，脖子和头一样长。 | 1 | equality | S2T317 |
| 脖子、尾巴、腿短于躯干，头长于躯干。 | 1 | body_ref | S3T68 |
| 脖子、腿、尾巴比躯干短，头比躯干长。 | 1 | body_ref | S3T15 |
| 脖子、腿比躯干短，头和尾巴比躯干长。 | 1 | body_ref | S3T24 |
| 脖子和头短于躯干，尾巴和腿长于躯干。 | 1 | body_ref | S3T59 |
| 脖子和尾巴均小于躯干，头和腿长于躯干。 | 1 | body_ref | S3T119 |
| 脖子和尾巴大于躯干，头和腿小于躯干。 | 1 | body_ref | S3T44 |
| 脖子和尾巴比躯干短或相等，头和腿比躯干长。 | 1 | equality, body_ref | S3T50 |
| 脖子和尾巴短于等于躯干，头和腿长于躯干。 | 1 | equality, body_ref | S3T112 |
| 脖子和尾巴长于躯干，头和腿短于躯干。 | 1 | body_ref | S3T60 |
| 脖子和尾巴长度差不多，头很长。 | 1 | equality | S2T210 |
| 脖子和尾巴长度差不多，头短，腿很长。 | 1 | equality | S2T143 |
| 脖子和腿均大于躯干，头和尾巴均小于躯干。 | 1 | body_ref | S3T83 |
| 脖子和腿比躯干短，头和尾巴比躯干长。 | 1 | body_ref | S3T73 |
| 脖子和腿比躯干短，尾巴和头比躯干长。 | 1 | body_ref | S3T27 |
| 脖子和腿长于躯干，头和尾巴短于躯干。 | 1 | body_ref | S3T127 |
| 脖子和腿长度差不多，四个部位都比较短，尾巴和头都很长。 | 1 | equality | S2T208 |
| 脖子比躯干短，尾巴、腿和头比躯干长。 | 1 | body_ref | S3T21 |
| 脖子短，头、腿、尾巴长度差不多。 | 1 | equality | S2T21 |
| 脖子短，头和尾巴长度差不多，四个部位都长。 | 1 | equality | S2T209 |
| 脖子短，头和尾巴长度差不多，腿很长。 | 1 | equality | S2T203 |
| 脖子短，尾巴和头长度差不多，四个部位都长，腿很长。 | 1 | equality | S2T207 |
| 脖子短，尾巴和头长度差不多，腿较短。 | 1 | equality | S2T167 |
| 脖子短，尾巴短，头比躯干短、比脖子长，腿较长。 | 1 | body_ref | S2T56 |
| 脖子较短，腿很长，尾巴和头长度差不多。 | 1 | equality | S2T265 |
| 脖子长于躯干，头、尾巴、腿短于躯干。 | 1 | body_ref | S3T86 |
| 腿、脖子和头比躯干长，尾巴比躯干短。 | 1 | body_ref | S3T33 |
| 腿和脖子短于等于躯干，头和尾巴长于躯干。 | 1 | equality, body_ref | S3T110 |
| 腿是最长，其他三个差不多。 | 1 | equality | S2T173 |
| 腿比躯干短，头、脖子、尾巴比躯干长。 | 1 | body_ref | S3T14 |
| 腿比较长一点，四个部位差不多长。 | 1 | equality | S1T79 |
| 腿短，头、脖子、尾巴长度差不多。 | 1 | equality | S2T181 |
| 腿短，尾巴长，脖子长，头长，头和尾巴长度差不多，脖子比尾巴和头长一点。 | 1 | equality | S2T86 |
| 腿短，脖子和尾巴长度差不多。 | 1 | equality | S2T129 |
| 腿长，尾巴短，头短，脖子短，脖子、头和尾巴长度差不多，头比脖子短。 | 1 | equality | S2T93 |
| 腿长，尾巴长，头和脖子差不多长，头和脖子都比尾巴和腿短。 | 1 | equality | S2T79 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头短，尾巴短，腿短。 | 2 | 0.333 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S1T102, S1T106 |
| 尾巴短，脖子短，头很长。 | 2 | 0.333 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S2T243, S2T307 |
| 四个部位一样长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T25 |
| 四个部位差不多长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T25 |
| 四个部位长度接近。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T86 |
| 头短，尾巴短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S1T80 |
| 腿长，尾巴短，头短，脖子短，脖子、头和尾巴长度差不多，头比脖子短。 | 1 | 0.167 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50; absolute_short:脖子 < 0.50; equality_range:脖子+头+尾巴 = | S2T93 |
| 头、脖子、尾巴均小于躯干，腿长于躯干。 | 1 | 0.250 | body_ref:头 < 0.50; body_ref:脖子 < 0.50; body_ref:尾巴 < 0.50 | S3T132 |
| 头、脖子、腿、尾巴均小于等于躯干。 | 1 | 0.250 | body_ref:头 < 0.50; body_ref:脖子 < 0.50; body_ref:尾巴 < 0.50 | S3T130 |
| 尾巴短，头短，脖子短，腿比较长。 | 1 | 0.250 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50; absolute_short:脖子 < 0.50 | S2T132 |
| 尾巴短，脖子长，头短，头和尾巴长度差不多。 | 1 | 0.250 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50; equality_range:头+尾巴 = | S2T300 |
| 尾巴长，腿短，脖子短，头短。 | 1 | 0.250 | absolute_short:腿 < 0.50; absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S2T59 |
| 脖子短，头短，腿短，尾巴短。 | 1 | 0.250 | absolute_short:脖子 < 0.50; absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S3T213 |
| 尾巴短，脖子短，腿较长。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T159 |
| 尾巴短，脖子较短，腿很长。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S2T156 |
| 尾巴短，腿短，头短。 | 1 | 0.333 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S1T161 |
| 尾巴短，腿短，头长。 | 1 | 0.333 | absolute_short:腿 < 0.50; absolute_long:头 > 0.50 | S1T303 |
| 尾巴长，脖子较长，头很短。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S2T228 |
| 脖子短，头短，尾巴长。 | 1 | 0.333 | absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S1T59 |
| 脖子短，尾巴和头长度差不多，腿较短。 | 1 | 0.333 | absolute_short:脖子 < 0.50; equality_range:尾巴+头 = | S2T167 |
| 脖子短，尾巴短，腿短。 | 1 | 0.333 | absolute_short:脖子 < 0.50; absolute_short:腿 < 0.50 | S1T273 |
| 脖子短，尾巴短，腿长。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute_long:腿 > 0.50 | S1T212 |
| 腿短，尾巴短，脖子短。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T278 |
| 腿长，尾巴短，头短。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50 | S1T132 |

### S314

- trial 数: 832; 非空文本: 828; fidelity 可评分率: 0.981; 平均 fidelity: 0.372; 完全忠实率: 0.149; 低 fidelity 率: 0.410.
- 旧版 region 覆盖率: 0.981; 旧版 region 有未处理片段率: 0.026.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 822 | 0.988 |
| comparison | 818 | 0.983 |
| body_ref | 722 | 0.868 |
| other | 6 | 0.007 |
| empty | 4 | 0.005 |
| count_abstract | 3 | 0.004 |
| superlative | 3 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 躯干长于脖子，且躯干长于腿。 | 72 |
| 躯干短于脖子，且脖子长于尾巴。 | 69 |
| 躯干长于脖子，且躯干短于腿。 | 60 |
| 躯干短于脖子，且脖子短于尾巴。 | 38 |
| 躯干长于脖子，且脖子短于尾巴。 | 33 |
| 躯干短于脖子，且躯干短于腿。 | 32 |
| 躯干短于脖子，且躯干长于腿。 | 31 |
| 躯干短于脖子，且躯干长于尾巴。 | 25 |
| 躯干长于脖子，且脖子短于腿。 | 24 |
| 躯干短于脖子，且脖子长于腿。 | 21 |
| 头短于腿。 | 21 |
| 躯干长于脖子，且头长于腿。 | 20 |
| 躯干短于脖子，且躯干长于头。 | 16 |
| 躯干短于脖子，且躯干短于尾巴。 | 15 |
| 头长于腿。 | 14 |
| 躯干长于脖子。 | 14 |
| 躯干长于脖子，且头短于腿。 | 13 |
| 躯干短于脖子，且头长于腿。 | 13 |
| 躯干短于脖子、长于尾巴。 | 13 |
| 躯干长于脖子，且头短于尾巴。 | 12 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 躯干长于脖子，且躯干长于腿。 | 72 | body_ref | S1T241, S1T245, S1T246, S1T249, S1T255, S1T256, S1T264, S1T265 |
| 躯干短于脖子，且脖子长于尾巴。 | 69 | body_ref | S2T33, S2T41, S2T42, S2T44, S2T45, S2T174, S2T176, S2T177 |
| 躯干长于脖子，且躯干短于腿。 | 60 | body_ref | S1T247, S1T248, S1T257, S1T258, S1T259, S1T274, S1T277, S1T279 |
| 躯干短于脖子，且脖子短于尾巴。 | 38 | body_ref | S2T179, S2T180, S2T184, S2T188, S2T189, S2T196, S2T198, S2T201 |
| 躯干长于脖子，且脖子短于尾巴。 | 33 | body_ref | S2T34, S2T35, S2T36, S2T38, S2T39, S2T43, S2T46, S2T175 |
| 躯干短于脖子，且躯干短于腿。 | 32 | body_ref | S1T242, S1T243, S1T261, S1T273, S1T275, S1T276, S1T278, S1T283 |
| 躯干短于脖子，且躯干长于腿。 | 31 | body_ref | S1T244, S1T250, S1T251, S1T252, S1T253, S1T254, S1T260, S1T262 |
| 躯干短于脖子，且躯干长于尾巴。 | 25 | body_ref | S2T117, S3T72, S3T74, S3T77, S3T80, S3T85, S3T87, S3T96 |
| 躯干长于脖子，且脖子短于腿。 | 24 | body_ref | S2T5, S2T11, S2T15, S2T17, S2T18, S2T21, S2T23, S2T25 |
| 躯干短于脖子，且脖子长于腿。 | 21 | body_ref | S2T7, S2T9, S2T12, S2T13, S2T16, S2T20, S2T22, S2T136 |
| 躯干长于脖子，且头长于腿。 | 20 | body_ref | S1T155, S1T158, S1T161, S1T163, S1T167, S1T170, S1T171, S1T172 |
| 躯干短于脖子，且躯干长于头。 | 16 | body_ref | S1T227, S1T228, S1T229, S1T234, S1T236, S1T237, S1T239, S2T87 |
| 躯干短于脖子，且躯干短于尾巴。 | 15 | body_ref | S2T118, S3T70, S3T73, S3T76, S3T79, S3T81, S3T83, S3T89 |
| 躯干长于脖子。 | 14 | body_ref | S1T133, S1T134, S1T136, S1T137, S1T138, S1T139, S1T141, S1T142 |
| 躯干短于脖子、长于尾巴。 | 13 | body_ref | S3T153, S3T158, S3T167, S3T168, S3T171, S3T176, S3T177, S3T181 |
| 躯干短于脖子，且头长于腿。 | 13 | body_ref | S1T156, S1T162, S1T166, S1T180, S1T183, S1T187, S1T189, S1T192 |
| 躯干长于脖子，且头短于腿。 | 13 | body_ref | S1T157, S1T159, S1T160, S1T164, S1T165, S1T169, S1T177, S1T184 |
| 躯干短于脖子，且头短于腿。 | 12 | body_ref | S1T168, S1T173, S1T174, S1T175, S1T181, S1T182, S2T47, S2T50 |
| 躯干长于脖子和腿。 | 12 | body_ref | S3T143, S3T148, S3T160, S3T162, S3T164, S3T165, S3T170, S3T173 |
| 躯干长于脖子，且头短于尾巴。 | 12 | body_ref | S1T196, S1T197, S1T205, S1T206, S1T214, S1T222, S1T224, S2T67 |
| 躯干长于脖子，且脖子长于尾巴。 | 12 | body_ref | S2T37, S2T40, S2T206, S2T225, S2T228, S2T239, S2T241, S2T242 |
| 躯干长于脖子，且脖子长于腿。 | 11 | body_ref | S2T6, S2T8, S2T10, S2T14, S2T19, S2T135, S2T143, S2T148 |
| 躯干长于脖子，且头长于尾巴。 | 10 | body_ref | S1T194, S1T198, S1T199, S1T201, S1T202, S1T203, S1T208, S1T212 |
| 躯干长于脖子，且躯干长于头。 | 10 | body_ref | S1T230, S1T232, S1T233, S1T235, S1T238, S2T85, S2T86, S2T88 |
| 躯干短于脖子和尾巴。 | 9 | body_ref | S3T144, S3T147, S3T151, S3T154, S3T155, S3T157, S3T178, S3T180 |
| 躯干短于脖子，且头长于尾巴。 | 9 | body_ref | S1T193, S1T195, S1T204, S1T207, S1T223, S2T68, S2T69, S2T73 |
| 躯干长于脖子、短于腿。 | 9 | body_ref | S3T152, S3T156, S3T159, S3T161, S3T163, S3T166, S3T169, S3T172 |
| 躯干短于脖子。 | 8 | body_ref | S1T135, S1T140, S1T143, S1T146, S1T148, S1T149, S1T153, S1T154 |
| 躯干短于脖子，且脖子短于腿。 | 6 | body_ref | S2T24, S2T26, S2T134, S2T145, S2T154, S2T167 |
| 躯干短于脖子，且脖子长于头。 | 6 | body_ref | S2T2, S2T4, S2T123, S2T130, S2T132, S2T133 |
| 躯干短于脖子，且腿短于尾巴。 | 6 | body_ref | S1T213, S1T218, S2T79, S2T80, S2T83, S2T84 |
| 躯干短于脖子，且躯干短于头。 | 6 | body_ref | S1T226, S2T90, S2T92, S2T93, S2T100, S3T41 |
| 躯干长于脖子，且短于腿。 | 6 | body_ref | S3T137, S3T139, S3T141, S3T142, S3T145, S3T146 |
| 腿比躯干短。 | 5 | body_ref | S1T46, S1T47, S1T48, S1T49, S1T52 |
| 躯干短于脖子，且头短于尾巴。 | 5 | body_ref | S1T200, S1T221, S2T65, S2T66, S3T19 |
| 躯干长于脖子，且脖子短于头。 | 5 | body_ref | S2T122, S2T124, S2T127, S2T129, S2T131 |
| 躯干长于脖子，且腿短于尾巴。 | 5 | body_ref | S1T209, S1T211, S1T215, S2T78, S2T82 |
| 躯干长于脖子，且躯干短于头。 | 5 | body_ref | S1T231, S1T240, S2T89, S2T91, S2T95 |
| 朝向向左。 | 4 | other | S1T37, S1T39, S1T40, S1T42 |
| 有一个部位比躯干长。 | 3 | body_ref, count_abstract | S1T43, S1T44, S1T45 |
| 躯干长于脖子，且脖子长于头。 | 3 | body_ref | S2T1, S2T3, S2T128 |
| 躯干长于脖子，且腿长于尾巴。 | 3 | body_ref | S1T210, S1T216, S2T77 |
| 脖子长于躯干、长于腿。 | 2 | body_ref | S1T60, S1T61 |
| 腿比躯干长。 | 2 | body_ref | S1T50, S1T51 |
| 腿长于躯干、长于脖子。 | 2 | body_ref | S1T56, S1T57 |
| 躯干短于脖子，且脖子短于头。 | 2 | body_ref | S2T125, S2T126 |
| 躯干短于脖子，且腿长于尾巴。 | 2 | body_ref | S1T217, S2T81 |
| 躯干短于脖子，且长于尾巴。 | 2 | body_ref | S3T136, S3T140 |
| 躯干长于尾巴。 | 2 | body_ref | S1T131, S1T132 |
| 躯干长于脖子、长于腿。 | 2 | body_ref | S1T55, S1T64 |
| 躯干长于脖子，且躯干长于尾巴。 | 2 | body_ref | S2T119, S2T121 |
| 去判断于脖子和尾巴。 | 1 | other | S3T190 |
| 朝向向右。 | 1 | other | S1T41 |
| 脖子长于头、长于躯干，长于尾巴、长于腿。 | 1 | body_ref | S1T67 |
| 脖子长于腿、长于躯干。 | 1 | body_ref | S1T54 |
| 脖子长于躯干、长于头、长于腿。 | 1 | body_ref | S1T65 |
| 腿长于头、长于躯干，长于脖子。 | 1 | body_ref | S1T68 |
| 腿长于脖子、长于躯干。 | 1 | body_ref | S1T59 |
| 腿长于躯干、长于尾巴、长于头、长于脖子。 | 1 | body_ref | S1T53 |
| 躯干短于脖子长于尾巴。 | 1 | body_ref | S3T149 |
| 躯干短于脖子，且短于尾巴。 | 1 | body_ref | S3T138 |
| 躯干短于脖子，且躯干长短于尾巴。 | 1 | body_ref | S3T118 |
| 躯干短于，脖子和尾巴。 | 1 | body_ref | S3T179 |
| 躯干短脖子、长于尾巴。 | 1 | body_ref | S3T182 |
| 躯干短脖子和尾巴。 | 1 | body_ref | S3T150 |
| 躯干长于脖子、长于腿、长于头。 | 1 | body_ref | S1T66 |
| 躯干长于脖子，且脖躯干短于腿。 | 1 | body_ref | S2T308 |
| 躯干长于脖子，且躯干短于尾巴。 | 1 | body_ref | S3T103 |
| 躯干长于脖子，且躯干长于尾盘。 | 1 | body_ref | S2T120 |
| 躯干长于脖子，躯干长于腿。 | 1 | body_ref | S3T109 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 躯干长于脖子，且躯干长于腿。 | 64 | 0.000 | body_ref:脖子 > 0.50; body_ref:腿 > 0.50 | S1T241, S1T245, S1T246, S1T249, S1T255, S1T264, S1T265, S1T266 |
| 躯干长于脖子，且躯干短于腿。 | 57 | 0.000 | body_ref:脖子 > 0.50; body_ref:腿 < 0.50 | S1T247, S1T248, S1T258, S1T259, S1T274, S1T277, S1T279, S1T280 |
| 躯干短于脖子，且躯干短于腿。 | 32 | 0.000 | body_ref:脖子 < 0.50; body_ref:腿 < 0.50 | S1T242, S1T243, S1T261, S1T273, S1T275, S1T276, S1T278, S1T283 |
| 躯干短于脖子，且躯干长于腿。 | 26 | 0.000 | body_ref:脖子 < 0.50; body_ref:腿 > 0.50 | S1T251, S1T253, S1T254, S1T260, S1T262, S1T267, S1T269, S1T271 |
| 躯干短于脖子，且躯干长于尾巴。 | 23 | 0.000 | body_ref:脖子 < 0.50; body_ref:尾巴 > 0.50 | S2T117, S3T72, S3T77, S3T80, S3T85, S3T96, S3T98, S3T100 |
| 躯干短于脖子，且脖子短于尾巴。 | 15 | 0.000 | body_ref:脖子 < 0.50; comparison:脖子 < 尾巴 | S2T180, S2T188, S2T189, S2T233, S2T237, S2T248, S2T275, S2T278 |
| 躯干短于脖子，且躯干短于尾巴。 | 14 | 0.000 | body_ref:脖子 < 0.50; body_ref:尾巴 < 0.50 | S2T118, S3T70, S3T73, S3T76, S3T79, S3T81, S3T89, S3T90 |
| 躯干短于脖子，且躯干长于头。 | 13 | 0.000 | body_ref:脖子 < 0.50; body_ref:头 > 0.50 | S1T227, S1T234, S1T236, S1T239, S2T87, S2T96, S2T97, S2T98 |
| 躯干长于脖子。 | 13 | 0.000 | body_ref:脖子 > 0.50 | S1T133, S1T134, S1T136, S1T137, S1T138, S1T139, S1T142, S1T144 |
| 躯干长于脖子和腿。 | 12 | 0.000 | body_ref:脖子 > 0.50; body_ref:腿 > 0.50 | S3T143, S3T148, S3T160, S3T162, S3T164, S3T165, S3T170, S3T173 |
| 躯干短于脖子和尾巴。 | 9 | 0.000 | body_ref:脖子 < 0.50; body_ref:尾巴 < 0.50 | S3T144, S3T147, S3T151, S3T154, S3T155, S3T157, S3T178, S3T180 |
| 躯干短于脖子。 | 8 | 0.000 | body_ref:脖子 < 0.50 | S1T135, S1T140, S1T143, S1T146, S1T148, S1T149, S1T153, S1T154 |
| 躯干长于脖子，且躯干长于头。 | 8 | 0.000 | body_ref:脖子 > 0.50; body_ref:头 > 0.50 | S1T232, S1T233, S1T235, S1T238, S2T85, S2T86, S2T94, S2T99 |
| 躯干短于脖子，且躯干短于头。 | 6 | 0.000 | body_ref:脖子 < 0.50; body_ref:头 < 0.50 | S1T226, S2T90, S2T92, S2T93, S2T100, S3T41 |
| 躯干长于脖子，且短于腿。 | 6 | 0.000 | body_ref:脖子 > 0.50 | S3T137, S3T139, S3T141, S3T142, S3T145, S3T146 |
| 躯干长于脖子，且躯干短于头。 | 4 | 0.000 | body_ref:脖子 > 0.50; body_ref:头 < 0.50 | S1T231, S2T89, S2T91, S2T95 |
| 躯干短于脖子，且头长于尾巴。 | 2 | 0.000 | body_ref:脖子 < 0.50; comparison:头 > 尾巴 | S1T195, S2T68 |
| 躯干短于脖子，且长于尾巴。 | 2 | 0.000 | body_ref:脖子 < 0.50 | S3T136, S3T140 |
| 躯干长于尾巴。 | 2 | 0.000 | body_ref:尾巴 > 0.50 | S1T131, S1T132 |
| 躯干长于脖子，且脖子短于尾巴。 | 2 | 0.000 | body_ref:脖子 > 0.50; comparison:脖子 < 尾巴 | S2T252, S2T272 |
| 躯干长于脖子，且躯干长于尾巴。 | 2 | 0.000 | body_ref:脖子 > 0.50; body_ref:尾巴 > 0.50 | S2T119, S2T121 |
| 躯干长于脖子、短于腿。 | 2 | 0.250 | body_ref:脖子 > 0.50; body_ref:腿 > 0.50; absolute_long:脖子 > 0.50 | S3T159, S3T186 |
| 头长于腿，脖子短于尾巴。 | 1 | 0.000 | comparison:头 > 腿; comparison:脖子 < 尾巴 | S1T119 |
| 腿比躯干短。 | 1 | 0.000 | body_ref:腿 < 0.50 | S1T52 |
| 躯干短于脖子，且头短于尾巴。 | 1 | 0.000 | body_ref:脖子 < 0.50; comparison:头 < 尾巴 | S3T19 |
| 躯干短于脖子，且头长于腿。 | 1 | 0.000 | body_ref:脖子 < 0.50; comparison:头 > 腿 | S2T54 |
| 躯干短于脖子，且短于尾巴。 | 1 | 0.000 | body_ref:脖子 < 0.50 | S3T138 |
| 躯干短于脖子，且脖子短于腿。 | 1 | 0.000 | body_ref:脖子 < 0.50; comparison:脖子 < 腿 | S2T134 |
| 躯干短于脖子，且腿短于尾巴。 | 1 | 0.000 | body_ref:脖子 < 0.50; comparison:腿 < 尾巴 | S1T218 |
| 躯干短于脖子，且躯干长短于尾巴。 | 1 | 0.000 | body_ref:脖子 < 0.50; body_ref:尾巴 < 0.50 | S3T118 |
| 躯干短脖子和尾巴。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:尾巴 < 0.50 | S3T150 |
| 躯干长于脖子，且头短于尾巴。 | 1 | 0.000 | body_ref:脖子 > 0.50; comparison:头 < 尾巴 | S1T214 |
| 躯干长于脖子，且头长于尾巴。 | 1 | 0.000 | body_ref:脖子 > 0.50; comparison:头 > 尾巴 | S1T212 |
| 躯干长于脖子，且脖子长于腿。 | 1 | 0.000 | body_ref:脖子 > 0.50; comparison:脖子 > 腿 | S2T14 |
| 躯干长于脖子，且腿短于尾巴。 | 1 | 0.000 | body_ref:脖子 > 0.50; comparison:腿 < 尾巴 | S2T78 |
| 躯干长于脖子，且躯干短于尾巴。 | 1 | 0.000 | body_ref:脖子 > 0.50; body_ref:尾巴 < 0.50 | S3T103 |
| 躯干长于脖子，且躯干长于尾盘。 | 1 | 0.000 | body_ref:脖子 > 0.50; body_ref:尾巴 > 0.50 | S2T120 |
| 躯干长于脖子，躯干长于腿。 | 1 | 0.000 | body_ref:脖子 > 0.50; body_ref:腿 > 0.50 | S3T109 |
| 躯干长于脖子、长于腿。 | 1 | 0.250 | body_ref:脖子 > 0.50; body_ref:腿 > 0.50; absolute_long:脖子 > 0.50 | S1T55 |
| 脖子长于腿，且头长约尾巴。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T8 |
| 躯干长于脖子，且脖躯干短于腿。 | 1 | 0.333 | body_ref:脖子 > 0.50; body_ref:腿 < 0.50 | S2T308 |

### S315

- trial 数: 704; 非空文本: 703; fidelity 可评分率: 0.997; 平均 fidelity: 0.893; 完全忠实率: 0.797; 低 fidelity 率: 0.037.
- 旧版 region 覆盖率: 0.997; 旧版 region 有未处理片段率: 0.001.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 701 | 0.996 |
| empty | 1 | 0.001 |
| equality | 1 | 0.001 |
| other | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，脖子短。 | 100 |
| 腿长，脖子长。 | 83 |
| 腿短，尾巴长。 | 70 |
| 腿短，尾巴短。 | 32 |
| 腿长，头短。 | 29 |
| 腿长。 | 19 |
| 腿长，脖子长，头短。 | 15 |
| 脖子长，尾巴短。 | 15 |
| 脖子长。 | 14 |
| 尾巴长，腿短。 | 14 |
| 腿短，尾巴长，脖子长。 | 13 |
| 腿短。 | 13 |
| 尾巴短。 | 12 |
| 腿长，头长。 | 11 |
| 腿短，脖子长。 | 11 |
| 腿短，尾巴长，头长。 | 10 |
| 脖子长，头长。 | 10 |
| 头长，尾巴长。 | 9 |
| 脖子短。 | 9 |
| 头长。 | 8 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 各部位平均。 | 1 | equality | S1T44 |
| 脖。 | 1 | other | S1T95 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴短。 | 3 | 0.000 | absolute_short:尾巴 < 0.50 | S1T276, S1T293, S1T304 |
| 脖子长，尾巴长。 | 2 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T63, S1T316 |
| 腿短。 | 2 | 0.000 | absolute_short:腿 < 0.50 | S1T31, S1T239 |
| 脖子长，腿长，尾巴短。 | 2 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S1T214, S1T215 |
| 各部位平均。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T44 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T24 |
| 头长，尾巴长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T28 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T111 |
| 脖子短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S1T308 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T90 |
| 脖子长，腿长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S1T262 |
| 腿和尾巴长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T97 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T101 |
| 腿长，尾巴长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T269 |
| 腿长，脖子长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S1T140 |
| 腿长，头长，脖子长，头短。 | 1 | 0.250 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T230 |
| 尾巴长，脖子长，腿短。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S2T4 |
| 腿短，尾巴长，头长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50 | S2T28 |
| 腿短，尾巴长，脖子长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S2T145 |
| 腿长，头短，脖子长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S2T136 |
| 腿长，脖子短，头长。 | 1 | 0.333 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S2T86 |

### S316

- trial 数: 384; 非空文本: 383; fidelity 可评分率: 0.995; 平均 fidelity: 0.943; 完全忠实率: 0.854; 低 fidelity 率: 0.010.
- 旧版 region 覆盖率: 0.995; 旧版 region 有未处理片段率: 0.135.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 383 | 0.997 |
| comparison | 111 | 0.289 |
| superlative | 67 | 0.174 |
| negation | 33 | 0.086 |
| equality | 10 | 0.026 |
| meta | 1 | 0.003 |
| empty | 1 | 0.003 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，尾巴短。 | 61 |
| 腿长，尾巴长。 | 47 |
| 腿短，脖子短。 | 24 |
| 腿短，脖子长。 | 23 |
| 腿短，头长于脖子。 | 7 |
| 腿短，不是脖子长，头短。 | 6 |
| 腿长，尾巴也长。 | 5 |
| 腿短，且是最短，头比脖子长。 | 4 |
| 腿短，且是最短，头比脖子短。 | 3 |
| 腿很长。 | 3 |
| 尾巴和腿很长。 | 3 |
| 腿长，尾巴也很长。 | 3 |
| 腿短，头比腿短。 | 3 |
| 腿短，且脖子是头、脖子、尾巴里最短。 | 3 |
| 腿短，脖子长，头短。 | 3 |
| 腿短，头比腿长。 | 3 |
| 腿短，尾巴不比脖子长。 | 2 |
| 腿短，头也短。 | 2 |
| 脖子比腿长，腿短。 | 2 |
| 腿短，头比脖子长。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿短，不是脖子长，头短。 | 6 | negation | S1T279, S1T280, S1T281, S1T283, S1T286, S1T287 |
| 腿短，且脖子不是头、脖子、尾巴里最短。 | 2 | negation | S1T137, S1T139 |
| 头很长，尾巴不长，脖子和腿也不长。 | 1 | negation | S1T34 |
| 尾巴很长，脖子不长。 | 1 | negation | S1T30 |
| 脖子不是最短，脖子比尾巴长，尾巴是头、脖子、尾巴里最短。 | 1 | negation | S1T153 |
| 脖子和尾巴很长，腿没有那么长，且头短于脖子。 | 1 | negation | S1T44 |
| 脖子比腿长，腿短，尾巴和腿差不多长，头也不短。 | 1 | equality, negation | S1T190 |
| 腿不是很短，脖子和尾巴比较长，比腿略长一些。 | 1 | negation | S1T69 |
| 腿不是很长，头、脖子、尾巴长度相近。 | 1 | equality, negation | S1T103 |
| 腿不是很长，脖子最长。 | 1 | negation | S1T101 |
| 腿不长，头比脖子长，尾巴和脖子差不多长。 | 1 | equality, negation | S1T172 |
| 腿很长，尾巴不长。 | 1 | negation | S1T45 |
| 腿比较短，脖子不是最短。 | 1 | negation | S1T109 |
| 腿没有明显长于脖子和尾巴，且头和脖子长度相似。 | 1 | equality, negation | S1T51 |
| 腿没有那么长，腿的长度已经很长了，还是。 | 1 | negation | S1T61 |
| 腿短不是最短，脖子和尾巴比脖子长一点，头是最短。 | 1 | negation | S1T258 |
| 腿短，且不是最短，脖子不是最长。 | 1 | negation | S1T257 |
| 腿短，且不是最短，脖子最长。 | 1 | negation | S1T271 |
| 腿短，且脖子不是另外三个里最短。 | 1 | negation | S1T129 |
| 腿短，且脖子不是最短。 | 1 | negation | S1T136 |
| 腿短，头和尾巴不知道谁长，脖子最长。 | 1 | meta | S1T125 |
| 腿短，头比腿长很多，脖子和腿差不多长，尾巴比腿长。 | 1 | equality | S1T233 |
| 腿短，是最短，脖子不是最长。 | 1 | negation | S1T268 |
| 腿短，脖子不是另外三个里最短。 | 1 | negation | S1T117 |
| 腿短，脖子不是头、脖子、尾巴里最短。 | 1 | negation | S1T112 |
| 腿短，脖子不是头、脖子和尾巴里最短。 | 1 | negation | S1T141 |
| 腿短，脖子与头相近，脖子肯定不是最短。 | 1 | equality, negation | S1T152 |
| 腿短，脖子和尾巴几乎一样长，远长于头和腿。 | 1 | equality | S1T66 |
| 腿短，脖子和尾巴很长，头、尾巴和头和脖子长度相似。 | 1 | equality | S1T50 |
| 腿短，腿不是最短。 | 1 | negation | S1T236 |
| 腿短，腿不是最短，头比腿短。 | 1 | negation | S1T244 |
| 腿短，腿是最短，头比脖子短，脖子和尾巴差不多长。 | 1 | equality | S1T239 |
| 腿短，腿是最短，脖子比头长，尾巴和头差不多。 | 1 | equality | S1T162 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿不是很短，脖子和尾巴比较长，比腿略长一些。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T69 |
| 腿没有明显长于脖子和尾巴，且头和脖子长度相似。 | 1 | 0.000 | equality_range:头+脖子 = | S1T51 |
| 尾巴和脖子很长，头更长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S1T41 |
| 腿短，不是脖子长，头短。 | 1 | 0.333 | absolute_short:脖子 < 0.50; absolute_short:头 < 0.50 | S1T287 |

### S317

- trial 数: 1024; 非空文本: 1024; fidelity 可评分率: 0.940; 平均 fidelity: 0.847; 完全忠实率: 0.725; 低 fidelity 率: 0.081.
- 旧版 region 覆盖率: 0.940; 旧版 region 有未处理片段率: 0.061.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| comparison | 785 | 0.767 |
| direct_absolute | 619 | 0.604 |
| body_ref | 400 | 0.391 |
| equality | 163 | 0.159 |
| superlative | 90 | 0.088 |
| negation | 37 | 0.036 |
| count_abstract | 10 | 0.010 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴短于3/4躯干，腿短于3/4躯干。 | 67 |
| 尾巴长于3/4躯干，脖子短于3/4躯干。 | 48 |
| 尾巴短于3/4躯干，腿长于3/4躯干。 | 46 |
| 尾巴小于脖子，头大于腿。 | 45 |
| 尾巴小于脖子。 | 39 |
| 尾巴长于3/4躯干，脖子长于3/4躯干。 | 38 |
| 尾巴小于脖子，头小于腿。 | 35 |
| 尾巴大于脖子，头大于腿。 | 33 |
| 尾巴等于脖子。 | 32 |
| 尾巴大于脖子，头小于腿。 | 29 |
| 尾巴大于脖子。 | 26 |
| 尾巴等于脖子，头等于腿。 | 18 |
| 尾巴小于脖子，头等于腿。 | 17 |
| 尾巴短于躯干。 | 17 |
| 头等于腿。 | 17 |
| 尾巴短于3/4躯干。 | 16 |
| 尾巴大于脖子，头等于腿。 | 15 |
| 脖子短于躯干。 | 13 |
| 尾巴等于脖子，头大于腿。 | 12 |
| 头短于躯干。 | 12 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 尾巴短于3/4躯干，腿短于3/4躯干。 | 67 | body_ref | S3T168, S3T169, S3T170, S3T171, S3T172, S3T173, S3T175, S3T179 |
| 尾巴长于3/4躯干，脖子短于3/4躯干。 | 48 | body_ref | S3T189, S3T209, S3T213, S3T214, S3T217, S3T218, S3T219, S3T238 |
| 尾巴短于3/4躯干，腿长于3/4躯干。 | 46 | body_ref | S3T177, S3T178, S3T185, S3T194, S3T200, S3T205, S3T206, S3T207 |
| 尾巴长于3/4躯干，脖子长于3/4躯干。 | 38 | body_ref | S3T157, S3T158, S3T164, S3T188, S3T191, S3T192, S3T195, S3T197 |
| 尾巴等于脖子。 | 32 | equality | S1T28, S1T89, S1T90, S1T149, S1T151, S1T154, S1T159, S1T164 |
| 尾巴等于脖子，头等于腿。 | 18 | equality | S1T111, S1T117, S1T152, S1T157, S1T169, S1T190, S1T192, S1T195 |
| 头等于腿。 | 17 | equality | S1T180, S1T182, S1T201, S1T203, S1T209, S1T210, S1T213, S1T214 |
| 尾巴小于脖子，头等于腿。 | 17 | equality | S1T79, S1T93, S1T112, S1T119, S1T124, S1T148, S1T174, S1T198 |
| 尾巴短于躯干。 | 17 | body_ref | S2T211, S2T212, S2T213, S2T214, S2T215, S2T216, S2T217, S3T2 |
| 尾巴短于3/4躯干。 | 16 | body_ref | S3T129, S3T131, S3T133, S3T134, S3T135, S3T136, S3T137, S3T140 |
| 尾巴大于脖子，头等于腿。 | 15 | equality | S1T70, S1T77, S1T92, S1T107, S1T118, S1T122, S1T219, S1T247 |
| 脖子短于躯干。 | 13 | body_ref | S2T56, S2T57, S2T59, S2T60, S2T219, S2T220, S2T221, S3T20 |
| 头短于躯干。 | 12 | body_ref | S2T98, S2T100, S2T104, S2T106, S3T9, S3T10, S3T11, S3T25 |
| 尾巴等于脖子，头大于腿。 | 12 | equality | S1T95, S1T116, S1T126, S1T128, S1T139, S1T146, S1T245, S1T252 |
| 尾巴长于躯干。 | 12 | body_ref | S2T209, S2T210, S2T218, S3T1, S3T3, S3T6, S3T32, S3T86 |
| 头长于躯干。 | 11 | body_ref | S2T96, S2T97, S2T99, S2T101, S2T102, S2T103, S2T105, S2T107 |
| 腿长于躯干。 | 10 | body_ref | S2T84, S2T152, S2T153, S2T158, S2T159, S3T15, S3T18, S3T19 |
| 脖子长于躯干。 | 9 | body_ref | S2T222, S2T237, S3T13, S3T22, S3T39, S3T40, S3T89, S3T102 |
| 腿短于躯干。 | 9 | body_ref | S2T53, S2T54, S2T151, S3T14, S3T16, S3T17, S3T37, S3T38 |
| 尾巴小于脖子，头小于等于腿。 | 7 | equality | S1T64, S1T125, S1T130, S1T133, S1T137, S1T156, S1T160 |
| 尾巴短于3/4躯干，脖子长于3/4躯干。 | 6 | body_ref | S3T155, S3T159, S3T161, S3T163, S3T166, S3T210 |
| 尾巴短于躯干，脖子长于躯干。 | 6 | body_ref | S3T70, S3T74, S3T75, S3T79, S3T80, S3T81 |
| 尾巴长于3/4躯干。 | 6 | body_ref | S3T130, S3T132, S3T138, S3T139, S3T144, S3T147 |
| 尾巴短于躯干，脖子短于躯干。 | 5 | body_ref | S3T71, S3T76, S3T78, S3T83, S3T85 |
| 尾巴等于脖子，头小于腿。 | 5 | equality | S1T271, S1T272, S1T283, S2T35, S2T36 |
| 尾巴长于3/4躯干，腿短于3/4躯干。 | 5 | body_ref | S3T174, S3T176, S3T183, S3T226, S3T227 |
| 尾巴长于3/4躯干，腿长于3/4躯干。 | 5 | body_ref | S3T193, S3T247, S3T262, S3T263, S4T41 |
| 尾巴长于躯干，脖子短于躯干。 | 5 | body_ref | S2T257, S3T66, S3T69, S3T77, S3T82 |
| 脖子不是最长的部位。 | 5 | negation | S2T176, S2T177, S2T178, S2T179, S2T181 |
| 腿不是最短的部位。 | 5 | negation | S2T185, S2T186, S2T187, S2T189, S2T190 |
| 尾巴和脖子都短于躯干。 | 4 | body_ref | S2T253, S2T254, S2T255, S2T256 |
| 尾巴短于3/4躯干，脖子短于3/4躯干。 | 4 | body_ref | S3T156, S3T162, S3T165, S3T167 |
| 没有部位达到最长或最短长度。 | 4 | negation | S2T111, S2T112, S2T315, S2T318 |
| 脖子加头没有尾巴加腿长。 | 4 | negation | S2T119, S2T120, S2T121, S2T122 |
| 腿没有达到最大长度。 | 4 | negation | S2T71, S2T154, S2T155, S2T156 |
| 尾巴大于躯干。 | 3 | body_ref | S3T60, S3T61, S3T62 |
| 尾巴小于躯干。 | 3 | body_ref | S3T59, S3T63, S3T64 |
| 尾巴长于躯干，脖子长于躯干。 | 3 | body_ref | S3T65, S3T72, S3T73 |
| 两个以上部位比躯干的一半长。 | 2 | body_ref | S2T62, S2T63 |
| 四个部位一样长。 | 2 | equality | S2T310, S2T311 |
| 小于两个部位一样长。 | 2 | equality, count_abstract | S2T313, S2T314 |
| 尾巴不短于脖子。 | 2 | negation | S2T67, S2T68 |
| 尾巴小于等于脖子，头大于腿。 | 2 | equality | S1T101, S1T138 |
| 尾巴短于3/4躯干，脖子和腿长于3/4躯干。 | 2 | body_ref | S3T153, S3T241 |
| 尾巴短于3/4躯干，腿大于3/4躯干。 | 2 | body_ref | S3T309, S4T51 |
| 有两个部位达到最长长度。 | 2 | count_abstract | S2T286, S2T288 |
| 没有部位达到最长长度。 | 2 | negation | S2T83, S2T94 |
| 脖子等于头。 | 2 | equality | S2T147, S2T148 |
| 脖子等于躯干。 | 2 | equality, body_ref | S2T58, S2T223 |
| 腿和尾巴长度相等。 | 2 | equality | S1T12, S2T73 |
| 腿没有达到最长或最短。 | 2 | negation | S2T76, S2T77 |
| 腿短于3/4躯干，尾巴短于3/4躯干。 | 2 | body_ref | S3T186, S3T203 |
| 一个部位达到最长长度。 | 1 | count_abstract | S2T89 |
| 只有头比躯干长。 | 1 | body_ref | S1T14 |
| 大于两个部位一样长。 | 1 | equality, count_abstract | S2T312 |
| 头不是最短的部位。 | 1 | negation | S2T192 |
| 头和腿一样长。 | 1 | equality | S2T247 |
| 头小于尾巴、等于腿、小于脖子。 | 1 | equality | S1T11 |
| 头比躯干长。 | 1 | body_ref | S2T55 |
| 头等于尾巴、大于脖子、大于腿。 | 1 | equality | S1T3 |
| 头等于脖子、小于腿、小于尾巴。 | 1 | equality | S1T1 |
| 头等于脖子，尾巴等于腿。 | 1 | equality | S1T73 |
| 尾巴、脖子短于3/4躯干，腿长于3/4躯干。 | 1 | body_ref | S3T154 |
| 尾巴、腿、脖子短于3/4躯干。 | 1 | body_ref | S3T151 |
| 尾巴、腿、脖子长于3/4躯干。 | 1 | body_ref | S3T152 |
| 尾巴和脖子一样长。 | 1 | equality | S2T236 |
| 尾巴和脖子都长于躯干。 | 1 | body_ref | S2T258 |
| 尾巴大于等于脖子，头大于腿。 | 1 | equality | S1T104 |
| 尾巴大于等于脖子，头小于等于腿。 | 1 | equality | S1T145 |
| 尾巴大于脖子，头大于等于腿。 | 1 | equality | S1T100 |
| 尾巴大于脖子，头小于等于腿。 | 1 | equality | S1T131 |
| 尾巴大约3/4躯干，脖子大约3/4躯干。 | 1 | body_ref | S3T294 |
| 尾巴小于头、等于脖子、等于腿。 | 1 | equality | S1T7 |
| 尾巴小于脖子、等于头、小于腿。 | 1 | equality | S1T19 |
| 尾巴小于脖子，头大于等于腿。 | 1 | equality | S1T98 |
| 尾巴小于腿、等于脖子、小于头。 | 1 | equality | S1T10 |
| 尾巴短于3/4躯干，脖子大小于3/4躯干。 | 1 | body_ref | S3T160 |
| 尾巴短于3/4躯干，腿大小于3/4躯干。 | 1 | body_ref | S3T198 |
| 尾巴短于躯干，脖子短于躯干，腿比躯干长。 | 1 | body_ref | S3T84 |
| 尾巴等于脖子、等于头、等于腿。 | 1 | equality | S1T217 |
| 尾巴等于腿、小于头、小于脖子。 | 1 | equality | S1T4 |
| 尾巴等于腿。 | 1 | equality | S1T278 |
| 尾巴长于躯干，腿短于躯干。 | 1 | body_ref | S3T67 |
| 尾巴长于躯干，腿长于躯干。 | 1 | body_ref | S3T68 |
| 有三个部位和躯干一样长。 | 1 | equality, body_ref, count_abstract | S1T15 |
| 有两个部位比躯干长。 | 1 | body_ref, count_abstract | S1T18 |
| 没有两个部位达到最长长度。 | 1 | count_abstract, negation | S2T287 |
| 没有出现最小长度，尾巴小于脖子。 | 1 | negation | S2T91 |
| 没有出现最长长度。 | 1 | negation | S2T90 |
| 没有达到最长长度。 | 1 | negation | S2T95 |
| 没有部位是最大或最小长度。 | 1 | negation | S2T80 |
| 没有部位是最长或者最短。 | 1 | negation | S2T78 |
| 没有部位是最长的。 | 1 | negation | S2T108 |
| 没有部位达到最长或最短的长度。 | 1 | negation | S2T114 |
| 脖子、腿长于躯干。 | 1 | body_ref | S3T99 |
| 脖子和腿短于躯干。 | 1 | body_ref | S3T100 |
| 脖子和腿短于躯干，头长于躯干。 | 1 | body_ref | S3T101 |
| 脖子小于头、等于腿、小于尾巴。 | 1 | equality | S1T6 |
| 脖子小于等于尾巴。 | 1 | equality | S2T65 |
| 脖子比躯干长，其余三个部位比躯干短。 | 1 | body_ref, count_abstract | S1T13 |
| 脖子短于躯干，腿短于躯干。 | 1 | body_ref | S3T90 |
| 脖子等于头等于尾巴。 | 1 | equality | S1T223 |
| 脖子等于腿。 | 1 | equality | S1T172 |
| 脖子等于腿，头等于尾巴。 | 1 | equality | S1T171 |
| 脖子长于躯干，头变长。 | 1 | body_ref | S3T104 |
| 脖子长于躯干，腿短于躯干。 | 1 | body_ref | S3T91 |
| 脖子长于躯干，腿长于躯干。 | 1 | body_ref | S3T92 |
| 脖子长于躯干，腿长于躯干，尾巴短于躯干。 | 1 | body_ref | S3T93 |
| 腿小于脖子、等于头、小于尾巴。 | 1 | equality | S1T16 |
| 腿等于尾巴、小于脖子、小于头。 | 1 | equality | S1T17 |
| 腿等于躯干。 | 1 | equality, body_ref | S2T157 |
| 腿长于3/4躯干，尾巴短于3/4躯干。 | 1 | body_ref | S3T204 |
| 腿长于3/4躯干，脖子短于3/4躯干。 | 1 | body_ref | S3T305 |
| 腿长于3/4躯干，脖子长于3/4躯干。 | 1 | body_ref | S3T306 |
| 达到最长或最短长度的部位数小于等于一。 | 1 | equality | S2T320 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴等于脖子。 | 10 | 0.000 | comparison:尾巴 = 脖子 | S1T154, S1T186, S1T189, S1T197, S1T211, S1T220, S1T243, S1T293 |
| 头短于躯干。 | 8 | 0.000 | body_ref:头 < 0.50 | S2T98, S2T100, S2T104, S2T106, S3T9, S3T10, S3T25, S3T28 |
| 头等于腿。 | 7 | 0.000 | comparison:头 = 腿 | S1T182, S1T203, S1T209, S1T210, S1T214, S1T232, S2T250 |
| 尾巴短于3/4躯干。 | 7 | 0.000 | body_ref:尾巴 < 0.38 | S3T131, S3T135, S3T143, S3T145, S3T148, S3T149, S3T150 |
| 头长于躯干。 | 6 | 0.000 | body_ref:头 > 0.50 | S2T96, S2T97, S2T102, S2T103, S2T107, S3T8 |
| 尾巴短于躯干。 | 6 | 0.000 | body_ref:尾巴 < 0.50 | S2T211, S3T5, S3T34, S3T97, S3T98, S3T109 |
| 尾巴短于3/4躯干，腿短于3/4躯干。 | 4 | 0.000 | body_ref:尾巴 < 0.38; body_ref:腿 < 0.38 | S3T173, S3T201, S3T202, S3T276 |
| 脖子短于躯干。 | 4 | 0.000 | body_ref:脖子 < 0.50 | S2T56, S2T221, S3T21, S3T41 |
| 腿短。 | 3 | 0.000 | absolute_short:腿 < 0.50 | S3T122, S3T124, S3T126 |
| 四个部位一样长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T310, S2T311 |
| 脖子等于躯干。 | 2 | 0.000 | body_ref:脖子 = 0.50 | S2T58, S2T223 |
| 腿比脖子短。 | 2 | 0.000 | comparison:腿 < 脖子 | S2T127, S2T128 |
| 头比脖子长。 | 1 | 0.000 | comparison:头 > 脖子 | S2T205 |
| 头比躯干长。 | 1 | 0.000 | body_ref:头 > 0.50 | S2T55 |
| 头等于脖子，尾巴等于腿。 | 1 | 0.000 | comparison:头 = 脖子; comparison:尾巴 = 腿 | S1T73 |
| 尾巴小于头。 | 1 | 0.000 | comparison:尾巴 < 头 | S2T292 |
| 尾巴小于躯干。 | 1 | 0.000 | body_ref:尾巴 < 0.50 | S3T59 |
| 尾巴最短。 | 1 | 0.000 | superlative:尾巴 < 脖子; superlative:尾巴 < 头; superlative:尾巴 < 腿 | S2T164 |
| 尾巴等于脖子，头大于腿。 | 1 | 0.000 | comparison:尾巴 = 脖子; comparison:头 > 腿 | S1T139 |
| 尾巴等于脖子，头等于腿。 | 1 | 0.000 | comparison:尾巴 = 脖子; comparison:头 = 腿 | S1T274 |
| 尾巴等于腿。 | 1 | 0.000 | comparison:尾巴 = 腿 | S1T278 |
| 脖子加头比尾巴加腿长。 | 1 | 0.000 | group_sum:脖子+头 > 尾巴+腿; comparison:头 > 尾巴 | S2T118 |
| 脖子小于头。 | 1 | 0.000 | comparison:脖子 < 头 | S2T146 |
| 脖子小于等于尾巴。 | 1 | 0.000 | comparison:脖子 < 尾巴 | S2T65 |
| 脖子小于腿。 | 1 | 0.000 | comparison:脖子 < 腿 | S2T149 |
| 脖子短，尾巴长。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_long:尾巴 > 0.50 | S2T64 |
| 腿和尾巴长度相等。 | 1 | 0.000 | equality_range:腿+尾巴 = | S2T73 |
| 腿等于躯干。 | 1 | 0.000 | body_ref:腿 = 0.50 | S2T157 |
| 尾巴小于头、等于脖子、等于腿。 | 1 | 0.200 | chained_comparison:尾巴 = 脖子; chained_comparison:尾巴 = 腿; comparison:尾巴+头 = 脖子; comparison:尾巴+头+脖子 = 腿 | S1T7 |
| 尾巴等于脖子、等于头、等于腿。 | 1 | 0.200 | comparison:尾巴 = 脖子; chained_comparison:尾巴 = 腿; comparison:尾巴+脖子 = 头; comparison:尾巴+脖子+头 = 腿 | S1T217 |
| 脖子和腿短于躯干，头长于躯干。 | 1 | 0.333 | body_ref:脖子 < 0.50; body_ref:腿 < 0.50 | S3T101 |
| 脖子是最长的部位。 | 1 | 0.333 | superlative:脖子 > 腿; superlative:脖子 > 尾巴 | S2T180 |
| 腿最短。 | 1 | 0.333 | superlative:腿 < 脖子; superlative:腿 < 尾巴 | S2T169 |
| 尾巴小于头、小于脖子、小于腿。 | 1 | 0.400 | comparison:尾巴 < 头; chained_comparison:尾巴 < 脖子; comparison:尾巴+头 < 脖子 | S1T2 |

### S318

- trial 数: 576; 非空文本: 575; fidelity 可评分率: 0.997; 平均 fidelity: 0.889; 完全忠实率: 0.747; 低 fidelity 率: 0.024.
- 旧版 region 覆盖率: 0.997; 旧版 region 有未处理片段率: 0.030.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 574 | 0.997 |
| comparison | 264 | 0.458 |
| body_ref | 122 | 0.212 |
| superlative | 30 | 0.052 |
| equality | 18 | 0.031 |
| negation | 15 | 0.026 |
| ranking | 7 | 0.012 |
| meta | 2 | 0.003 |
| empty | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴短，脖子长。 | 28 |
| 尾巴比躯干短，脖子比躯干长。 | 25 |
| 尾巴长，腿短。 | 21 |
| 尾巴比躯干短，脖子比躯干短。 | 20 |
| 尾巴比躯干长，腿比躯干短。 | 15 |
| 尾巴长，腿长。 | 14 |
| 尾巴短，脖子短。 | 12 |
| 尾巴比躯干长，腿比躯干长。 | 11 |
| 脖子很长。 | 9 |
| 头很长。 | 8 |
| 尾巴比腿短，头比脖子长。 | 8 |
| 尾巴长。 | 7 |
| 脖子长。 | 6 |
| 尾巴比腿长，头比脖子长。 | 6 |
| 头和尾巴长。 | 5 |
| 尾巴比躯干长，腿比较长。 | 5 |
| 尾巴比腿短，脖子长。 | 5 |
| 尾巴比腿长，头比脖子短。 | 5 |
| 只有头短。 | 5 |
| 尾巴比躯干短，头很长。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 尾巴比躯干短，脖子比躯干长。 | 25 | body_ref | S2T109, S2T114, S2T115, S2T117, S2T119, S2T120, S2T123, S2T125 |
| 尾巴比躯干短，脖子比躯干短。 | 20 | body_ref | S2T124, S2T135, S2T139, S2T140, S2T144, S2T145, S2T146, S2T150 |
| 尾巴比躯干长，腿比躯干短。 | 15 | body_ref | S2T91, S2T121, S2T132, S2T133, S2T141, S2T143, S2T148, S2T151 |
| 尾巴比躯干长，腿比躯干长。 | 11 | body_ref | S2T113, S2T118, S2T122, S2T129, S2T136, S2T142, S2T159, S2T176 |
| 尾巴比躯干长，腿比较长。 | 5 | body_ref | S2T108, S2T153, S2T161, S2T162, S2T167 |
| 尾巴比躯干短，头很长。 | 4 | body_ref | S2T76, S2T77, S2T80, S2T83 |
| 尾巴比躯干短，头比脖子长。 | 4 | body_ref | S2T89, S2T90, S2T92, S2T93 |
| 尾巴比躯干长，腿很短。 | 3 | body_ref | S2T86, S2T94, S2T112 |
| 头和尾巴差不多长，脖子短，腿长。 | 2 | equality | S1T238, S1T239 |
| 尾巴最长，头第二长，脖子和腿短。 | 2 | ranking | S1T91, S1T92 |
| 尾巴比躯干短，头和脖子长。 | 2 | body_ref | S2T105, S2T106 |
| 尾巴比躯干短，脖子不知道比躯干短不短。 | 2 | meta, body_ref, negation | S2T157, S2T160 |
| 尾巴比躯干长，头比脖子长。 | 2 | body_ref | S2T96, S2T111 |
| 腿比尾巴长，躯干脖子长。 | 2 | body_ref | S1T53, S1T54 |
| 从长到短是尾巴、躯干、头、脖子、腿。 | 1 | ranking, body_ref | S1T185 |
| 从长到短是尾巴、躯干、腿、头、脖子。 | 1 | ranking, body_ref | S1T186 |
| 判断不出哪一部分更长，尾巴不短。 | 1 | negation | S1T248 |
| 头和尾巴一样长。 | 1 | equality | S1T21 |
| 头和尾巴差不多长，脖子也不太长，腿很短。 | 1 | equality, negation | S1T265 |
| 头和尾巴差不多长，脖子也差不多长，腿短。 | 1 | equality | S1T224 |
| 头和尾巴差不多长，腿不短。 | 1 | equality, negation | S1T261 |
| 头和尾巴差不多长，腿不短，脖子比较长，算太长。 | 1 | equality, negation | S1T263 |
| 头和尾巴差不多长，腿比较短。 | 1 | equality | S1T262 |
| 头和尾巴差不多，头更长，脖子很长，腿很长。 | 1 | equality | S1T245 |
| 头和尾巴差不多，脖子长，腿长。 | 1 | equality | S1T206 |
| 头和尾巴相比，尾巴更长，脖子和尾巴都很长，躯干腿很短。 | 1 | body_ref | S1T259 |
| 头并没有很明显的长，尾巴很长，和腿差不多长，腿不短。 | 1 | equality, negation | S1T257 |
| 头最短，脖子比腿长，尾巴和脖子差不多长。 | 1 | equality | S1T146 |
| 头最长，躯干头和尾巴相差很大。 | 1 | body_ref | S1T202 |
| 头比尾巴长，脖子不短。 | 1 | negation | S1T236 |
| 头比脖子长，尾巴和腿差不多长。 | 1 | equality | S1T112 |
| 头没有明显更长，尾巴和腿差别不大。 | 1 | negation | S1T165 |
| 头没有比尾巴长很多，脖子很长。 | 1 | negation | S1T215 |
| 头短，尾巴长，腿不短。 | 1 | negation | S1T282 |
| 头长，尾巴长，腿不短。 | 1 | negation | S1T292 |
| 尾巴不短，腿长，脖子和头都比较长。 | 1 | negation | S2T44 |
| 尾巴和头都很长，腿不短。 | 1 | negation | S1T195 |
| 尾巴和躯干看不清。 | 1 | body_ref | S2T110 |
| 尾巴最长，腿第二长。 | 1 | ranking | S1T192 |
| 尾巴最长，躯干和腿的差距比较大，脖子和头都比较短。 | 1 | body_ref | S1T99 |
| 尾巴比腿短，躯干头比脖子长。 | 1 | body_ref | S1T103 |
| 尾巴比腿长，躯干差距很大。 | 1 | body_ref | S1T102 |
| 尾巴比腿长，躯干脖子短。 | 1 | body_ref | S1T51 |
| 尾巴比躯干看不清，脖子比较短，腿比较长。 | 1 | body_ref | S2T107 |
| 尾巴比躯干短一点，脖子比躯干长。 | 1 | body_ref | S2T130 |
| 尾巴比躯干短，头比脖子短。 | 1 | body_ref | S1T162 |
| 尾巴比躯干短，头比躯干和脖子长。 | 1 | body_ref | S2T104 |
| 尾巴比躯干短，腿很短。 | 1 | body_ref | S2T97 |
| 尾巴比躯干短，腿比躯干长。 | 1 | body_ref | S2T128 |
| 尾巴比躯干长。 | 1 | body_ref | S2T98 |
| 尾巴比躯干长，头和脖子很短。 | 1 | body_ref | S2T78 |
| 尾巴比躯干长，脖子和腿比躯干长。 | 1 | body_ref | S2T158 |
| 尾巴比躯干长，脖子很长。 | 1 | body_ref | S2T79 |
| 尾巴比躯干长，脖子比躯干长。 | 1 | body_ref | S2T181 |
| 尾巴比躯干长，腿和躯干差不多。 | 1 | equality, body_ref | S2T116 |
| 尾巴比躯干长，腿很长。 | 1 | body_ref | S2T85 |
| 看不清尾巴长还是躯干长。 | 1 | body_ref | S2T87 |
| 脖子很长，尾巴也很长，头和腿很短，它们都短于躯干。 | 1 | body_ref | S1T164 |
| 脖子最长，腿第二长。 | 1 | ranking | S1T193 |
| 脖子没有明显的优势，腿比较长，腿和尾巴都很长。 | 1 | negation | S1T272 |
| 腿和尾巴差不多长，脖子长，头短。 | 1 | equality | S1T121 |
| 腿很短，尾巴和腿差不多长，头很长。 | 1 | equality | S1T110 |
| 腿最短，脖子和尾巴一样长，比腿长，头最长。 | 1 | equality | S1T63 |
| 腿最长，脖子、头次之，尾巴最短。 | 1 | ranking | S1T159 |
| 腿比尾巴长一点，头和脖子差不多长。 | 1 | equality | S1T109 |
| 腿比躯干短，头、脖子和尾巴都很长。 | 1 | body_ref | S1T190 |
| 腿比躯干短，脖子比躯干长。 | 1 | body_ref | S2T193 |
| 腿比躯干长。 | 1 | body_ref | S1T163 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头很长。 | 2 | 0.000 | absolute_long:头 > 0.50 | S2T9, S2T62 |
| 头没有比尾巴长很多，脖子很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T215 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T67 |
| 尾巴很长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T246 |
| 尾巴比腿长。 | 1 | 0.000 | comparison:尾巴 > 腿 | S2T52 |
| 看不清尾巴长还是躯干长。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S2T87 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T303 |
| 四个部位都长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S2T66 |
| 头和尾巴差不多长，脖子也不太长，腿很短。 | 1 | 0.333 | equality_range:头+尾巴 =; absolute_short:脖子 < 0.50 | S1T265 |
| 头长，尾巴短，脖子比较短。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T278 |
| 尾巴比腿长，头长，脖子短。 | 1 | 0.333 | comparison:尾巴 > 腿; absolute_long:头 > 0.50 | S1T81 |
| 尾巴比腿长，脖子短，头长。 | 1 | 0.333 | comparison:尾巴 > 腿; absolute_long:头 > 0.50 | S1T62 |
| 从长到短是尾巴、躯干、头、脖子、腿。 | 1 | 0.429 | body_ref:尾巴 = 0.50; body_ref:头 = 0.50; body_ref:脖子 = 0.50; body_ref:腿 = 0.50 | S1T185 |

### S319

- trial 数: 1472; 非空文本: 1441; fidelity 可评分率: 0.933; 平均 fidelity: 0.883; 完全忠实率: 0.737; 低 fidelity 率: 0.058.
- 旧版 region 覆盖率: 0.933; 旧版 region 有未处理片段率: 0.050.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 1369 | 0.930 |
| comparison | 518 | 0.352 |
| body_ref | 309 | 0.210 |
| superlative | 276 | 0.188 |
| equality | 66 | 0.045 |
| group_sum | 44 | 0.030 |
| empty | 31 | 0.021 |
| count_abstract | 23 | 0.016 |
| meta | 6 | 0.004 |
| other | 2 | 0.001 |
| negation | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头最长。 | 59 |
| 头最短。 | 49 |
| 尾巴最短。 | 46 |
| 头短，尾巴短。 | 45 |
| 头短，尾巴长。 | 42 |
| 头和脖子都很长。 | 37 |
| 腿长。 | 36 |
| 头长，腿长。 | 34 |
| 头和脖子特别长。 | 34 |
| 头长，腿短。 | 33 |
| 四个部位都很长。 | 31 |
| 头长于躯干。 | 29 |
| 脖子最长。 | 28 |
| 头短于躯干。 | 27 |
| 头和脖子的比例大于腿和尾巴。 | 26 |
| 尾巴最长。 | 25 |
| 头短于躯干，尾巴长。 | 20 |
| 尾巴特别短。 | 18 |
| 头长于躯干，腿短。 | 17 |
| 头长于躯干，腿长。 | 16 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头长于躯干。 | 29 | body_ref | S4T72, S4T78, S4T80, S4T105, S4T115, S4T118, S4T120, S4T122 |
| 头短于躯干。 | 27 | body_ref | S4T70, S4T73, S4T77, S4T79, S4T82, S4T88, S4T98, S4T99 |
| 头和脖子的比例大于腿和尾巴。 | 26 | group_sum | S1T88, S1T100, S1T103, S1T108, S1T109, S1T111, S1T113, S1T115 |
| 头短于躯干，尾巴长。 | 20 | body_ref | S4T209, S4T211, S4T215, S4T217, S4T219, S4T220, S4T225, S4T226 |
| 头长于躯干，腿短。 | 17 | body_ref | S4T177, S4T178, S4T181, S4T189, S4T193, S4T197, S4T198, S4T203 |
| 头长于躯干，腿长。 | 16 | body_ref | S4T173, S4T182, S4T190, S4T191, S4T195, S4T201, S4T202, S4T216 |
| 头短于躯干，尾巴很长。 | 15 | body_ref | S4T281, S4T285, S4T287, S4T288, S4T289, S4T290, S4T293, S4T294 |
| 头比躯干长，腿长。 | 12 | body_ref | S5T4, S5T5, S5T6, S5T7, S5T10, S5T11, S5T15, S5T18 |
| 头短于躯干，尾巴短。 | 12 | body_ref | S4T210, S4T212, S4T214, S4T222, S4T223, S4T227, S4T230, S4T238 |
| 头和脖子的比例小于腿和尾巴。 | 10 | group_sum | S1T98, S1T101, S1T102, S1T104, S1T112, S1T118, S1T122, S1T123 |
| 头比躯干长，腿短。 | 10 | body_ref | S5T2, S5T16, S5T21, S5T24, S5T26, S5T31, S5T32, S5T33 |
| 脖子短于躯干。 | 10 | body_ref | S4T66, S4T68, S4T69, S4T93, S4T96, S4T97, S4T100, S4T110 |
| 头长于躯干，腿很长。 | 8 | body_ref | S4T279, S4T280, S4T283, S4T286, S4T301, S4T304, S4T307, S4T309 |
| 四个部位长度都差不多。 | 7 | equality | S2T49, S2T97, S2T99, S2T145, S2T146, S2T147, S2T176 |
| 头比躯干短，尾巴长。 | 7 | body_ref | S5T1, S5T8, S5T13, S5T19, S5T20, S5T27, S5T36 |
| 头短于躯干，尾巴很短。 | 7 | body_ref | S4T291, S4T292, S4T300, S4T305, S4T306, S4T314, S4T316 |
| 头长于躯干，腿很短。 | 7 | body_ref | S4T267, S4T282, S4T284, S4T298, S4T299, S4T308, S4T318 |
| 四个部位长度差不多。 | 6 | equality | S2T81, S2T154, S2T155, S2T167, S2T172, S2T183 |
| 四个部位长度很平均。 | 6 | equality | S2T240, S3T5, S3T9, S3T27, S3T207, S3T302 |
| 脖子长于躯干。 | 6 | body_ref | S2T199, S4T84, S4T90, S4T94, S4T123, S4T125 |
| 选错了。 | 6 | meta | S2T201, S2T267, S2T283, S3T66, S3T102, S3T123 |
| 四个部位差不多长。 | 5 | equality | S2T162, S2T163, S3T86, S3T87, S3T234 |
| 四个部位长度都很平均。 | 5 | equality | S3T127, S3T128, S3T129, S3T134, S3T204 |
| 头短于躯干，尾巴比较长。 | 5 | body_ref | S4T260, S4T265, S4T270, S4T275, S4T278 |
| 头比躯干短，尾巴短。 | 4 | body_ref | S5T3, S5T9, S5T12, S5T22 |
| 头短于躯干，尾巴比较短。 | 4 | body_ref | S4T261, S4T263, S4T266, S4T273 |
| 头短于躯干，脖子长。 | 4 | body_ref | S4T175, S4T185, S4T188, S4T199 |
| 头短于躯干，腿长。 | 4 | body_ref | S4T162, S4T166, S4T179, S4T180 |
| 脖子比躯干长。 | 4 | body_ref | S2T254, S4T52, S4T53, S4T55 |
| 腿比躯干短。 | 4 | body_ref | S2T245, S4T45, S4T46, S4T48 |
| 腿比躯干长。 | 4 | body_ref | S4T14, S4T15, S4T18, S4T42 |
| 四个部位一样长。 | 3 | equality | S3T313, S3T315, S4T104 |
| 头短于躯干，尾巴短于躯干。 | 3 | body_ref | S4T296, S4T310, S4T317 |
| 头短于躯干，脖子短。 | 3 | body_ref | S4T187, S4T204, S4T208 |
| 三个部位都比躯干长。 | 2 | body_ref, count_abstract | S4T50, S4T51 |
| 三个部位长于躯干。 | 2 | body_ref, count_abstract | S2T305, S2T307 |
| 四个部位比较平均。 | 2 | equality | S2T70, S2T204 |
| 四个部位都小于躯干。 | 2 | body_ref | S3T156, S4T61 |
| 四个部位都差不多长。 | 2 | equality | S2T141, S3T233 |
| 四个部位都很平均。 | 2 | equality | S3T109, S4T127 |
| 四个部位都比躯干短。 | 2 | body_ref | S4T37, S4T43 |
| 四个部位都比躯干长一点。 | 2 | body_ref | S3T50, S3T51 |
| 四个部位都长于躯干。 | 2 | body_ref | S2T200, S4T113 |
| 头和脖子特别长，其他两个部位特别短。 | 2 | count_abstract | S2T177, S2T178 |
| 头短于躯干，头比脖子短。 | 2 | body_ref | S4T159, S4T163 |
| 头短于躯干，尾巴最短。 | 2 | body_ref | S4T258, S4T269 |
| 头长于躯干，尾巴比较短。 | 2 | body_ref | S4T274, S4T277 |
| 头长于躯干，腿比较短。 | 2 | body_ref | S4T262, S4T272 |
| 腿短于躯干。 | 2 | body_ref | S4T85, S4T86 |
| 除了尾巴以外，其他部位都一样长。 | 2 | equality | S3T179, S3T190 |
| 除了尾巴，其他三个部位都比躯干长。 | 2 | body_ref, count_abstract | S2T210, S2T257 |
| 三个部位是腿比较短。 | 1 | count_abstract | S2T293 |
| 三个部位都比较长。 | 1 | count_abstract | S3T286 |
| 四个部位比躯干短。 | 1 | body_ref | S3T22 |
| 四个部位都一样短。 | 1 | equality | S3T218 |
| 四个部位都一样长。 | 1 | equality | S3T176 |
| 四个部位都不长，尾巴最短。 | 1 | negation | S2T171 |
| 四个部位都差不多。 | 1 | equality | S2T151 |
| 四个部位都很长，腿短于躯干。 | 1 | body_ref | S3T159 |
| 四个部位都比躯干要长。 | 1 | body_ref | S2T195 |
| 四个部位都比躯干长。 | 1 | body_ref | S2T239 |
| 四个部位长度平均。 | 1 | equality | S3T225 |
| 四个部位长度比较平均。 | 1 | equality | S2T110 |
| 四个部位长度都小于躯干。 | 1 | body_ref | S3T270 |
| 四个部位长度都小于躯干，且差不多长。 | 1 | equality, body_ref | S3T78 |
| 四个长度很平均。 | 1 | equality | S3T307 |
| 头和尾巴都小于躯干。 | 1 | body_ref | S4T297 |
| 头和尾巴都比躯干短。 | 1 | body_ref | S5T37 |
| 头和脖子比例小于腿和尾巴。 | 1 | group_sum | S1T134 |
| 头和脖子比较短，腿短于躯干。 | 1 | body_ref | S3T17 |
| 头和脖子的比例大于尾巴和腿。 | 1 | group_sum | S1T151 |
| 头和脖子的比例小于腿和尾巴，腿长。 | 1 | group_sum | S1T161 |
| 头和脖子的比例等于腿和尾巴。 | 1 | equality, group_sum | S1T138 |
| 头和脖子的比例长于腿和尾巴。 | 1 | group_sum | S1T95 |
| 头和脖子都长于躯干。 | 1 | body_ref | S4T114 |
| 头和脖子长于躯干。 | 1 | body_ref | S4T117 |
| 头和腿一样长。 | 1 | equality | S1T310 |
| 头和腿长于躯干。 | 1 | body_ref | S2T309 |
| 头小于躯干。 | 1 | body_ref | S4T71 |
| 头小于躯干，尾巴最长。 | 1 | body_ref | S4T268 |
| 头小于躯干，尾巴比较短。 | 1 | body_ref | S4T264 |
| 头小于躯干，脖子长。 | 1 | body_ref | S4T186 |
| 头明显短，头最短，其他都很平均。 | 1 | equality | S3T200 |
| 头有点小。 | 1 | other | S1T255 |
| 头比躯干小，尾巴短。 | 1 | body_ref | S5T29 |
| 头比躯干短。 | 1 | body_ref | S2T247 |
| 头比躯干短，腿短。 | 1 | body_ref | S5T17 |
| 头比躯干长，尾巴短。 | 1 | body_ref | S5T14 |
| 头短于躯干，四个部位都比躯干长。 | 1 | body_ref | S4T164 |
| 头短于躯干，四个部位都比较长，除了脖子短。 | 1 | body_ref | S4T183 |
| 头短于躯干，头比脖子长。 | 1 | body_ref | S4T158 |
| 头短于躯干，尾巴短，其他两个部位都很长。 | 1 | body_ref, count_abstract | S4T184 |
| 头短于躯干，腿短。 | 1 | body_ref | S4T161 |
| 头短于躯干，腿短于躯干。 | 1 | body_ref | S4T92 |
| 头短于躯干，腿短，脖子长。 | 1 | body_ref | S4T170 |
| 头短于躯干，腿长于躯干。 | 1 | body_ref | S4T213 |
| 头长于躯干，其他部位都很长。 | 1 | body_ref | S4T172 |
| 头长于躯干，尾巴比较长。 | 1 | body_ref | S4T276 |
| 头长于躯干，尾巴长。 | 1 | body_ref | S4T174 |
| 头长于躯干，腿也很长。 | 1 | body_ref | S4T271 |
| 头长于躯干，腿相对较短。 | 1 | body_ref | S4T320 |
| 头长于躯干，腿长于躯干。 | 1 | body_ref | S4T196 |
| 尾巴和头一样长。 | 1 | equality | S1T212 |
| 尾巴和腿的比例大于头和脖子的比例。 | 1 | group_sum | S1T187 |
| 尾巴和腿都小于躯干。 | 1 | body_ref | S4T111 |
| 尾巴明显短，尾巴最短，其他都平均。 | 1 | equality | S3T199 |
| 比较均匀。 | 1 | equality | S1T278 |
| 脖子和腿比躯干长。 | 1 | body_ref | S2T246 |
| 腿、头和脖子都差不多。 | 1 | equality | S3T171 |
| 腿和头长于躯干，腿比较短。 | 1 | body_ref | S4T259 |
| 腿和尾巴的比例大于头和脖子。 | 1 | group_sum | S1T147 |
| 腿和尾巴的比例大于脖子和头的比例。 | 1 | group_sum | S1T172 |
| 腿和尾巴短于躯干。 | 1 | body_ref | S4T63 |
| 腿和脖子都是一样长。 | 1 | equality | S1T4 |
| 腿和躯干一样长。 | 1 | equality, body_ref | S4T60 |
| 腿最短，其他三个部位都是平均长度。 | 1 | equality, count_abstract | S3T148 |
| 长度都很平均。 | 1 | equality | S4T108 |
| 除了头以外，其他部位都一样长。 | 1 | equality | S3T183 |
| 除了头以外，其他部位都很长，很平均。 | 1 | equality | S3T197 |
| 除了尾巴以外，其他三个部位都很长，而且很平均。 | 1 | equality, count_abstract | S3T145 |
| 除了尾巴，三个部位都很长。 | 1 | count_abstract | S3T283 |
| 除了尾巴，三个部位长度差不多。 | 1 | equality, count_abstract | S3T77 |
| 除了尾巴，其他三个部位都很短。 | 1 | count_abstract | S2T62 |
| 除了尾巴，其他三个部位都很长。 | 1 | count_abstract | S3T314 |
| 除了尾巴，其他三个部位都比脖子要长。 | 1 | count_abstract | S2T228 |
| 除了尾巴，其他部位都比躯干长。 | 1 | body_ref | S4T64 |
| 除了尾巴，都小于躯干。 | 1 | body_ref | S4T62 |
| 除了脖子以外，三个部位平均较长。 | 1 | equality, count_abstract | S3T95 |
| 除了脖子以外，三个部位都很长。 | 1 | count_abstract | S3T91 |
| 除了脖子以外，其他三个部位平均较长。 | 1 | equality, count_abstract | S3T94 |
| 除了脖子以外，其他三个部位都很长。 | 1 | count_abstract | S3T141 |
| 除了脖子以外，其他部位都一样长。 | 1 | equality | S3T175 |
| 除了脖子，都很平衡。 | 1 | other | S4T95 |
| 除了脖子，都比躯干短。 | 1 | body_ref | S4T41 |
| 除了脖子，都长于躯干。 | 1 | body_ref | S4T91 |
| 除了腿以外，其他三个部位都很长。 | 1 | count_abstract | S3T140 |
| 除了腿以外，其他部位都很长，且平均。 | 1 | equality | S3T198 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度都差不多。 | 7 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T49, S2T97, S2T99, S2T145, S2T146, S2T147, S2T176 |
| 四个部位长度差不多。 | 6 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T81, S2T154, S2T155, S2T167, S2T172, S2T183 |
| 四个部位长度很平均。 | 6 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T240, S3T5, S3T9, S3T27, S3T207, S3T302 |
| 四个部位差不多长。 | 5 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T162, S2T163, S3T86, S3T87, S3T234 |
| 四个部位长度都很平均。 | 5 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T127, S3T128, S3T129, S3T134, S3T204 |
| 四个部位一样长。 | 3 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T313, S3T315, S4T104 |
| 四个部位比较平均。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T70, S2T204 |
| 四个部位都差不多长。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T141, S3T233 |
| 四个部位都很平均。 | 2 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T109, S4T127 |
| 头特别长。 | 2 | 0.000 | absolute_long:头 > 0.50 | S1T293, S2T164 |
| 除了尾巴以外，其他部位都一样长。 | 2 | 0.000 | equality_range:脖子+头+腿 = | S3T179, S3T190 |
| 除了尾巴都很长。 | 2 | 0.000 | absolute_long:尾巴 > 0.50 | S2T64, S2T182 |
| 除了尾巴，都很短。 | 2 | 0.125 | exclusion:脖子 < 0.50; exclusion:头 < 0.50; exclusion:腿 < 0.50; exclusion:尾巴 > 0.50 | S1T50, S4T83 |
| 头和脖子比较长，腿比较短。 | 2 | 0.333 | absolute_short:腿 < 0.50; absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T5, S2T6 |
| 头最长。 | 2 | 0.333 | superlative:头 > 腿; superlative:头 > 尾巴 | S3T154, S4T11 |
| 四个部位都一样短。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T218 |
| 四个部位都一样长。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T176 |
| 四个部位都小于躯干。 | 1 | 0.000 | body_ref:脖子 < 0.50; body_ref:头 < 0.50; body_ref:腿 < 0.50; body_ref:尾巴 < 0.50 | S3T156 |
| 四个部位都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T151 |
| 四个部位长度平均。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T225 |
| 四个部位长度比较平均。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S2T110 |
| 四个长度很平均。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S3T307 |
| 头和脖子相对就长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T168 |
| 头和脖子都比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T71 |
| 头和腿一样长。 | 1 | 0.000 | equality_range:头+腿 = | S1T310 |
| 头和腿长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S3T130 |
| 头比较长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S3T71 |
| 头短于躯干，腿长于躯干。 | 1 | 0.000 | body_ref:头 < 0.50; body_ref:腿 > 0.50 | S4T213 |
| 头短，尾巴短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:尾巴 < 0.50 | S5T95 |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T298 |
| 头长，腿长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S5T124 |
| 脖子和腿都很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S1T259 |
| 脖子很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T136 |
| 脖子特别长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S2T161 |
| 腿、头和脖子都差不多。 | 1 | 0.000 | equality_range:腿+头+脖子 = | S3T171 |
| 腿和躯干一样长。 | 1 | 0.000 | body_ref:腿 = 0.50 | S4T60 |
| 腿比躯干短。 | 1 | 0.000 | body_ref:腿 < 0.50 | S2T245 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S1T232 |
| 除了头以外，其他部位都一样长。 | 1 | 0.000 | equality_range:脖子+腿+尾巴 = | S3T183 |
| 除了头，都很长。 | 1 | 0.000 | exclusion:脖子 > 0.50; exclusion:腿 > 0.50; exclusion:尾巴 > 0.50; exclusion:头 < 0.50 | S2T297 |
| 除了尾巴，三个部位长度差不多。 | 1 | 0.000 | equality_range:脖子+头+腿 = | S3T77 |
| 除了尾巴，都很长。 | 1 | 0.000 | exclusion:脖子 > 0.50; exclusion:头 > 0.50; exclusion:腿 > 0.50; exclusion:尾巴 < 0.50 | S2T304 |
| 除了脖子以外，其他三个部位平均较长。 | 1 | 0.000 | exclusion:头 > 0.50; exclusion:腿 > 0.50; exclusion:尾巴 > 0.50; exclusion:脖子 < 0.50 | S3T94 |
| 除了脖子以外，其他部位都一样长。 | 1 | 0.000 | equality_range:头+腿+尾巴 = | S3T175 |
| 除了脖子都很长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S4T76 |
| 除了脖子，都很短。 | 1 | 0.000 | exclusion:头 < 0.50; exclusion:腿 < 0.50; exclusion:尾巴 < 0.50; exclusion:脖子 > 0.50 | S4T25 |
| 除了腿都很长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S2T78 |
| 除了腿，都很长。 | 1 | 0.250 | exclusion:头 > 0.50; exclusion:尾巴 > 0.50; exclusion:腿 < 0.50 | S2T27 |
| 头和脖子都很长，腿很短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S3T85 |
| 头和脖子都特别长，腿特别短。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T85 |
| 头最短。 | 1 | 0.333 | superlative:头 < 脖子; superlative:头 < 尾巴 | S3T120 |

### S320

- trial 数: 768; 非空文本: 768; fidelity 可评分率: 1.000; 平均 fidelity: 0.905; 完全忠实率: 0.806; 低 fidelity 率: 0.034.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 768 | 1.000 |
| superlative | 5 | 0.007 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 四个部位都长。 | 79 |
| 头长，尾巴长。 | 65 |
| 脖子短。 | 44 |
| 腿短。 | 44 |
| 尾巴短。 | 42 |
| 腿长。 | 38 |
| 头短。 | 35 |
| 头长，尾巴短。 | 34 |
| 头长，腿长。 | 33 |
| 头短，腿长。 | 32 |
| 头长，脖子长。 | 30 |
| 脖子长，腿长。 | 26 |
| 头长。 | 25 |
| 头短，腿短。 | 22 |
| 脖子长。 | 21 |
| 腿长，尾巴长。 | 21 |
| 脖子长，尾巴长。 | 20 |
| 尾巴长。 | 15 |
| 脖子和腿长。 | 15 |
| 头和腿长。 | 13 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿长。 | 8 | 0.000 | absolute_long:腿 > 0.50 | S1T99, S1T104, S1T134, S1T184, S2T98, S2T99, S2T103, S2T133 |
| 脖子长。 | 5 | 0.000 | absolute_long:脖子 > 0.50 | S1T193, S1T222, S1T250, S2T10, S2T170 |
| 头长，腿长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T264, S2T91, S2T264 |
| 头长，尾巴长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S2T179, S3T42 |
| 四个部位都长。 | 2 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T116, S1T236 |
| 头长，脖子长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T126 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S1T286 |
| 尾巴长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S2T158 |
| 脖子和尾巴长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T57 |
| 脖子和腿长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:腿 > 0.50 | S1T25 |
| 脖子长，尾巴长，脖子长。 | 1 | 0.333 | absolute_long:脖子 > 0.50 | S2T3 |

### S321

- trial 数: 768; 非空文本: 709; fidelity 可评分率: 0.923; 平均 fidelity: 0.844; 完全忠实率: 0.667; 低 fidelity 率: 0.057.
- 旧版 region 覆盖率: 0.923; 旧版 region 有未处理片段率: 0.026.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 709 | 0.923 |
| comparison | 243 | 0.316 |
| superlative | 91 | 0.118 |
| empty | 59 | 0.077 |
| count_abstract | 14 | 0.018 |
| negation | 13 | 0.017 |
| body_ref | 12 | 0.016 |
| equality | 9 | 0.012 |
| ranking | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，头短。 | 43 |
| 腿长，头长。 | 33 |
| 腿短，头长。 | 27 |
| 腿短，尾巴长。 | 26 |
| 腿短，尾巴短。 | 17 |
| 腿长。 | 14 |
| 腿短。 | 14 |
| 头比脖子短，腿比尾巴长。 | 13 |
| 头长。 | 12 |
| 腿相对较长，头相对较短。 | 11 |
| 头长，腿短。 | 9 |
| 头短。 | 9 |
| 腿短，头短。 | 8 |
| 腿长，脖子短。 | 7 |
| 腿最短。 | 7 |
| 头比脖子长，腿比尾巴短。 | 7 |
| 头和尾巴短。 | 6 |
| 腿比尾巴长，头比脖子长。 | 6 |
| 头最短。 | 6 |
| 腿相对较长，头相对较长。 | 6 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿长，头超过躯干了。 | 4 | body_ref | S2T255, S2T257, S2T261, S2T262 |
| 头和脖子是最短的两个部位。 | 3 | count_abstract | S1T85, S1T86, S1T88 |
| 腿和头是最短的两个部位。 | 3 | count_abstract | S1T83, S1T94, S1T96 |
| 腿长，头没有超过躯干。 | 3 | body_ref, negation | S2T253, S2T256, S2T259 |
| 尾巴和头是最短的两个部位。 | 2 | count_abstract | S1T89, S1T90 |
| 脖子和尾巴是最短的两个部位。 | 2 | count_abstract | S1T87, S1T92 |
| 腿不是最短的，头比脖子长。 | 2 | negation | S2T92, S2T109 |
| 腿和脖子是最短的两个部位。 | 2 | count_abstract | S1T91, S1T95 |
| 腿长，头超过躯干。 | 2 | body_ref | S2T254, S2T260 |
| 四个部位长度都差不多。 | 1 | equality | S1T4 |
| 头和尾巴是最短的两个部位。 | 1 | count_abstract | S1T93 |
| 头和脖子一样长。 | 1 | equality | S1T143 |
| 头和脖子和尾巴差不多长。 | 1 | equality | S2T142 |
| 头和脖子差不多长，尾巴非常短，腿很长。 | 1 | equality | S2T143 |
| 头比脖子长，腿和尾巴差不多。 | 1 | equality | S2T26 |
| 尾巴短，头和脖子一样长。 | 1 | equality | S1T144 |
| 腿不是最短的，头比脖子短。 | 1 | negation | S2T90 |
| 腿不是最短的，头短于脖子。 | 1 | negation | S2T93 |
| 腿不是最短，头比脖子和尾巴短。 | 1 | negation | S2T106 |
| 腿不是最短，头比脖子长。 | 1 | negation | S2T108 |
| 腿不是最短，头较脖子更短。 | 1 | negation | S2T102 |
| 腿不是最短，尾巴比头和脖子都长。 | 1 | negation | S2T103 |
| 腿不是最短，腿较长。 | 1 | negation | S2T148 |
| 腿和尾巴是最短的两个部位。 | 1 | count_abstract | S1T84 |
| 腿和尾巴都相对较长，头和脖子也差不多。 | 1 | equality | S1T74 |
| 腿最短，尾巴第二短。 | 1 | ranking | S2T203 |
| 腿比尾巴短，头和脖子一样长。 | 1 | equality | S2T82 |
| 腿比尾巴长，头与脖子差不多长。 | 1 | equality | S1T68 |
| 腿相对较长，头没有超过躯干。 | 1 | body_ref, negation | S2T258 |
| 腿短，头没超过躯干。 | 1 | body_ref | S2T263 |
| 腿短，头超过躯干了。 | 1 | body_ref | S2T264 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿长。 | 6 | 0.000 | absolute_long:腿 > 0.50 | S1T198, S1T206, S1T303, S1T307, S1T318, S2T275 |
| 头长。 | 4 | 0.000 | absolute_long:头 > 0.50 | S1T178, S1T190, S1T302, S1T306 |
| 腿短。 | 3 | 0.000 | absolute_short:腿 < 0.50 | S1T126, S1T319, S2T238 |
| 头和脖子整体比较长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T248, S1T252 |
| 头长，腿短。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_short:腿 < 0.50 | S1T214, S1T217 |
| 腿相对较短。 | 2 | 0.000 | absolute_short:腿 < 0.50 | S2T74, S2T127 |
| 腿长，头长。 | 2 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S1T109, S1T148 |
| 四个部位长度都差不多。 | 1 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T4 |
| 头比尾巴和脖子长。 | 1 | 0.000 | comparison:头 > 尾巴+脖子 | S2T175 |
| 头比脖子略短。 | 1 | 0.000 | comparison:头 < 脖子 | S1T75 |
| 头相对，脖子较短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S2T71 |
| 头短。 | 1 | 0.000 | absolute_short:头 < 0.50 | S1T192 |
| 头长，腿长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T232 |
| 脖子比头短。 | 1 | 0.000 | comparison:脖子 < 头 | S1T1 |
| 脖子短。 | 1 | 0.000 | absolute_short:脖子 < 0.50 | S1T283 |
| 腿不是最短，头较脖子更短。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_short:脖子 < 0.50 | S2T102 |
| 腿不是最短，腿较长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S2T148 |
| 腿和头短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S1T168 |
| 腿和尾巴短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T142 |
| 腿比尾巴短，头比脖子长、比尾巴长。 | 1 | 0.000 | comparison:腿 < 尾巴; comparison:头 > 脖子 | S1T65 |
| 腿比较短。 | 1 | 0.000 | absolute_short:腿 < 0.50 | S2T194 |
| 腿短，头短。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_short:头 < 0.50 | S1T310 |
| 腿短，头长。 | 1 | 0.000 | absolute_short:腿 < 0.50; absolute_long:头 > 0.50 | S1T118 |
| 腿长，头没有超过躯干。 | 1 | 0.000 | absolute_long:腿 > 0.50; body_ref:头 > 0.50 | S2T259 |
| 腿长，头短。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_short:头 < 0.50 | S1T133 |
| 腿长，脖子短。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_short:脖子 < 0.50 | S2T315 |
| 头比脖子短，腿在最短的两个之中。 | 1 | 0.250 | superlative:腿 < 脖子; superlative:腿 < 头; superlative:腿 < 尾巴 | S2T89 |
| 尾巴和头相对是最短。 | 1 | 0.333 | superlative:头 < 脖子; superlative:头 < 尾巴 | S2T51 |
| 腿较，尾巴和脖子都长，头也长。 | 1 | 0.333 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S1T78 |
| 腿非常短，头、脖子比较长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T31 |

### S322

- trial 数: 896; 非空文本: 859; fidelity 可评分率: 0.934; 平均 fidelity: 0.898; 完全忠实率: 0.750; 低 fidelity 率: 0.046.
- 旧版 region 覆盖率: 0.934; 旧版 region 有未处理片段率: 0.062.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 859 | 0.959 |
| comparison | 285 | 0.318 |
| equality | 258 | 0.288 |
| superlative | 194 | 0.217 |
| body_ref | 150 | 0.167 |
| empty | 37 | 0.041 |
| negation | 37 | 0.041 |
| count_abstract | 2 | 0.002 |
| ranking | 1 | 0.001 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，尾巴短。 | 54 |
| 头最长。 | 48 |
| 腿长，尾巴长。 | 43 |
| 腿短，头长。 | 42 |
| 腿短，头短。 | 42 |
| 头不是最长。 | 21 |
| 头和脖子长度相等。 | 17 |
| 尾巴和腿长度相等。 | 11 |
| 腿很长，头很短。 | 11 |
| 脖子和尾巴长度相等。 | 10 |
| 腿很短，头很长。 | 9 |
| 脖子和腿长度相等。 | 8 |
| 头略长于脖子。 | 7 |
| 腿很短，头很短。 | 7 |
| 脖子和尾巴一样长。 | 7 |
| 头和尾巴长度相等。 | 7 |
| 腿短，脖子比尾巴长。 | 7 |
| 腿短，尾巴长。 | 6 |
| 腿短，头比尾巴长。 | 6 |
| 腿长，脖子比躯干短。 | 6 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 头不是最长。 | 21 | negation | S1T186, S1T187, S1T223, S1T226, S1T227, S1T230, S1T233, S1T236 |
| 头和脖子长度相等。 | 17 | equality | S1T77, S1T78, S1T80, S1T150, S1T151, S1T173, S1T174, S1T181 |
| 尾巴和腿长度相等。 | 11 | equality | S1T76, S1T84, S1T87, S1T153, S1T155, S1T158, S1T243, S1T252 |
| 脖子和尾巴长度相等。 | 10 | equality | S1T159, S1T194, S1T220, S1T232, S1T241, S1T250, S1T263, S1T272 |
| 脖子和腿长度相等。 | 8 | equality | S1T81, S1T99, S1T183, S1T193, S1T242, S1T251, S1T261, S1T276 |
| 头和尾巴长度相等。 | 7 | equality | S1T148, S1T198, S1T219, S1T221, S1T225, S1T229, S1T231 |
| 脖子和尾巴一样长。 | 7 | equality | S1T108, S1T135, S1T257, S1T298, S1T304, S1T313, S1T317 |
| 头和腿长度相等。 | 6 | equality | S1T199, S1T209, S1T216, S1T224, S1T265, S1T286 |
| 脖子和腿一样长。 | 6 | equality | S1T104, S1T109, S1T294, S1T306, S1T318, S2T207 |
| 腿长，脖子比躯干短。 | 6 | body_ref | S2T291, S2T293, S3T35, S3T36, S3T46, S3T47 |
| 头和尾巴一样长。 | 5 | equality | S1T122, S1T163, S1T295, S1T299, S1T308 |
| 头和脖子一样长。 | 5 | equality | S1T56, S1T105, S1T107, S2T205, S2T209 |
| 头和脖子差不多长。 | 5 | equality | S1T50, S1T51, S1T53, S1T141, S1T145 |
| 腿短，头比躯干短。 | 5 | body_ref | S3T1, S3T4, S3T5, S3T6, S3T8 |
| 腿长，头比躯干短。 | 4 | body_ref | S2T296, S2T298, S3T7, S3T9 |
| 四个部位都短于躯干。 | 3 | body_ref | S2T14, S2T16, S2T201 |
| 头、脖子、尾巴都短于躯干，腿最长。 | 3 | body_ref | S2T53, S2T72, S2T78 |
| 头和尾巴差不多长。 | 3 | equality | S1T140, S1T142, S1T146 |
| 头和脖子长度相近。 | 3 | equality | S1T117, S1T168, S1T171 |
| 脖子和躯干一样长。 | 3 | equality, body_ref | S2T102, S2T206, S2T208 |
| 脖子比躯干短。 | 3 | body_ref | S2T186, S2T188, S2T189 |
| 腿和尾巴长度相等。 | 3 | equality | S1T192, S1T211, S1T218 |
| 腿短，尾巴比躯干短。 | 3 | body_ref | S3T65, S3T66, S3T70 |
| 腿短，尾巴比躯干长。 | 3 | body_ref | S2T288, S3T64, S3T67 |
| 腿长，头比躯干长。 | 3 | body_ref | S2T297, S3T2, S3T3 |
| 腿长，尾巴和躯干一样长。 | 3 | equality, body_ref | S2T284, S2T285, S2T287 |
| 腿长，尾巴比躯干短。 | 3 | body_ref | S2T286, S2T289, S2T290 |
| 头、腿、尾巴长度相等。 | 2 | equality | S1T152, S1T184 |
| 头和尾巴短于躯干。 | 2 | body_ref | S2T37, S2T38 |
| 头和腿一样长。 | 2 | equality | S1T301, S1T314 |
| 头比脖子短，脖子和躯干一样长。 | 2 | equality, body_ref | S2T130, S2T131 |
| 头比腿短，脖子和尾巴一样长。 | 2 | equality | S2T160, S2T165 |
| 头比躯干短。 | 2 | body_ref | S2T190, S2T191 |
| 头比躯干短，腿和尾巴一样长。 | 2 | equality, body_ref | S2T192, S2T193 |
| 头比躯干长，脖子和腿一样长。 | 2 | equality, body_ref | S2T194, S2T195 |
| 头短于躯干。 | 2 | body_ref | S2T121, S2T122 |
| 尾巴和腿长度相等，头略短于脖子。 | 2 | equality | S1T88, S1T93 |
| 尾巴比躯干短，头和脖子一样长。 | 2 | equality, body_ref | S2T183, S2T184 |
| 尾巴比躯干短，脖子和躯干一样长。 | 2 | equality, body_ref | S2T185, S2T187 |
| 尾巴短于躯干。 | 2 | body_ref | S2T126, S2T127 |
| 脖子、腿和尾巴短于躯干，头最长。 | 2 | body_ref | S2T66, S2T89 |
| 脖子和头长度相等。 | 2 | equality | S1T83, S1T262 |
| 脖子和腿一样长，头和尾巴一样长。 | 2 | equality | S2T104, S2T140 |
| 脖子和躯干一样长，其他三个部位都短于躯干。 | 2 | equality, body_ref, count_abstract | S2T49, S2T87 |
| 脖子比腿短，头和尾巴一样长。 | 2 | equality | S2T147, S2T149 |
| 脖子短于躯干。 | 2 | body_ref | S2T123, S2T125 |
| 腿短，脖子和躯干一样长。 | 2 | equality, body_ref | S2T294, S2T295 |
| 腿长，头和尾巴一样长。 | 2 | equality | S2T316, S3T88 |
| 腿长，脖子和尾巴一样长。 | 2 | equality | S2T304, S3T39 |
| 只有脖子略长于躯干。 | 1 | body_ref | S2T41 |
| 只有腿短于躯干。 | 1 | body_ref | S2T39 |
| 四个部位均短于躯干，头和脖子位于中间。 | 1 | body_ref | S2T109 |
| 四个部位都略短于躯干，尾巴和腿长度一样。 | 1 | equality, body_ref | S2T40 |
| 四个部位长度各不相同，脖子长于躯干。 | 1 | body_ref | S2T111 |
| 四个部位长度都不一样，尾巴最长，头和腿短于躯干，脖子和尾巴长于躯干。 | 1 | equality, body_ref, negation | S2T106 |
| 头、尾巴、脖子长度相等。 | 1 | equality | S1T149 |
| 头、脖子、尾巴都短于躯干。 | 1 | body_ref | S2T51 |
| 头、脖子、尾巴都长于躯干，腿最短。 | 1 | body_ref | S2T79 |
| 头、脖子、尾巴长度相等。 | 1 | equality | S1T240 |
| 头、脖子、腿都短于躯干。 | 1 | body_ref | S2T50 |
| 头、脖子、腿都短于躯干，尾巴最长。 | 1 | body_ref | S2T115 |
| 头、脖子、腿长度相等。 | 1 | equality | S1T282 |
| 头、脖子和尾巴都短于躯干。 | 1 | body_ref | S2T36 |
| 头、脖子和尾巴都短于躯干，腿最长、长于躯干。 | 1 | body_ref | S2T86 |
| 头、脖子和腿和躯干一样长，头最短。 | 1 | equality, body_ref | S2T80 |
| 头、脖子和腿短于躯干，尾巴最长。 | 1 | body_ref | S2T77 |
| 头、脖子和腿都短于躯干。 | 1 | body_ref | S2T35 |
| 头、脖子和腿长度相等。 | 1 | equality | S1T246 |
| 头、脖子和躯干长度相等，腿和尾巴短于躯干。 | 1 | equality, body_ref | S2T56 |
| 头、腿和尾巴短于躯干，脖子最长。 | 1 | body_ref | S2T91 |
| 头、腿和尾巴都短于躯干，脖子是最长。 | 1 | body_ref | S2T68 |
| 头、腿和尾巴都短于躯干，脖子最长。 | 1 | body_ref | S2T81 |
| 头、腿和尾巴长于躯干，脖子最短。 | 1 | body_ref | S2T47 |
| 头、腿和躯干一样长，脖子和尾巴短于躯干，尾巴最短。 | 1 | equality, body_ref | S2T85 |
| 头不是最长，头和脖子一样长，腿和尾巴一样长。 | 1 | equality, negation | S2T7 |
| 头不是最长，尾巴最长。 | 1 | negation | S2T6 |
| 头不是最长，脖子最长， | 1 | negation | S2T4 |
| 头不是最长，腿最长， | 1 | negation | S2T3 |
| 头和尾巴一样是最长。 | 1 | equality | S2T12 |
| 头和尾巴一样是最长，脖子和腿一样长。 | 1 | equality | S2T11 |
| 头和尾巴一样长，四个部位都短于躯干。 | 1 | equality, body_ref | S2T93 |
| 头和尾巴一样长，脖子和腿一样长，四个部位都短于躯干。 | 1 | equality, body_ref | S2T95 |
| 头和尾巴一样长，脖子最长。 | 1 | equality | S2T24 |
| 头和尾巴一样长，腿和躯干一样长，脖子最长。 | 1 | equality, body_ref | S2T88 |
| 头和尾巴一样长，长于躯干。 | 1 | equality, body_ref | S2T118 |
| 头和尾巴都短于躯干，腿是最长，脖子也略短于躯干。 | 1 | body_ref | S2T69 |
| 头和尾巴长度相等，略短于脖子。 | 1 | equality | S1T191 |
| 头和尾巴长度相近。 | 1 | equality | S1T176 |
| 头和脖子一样长，与腿相近。 | 1 | equality | S1T205 |
| 头和脖子一样长，且头不是最长。 | 1 | equality, negation | S2T26 |
| 头和脖子一样长，尾巴最短。 | 1 | equality | S1T310 |
| 头和脖子一样长，腿是最短。 | 1 | equality | S2T63 |
| 头和脖子一样长，腿最长，头、脖子、尾巴短于躯干。 | 1 | equality, body_ref | S2T107 |
| 头和脖子短于躯干，腿和尾巴长于躯干。 | 1 | body_ref | S2T94 |
| 头和脖子都短于躯干。 | 1 | body_ref | S2T100 |
| 头和脖子都长，腿和尾巴都短于躯干。 | 1 | body_ref | S2T70 |
| 头和脖子长于躯干，腿和尾巴短于躯干。 | 1 | body_ref | S2T65 |
| 头和脖子长度相等，略长于尾巴。 | 1 | equality | S1T207 |
| 头和腿一样长，且不是最长。 | 1 | equality, negation | S1T302 |
| 头和腿一样长，且头不是最长。 | 1 | equality, negation | S2T32 |
| 头和腿一样长，中等偏长，脖子和尾巴一样长，略长于前两者。 | 1 | equality | S1T39 |
| 头和腿一样长，尾巴是最长。 | 1 | equality | S2T13 |
| 头和腿一样长，尾巴最长。 | 1 | equality | S2T61 |
| 头和腿一样长，脖子和尾巴略短于躯干，尾巴最短。 | 1 | equality, body_ref | S2T83 |
| 头和腿的长度一样。 | 1 | equality | S1T312 |
| 头和腿短于躯干。 | 1 | body_ref | S2T101 |
| 头和腿长度相近。 | 1 | equality | S1T157 |
| 头和躯干一样长，脖子最短，腿和尾巴长于躯干。 | 1 | equality, body_ref | S2T84 |
| 头很短，脖子和尾巴一样长。 | 1 | equality | S2T221 |
| 头很长，头和尾巴一样长。 | 1 | equality | S2T222 |
| 头是最长，头和尾巴一样长。 | 1 | equality | S2T34 |
| 头是最长，头和脖子一样长。 | 1 | equality | S2T19 |
| 头是最长，头和脖子长度一样。 | 1 | equality | S2T10 |
| 头是最长，腿、尾巴和躯干差不多长。 | 1 | equality, body_ref | S2T54 |
| 头最短且和腿长度相近。 | 1 | equality | S1T166 |
| 头最长，头和脖子一样长。 | 1 | equality | S2T5 |
| 头最长，尾巴和头差不多长。 | 1 | equality | S2T1 |
| 头最长，脖子、尾巴、腿都短于躯干。 | 1 | body_ref | S2T90 |
| 头最长，脖子和腿差不多长。 | 1 | equality | S2T2 |
| 头最长，脖子和腿短于躯干。 | 1 | body_ref | S2T74 |
| 头比腿长，四个部位长度不相等。 | 1 | equality | S2T158 |
| 头比腿长，头和尾巴一样长。 | 1 | equality | S2T163 |
| 头比腿长，脖子和腿一样长。 | 1 | equality | S2T157 |
| 头比躯干长。 | 1 | body_ref | S2T196 |
| 头比躯干长，脖子和躯干一样长。 | 1 | equality, body_ref | S2T197 |
| 头略短于脖子、头、脖子、腿差不多长。 | 1 | equality | S1T114 |
| 头略长于脖子，头、脖子、尾巴三者长度相近。 | 1 | equality | S1T139 |
| 头略长于腿，脖子略长于腿，尾巴和腿差不多长，头比较短。 | 1 | equality | S1T49 |
| 头短于躯干，脖子最长。 | 1 | body_ref | S2T119 |
| 头长于躯干。 | 1 | body_ref | S2T120 |
| 尾巴和头长度相等。 | 1 | equality | S1T270 |
| 尾巴和腿一样长，头和脖子一样长。 | 1 | equality | S1T208 |
| 尾巴和腿差不多长。 | 1 | equality | S1T132 |
| 尾巴和腿短于躯干，脖子和头长于躯干，脖子最长。 | 1 | body_ref | S2T76 |
| 尾巴和腿长度相近。 | 1 | equality | S1T172 |
| 尾巴和躯干一样长，腿、脖子和头都短于躯干。 | 1 | equality, body_ref | S2T114 |
| 尾巴很短，腿和头差不多长，略长于尾巴，脖子最长。 | 1 | equality | S1T30 |
| 尾巴最长，头和腿短于躯干。 | 1 | body_ref | S2T62 |
| 尾巴比躯干短。 | 1 | body_ref | S2T182 |
| 尾巴略长于头，四个部位长度不相等。 | 1 | equality | S1T85 |
| 短于躯干，两个长于躯干，脖子是最长。 | 1 | body_ref | S2T17 |
| 短于躯干，头和躯干一样长。 | 1 | equality, body_ref | S2T203 |
| 短于躯干，尾巴和躯干一样长。 | 1 | equality, body_ref | S2T202 |
| 短于躯干，有两个长于躯干。 | 1 | body_ref | S2T20 |
| 短于躯干，腿最长，脖子很长。 | 1 | body_ref | S2T204 |
| 脖子、头、尾巴差不多长，腿最短。 | 1 | equality | S1T23 |
| 脖子、头、腿长度相等。 | 1 | equality | S1T185 |
| 脖子、头和腿一样长，尾巴最短。 | 1 | equality | S2T105 |
| 脖子、腿、尾巴都短于躯干，头最长。 | 1 | body_ref | S2T52 |
| 脖子、腿、尾巴长度相等。 | 1 | equality | S1T260 |
| 脖子和头一样长。 | 1 | equality | S1T45 |
| 脖子和头一样长，尾巴和腿一样长，脖子和头略短于尾巴和腿。 | 1 | equality | S1T7 |
| 脖子和头差不多长，腿和尾巴差不多长，且长于前二者。 | 1 | equality | S1T26 |
| 脖子和头长度相近。 | 1 | equality | S1T96 |
| 脖子和尾巴一样短，都短于躯干，腿最长，腿和头差不多长。 | 1 | equality, body_ref | S2T73 |
| 脖子和尾巴一样长，且头不是最长。 | 1 | equality, negation | S2T27 |
| 脖子和尾巴一样长，且尾巴最短。 | 1 | equality | S1T307 |
| 脖子和尾巴一样长，和头一样长。 | 1 | equality | S2T174 |
| 脖子和尾巴一样长，头、脖子和尾巴短于躯干，腿最长。 | 1 | equality, body_ref | S2T92 |
| 脖子和尾巴一样长，头不是最长。 | 1 | equality, negation | S2T18 |
| 脖子和尾巴一样长，头最长。 | 1 | equality | S2T25 |
| 脖子和尾巴一样长，头略短于二者，腿最短。 | 1 | equality | S1T18 |
| 脖子和尾巴短于躯干，腿最长。 | 1 | body_ref | S2T60 |
| 脖子和尾巴长于躯干，头和腿短于躯干。 | 1 | body_ref | S2T96 |
| 脖子和腿一样长，且头不是最长。 | 1 | equality, negation | S1T320 |
| 脖子和腿一样长，且都是四个部位中最长。 | 1 | equality | S2T59 |
| 脖子和腿一样长，头和尾巴比脖子和腿略长。 | 1 | equality | S1T11 |
| 脖子和腿一样长，头最长。 | 1 | equality | S2T22 |
| 脖子和腿一样长，尾巴和躯干一样长，头最短。 | 1 | equality, body_ref | S2T98 |
| 脖子和腿一样长，尾巴很短，头略长于尾巴。 | 1 | equality | S1T14 |
| 脖子和腿一样长，尾巴最短，头最长。 | 1 | equality | S2T99 |
| 脖子和腿一样，头略短，尾巴略短。 | 1 | equality | S1T38 |
| 脖子和腿差不多长。 | 1 | equality | S1T131 |
| 脖子和腿差不多长，头和尾巴差不多长，且长于前二者。 | 1 | equality | S1T22 |
| 脖子和腿差不多长，头和尾巴略长于脖子和腿。 | 1 | equality | S1T42 |
| 脖子和腿差不多长，头很短，尾巴比头长一些。 | 1 | equality | S1T9 |
| 脖子和腿略短于躯干。 | 1 | body_ref | S2T42 |
| 脖子和腿短于躯干，头和尾巴长于躯干。 | 1 | body_ref | S2T15 |
| 脖子和腿短于躯干，尾巴是最长。 | 1 | body_ref | S2T67 |
| 脖子和腿都是中等长度，头比脖子略长，和尾巴差不多长。 | 1 | equality | S1T1 |
| 脖子和腿长度一样，头不是最长。 | 1 | equality, negation | S2T9 |
| 脖子和腿长度相等，头不是最长。 | 1 | equality, negation | S1T289 |
| 脖子和腿长度相近。 | 1 | equality | S1T95 |
| 脖子和躯干一样长，头最短，腿和尾巴长于躯干。 | 1 | equality, body_ref | S2T110 |
| 脖子和躯干一样长，腿和尾巴一样长，头最短。 | 1 | equality, body_ref | S2T97 |
| 脖子很短，腿和头差不多长，尾巴最长。 | 1 | equality | S1T6 |
| 脖子是最长，尾巴和躯干一样长，头和腿短于躯干。 | 1 | equality, body_ref | S2T48 |
| 脖子最短，头略长于脖子，尾巴和腿一样。 | 1 | equality | S1T40 |
| 脖子最长，头不是最长。 | 1 | negation | S2T8 |
| 脖子最长，头和尾巴差不多长，腿最短。 | 1 | equality | S1T24 |
| 脖子比尾巴短，头和脖子一样短。 | 1 | equality | S2T173 |
| 脖子比尾巴长，头和尾巴一样长。 | 1 | equality | S2T172 |
| 脖子比腿短，头和腿一样长。 | 1 | equality | S2T152 |
| 脖子比腿短，头和躯干一样长。 | 1 | equality, body_ref | S2T139 |
| 脖子比腿长，头和尾巴一样长，且都是四个部位中最长。 | 1 | equality | S2T143 |
| 脖子比腿长，头比尾巴长，头和脖子差不多长。 | 1 | equality | S2T146 |
| 脖子比腿长，脖子和尾巴一样长。 | 1 | equality | S2T137 |
| 脖子比腿长，腿和尾巴一样长。 | 1 | equality | S2T138 |
| 脖子长于躯干。 | 1 | body_ref | S2T124 |
| 腿、尾巴和头一样短于躯干，脖子最长。 | 1 | equality, body_ref | S2T75 |
| 腿中等偏长，头和脖子一样长。 | 1 | equality | S2T231 |
| 腿和尾巴一样长，且头不是最长。 | 1 | equality, negation | S2T31 |
| 腿和尾巴短于躯干，脖子和头长于躯干。 | 1 | body_ref | S2T103 |
| 腿和尾巴长度相等，头略长于脖子。 | 1 | equality | S1T91 |
| 腿和尾巴长度相近，头和脖子长度相近。 | 1 | equality | S1T90 |
| 腿和躯干一样长。 | 1 | equality, body_ref | S2T176 |
| 腿和躯干一样长，头、脖子、尾巴都短于躯干。 | 1 | equality, body_ref | S2T55 |
| 腿和躯干一样长，头和尾巴一样长，脖子是最长。 | 1 | equality, body_ref | S2T58 |
| 腿和躯干一样长，脖子和躯干一样长，尾巴最长，头略短于躯干。 | 1 | equality, body_ref | S2T82 |
| 腿最短，其余部位较长且不相等。 | 1 | equality | S1T89 |
| 腿最短，尾巴第二，头最长，头比脖子长一些。 | 1 | ranking | S1T16 |
| 腿最短，脖子和头差不多长，尾巴最长。 | 1 | equality | S1T21 |
| 腿比尾巴短，脖子和尾巴一样长。 | 1 | equality | S2T178 |
| 腿比躯干短，头和尾巴一样长。 | 1 | equality, body_ref | S2T113 |
| 腿比躯干短，头和脖子一样长。 | 1 | equality, body_ref | S2T198 |
| 腿比躯干长，头最长。 | 1 | body_ref | S2T200 |
| 腿比躯干长，头长于脖子。 | 1 | body_ref | S2T199 |
| 腿短于躯干，头最长。 | 1 | body_ref | S2T116 |
| 腿短于躯干，脖子最长。 | 1 | body_ref | S2T117 |
| 腿短，头、脖子、尾巴长于躯干。 | 1 | body_ref | S3T71 |
| 腿短，头和尾巴比躯干短。 | 1 | body_ref | S3T68 |
| 腿短，头长，脖子和尾巴一样长。 | 1 | equality | S2T301 |
| 腿短，尾巴和躯干一样长。 | 1 | equality, body_ref | S2T283 |
| 腿短，脖子比躯干短。 | 1 | body_ref | S2T292 |
| 腿长，头、脖子、尾巴都不一样长。 | 1 | equality, negation | S3T89 |
| 腿长，头和尾巴比躯干短。 | 1 | body_ref | S3T69 |
| 腿长，脖子和躯干一样长。 | 1 | equality, body_ref | S2T303 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头最长。 | 7 | 0.286 | superlative:头 > 脖子; superlative:头 > 尾巴; superlative:头 > 腿 | S1T190, S1T253, S1T256, S1T275, S1T277, S1T281, S1T319 |
| 头中等偏长。 | 3 | 0.000 | absolute_long:头 > 0.50 | S2T217, S2T218, S2T220 |
| 脖子和躯干一样长。 | 3 | 0.000 | body_ref:脖子 = 0.50 | S2T102, S2T206, S2T208 |
| 头和尾巴长。 | 2 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T69, S1T169 |
| 头和尾巴长度相等。 | 2 | 0.000 | equality_range:头+尾巴 = | S1T219, S1T229 |
| 头、尾巴、脖子长度相等。 | 1 | 0.000 | equality_range:头+尾巴+脖子 = | S1T149 |
| 头和尾巴一样长。 | 1 | 0.000 | equality_range:头+尾巴 = | S1T299 |
| 头和脖子长度相等。 | 1 | 0.000 | equality_range:头+脖子 = | S1T77 |
| 头比腿长，四个部位长度不相等。 | 1 | 0.000 | comparison:头 > 腿 | S2T158 |
| 头比躯干短。 | 1 | 0.000 | body_ref:头 < 0.50 | S2T191 |
| 尾巴和腿长度相等。 | 1 | 0.000 | equality_range:尾巴+腿 = | S1T84 |
| 短于躯干，头和躯干一样长。 | 1 | 0.000 | body_ref:头 = 0.50 | S2T203 |
| 短于躯干，尾巴和躯干一样长。 | 1 | 0.000 | body_ref:尾巴 = 0.50 | S2T202 |
| 脖子、腿、尾巴长度相等。 | 1 | 0.000 | equality_range:脖子+腿+尾巴 = | S1T260 |
| 脖子和头长。 | 1 | 0.000 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50 | S1T63 |
| 脖子和尾巴一样长。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T135 |
| 脖子和尾巴一样长，头不是最长。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S2T18 |
| 脖子和尾巴长度相等。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T194 |
| 脖子和腿长度一样，头不是最长。 | 1 | 0.000 | equality_range:脖子+腿 = | S2T9 |
| 脖子和腿长度相等。 | 1 | 0.000 | equality_range:脖子+腿 = | S1T251 |
| 脖子和腿长度相等，头不是最长。 | 1 | 0.000 | equality_range:脖子+腿 = | S1T289 |
| 脖子和腿长度相近。 | 1 | 0.000 | equality_range:脖子+腿 = | S1T95 |
| 脖子短于躯干。 | 1 | 0.000 | body_ref:脖子 < 0.50 | S2T123 |
| 腿和尾巴长度相等。 | 1 | 0.000 | equality_range:腿+尾巴 = | S1T192 |
| 腿和躯干一样长。 | 1 | 0.000 | body_ref:腿 = 0.50 | S2T176 |
| 腿长，脖子比躯干短。 | 1 | 0.000 | absolute_long:腿 > 0.50; body_ref:脖子 < 0.50 | S2T293 |
| 脖子和躯干一样长，其他三个部位都短于躯干。 | 1 | 0.250 | body_ref:脖子 = 0.50; body_ref:腿 < 0.50; body_ref:尾巴 < 0.50 | S2T49 |
| 头和尾巴一样长，四个部位都短于躯干。 | 1 | 0.333 | body_ref:头 < 0.50; body_ref:尾巴 < 0.50 | S2T93 |
| 头、脖子和腿和躯干一样长，头最短。 | 1 | 0.429 | body_ref:头 = 0.50; body_ref:脖子 = 0.50; body_ref:腿 = 0.50; equality_range:头+脖子+腿 = | S2T80 |

### S323

- trial 数: 256; 非空文本: 254; fidelity 可评分率: 0.980; 平均 fidelity: 0.931; 完全忠实率: 0.820; 低 fidelity 率: 0.023.
- 旧版 region 覆盖率: 0.980; 旧版 region 有未处理片段率: 0.012.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 251 | 0.980 |
| comparison | 70 | 0.273 |
| meta | 3 | 0.012 |
| empty | 2 | 0.008 |
| count_abstract | 1 | 0.004 |
| negation | 1 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴短，腿短。 | 40 |
| 尾巴长，头短。 | 34 |
| 尾巴长，头长。 | 28 |
| 尾巴短，腿长。 | 27 |
| 尾巴和头都比较长。 | 5 |
| 腿短，尾巴长。 | 4 |
| 选错了。 | 3 |
| 四个部位都比较长。 | 3 |
| 尾巴和腿都比较短。 | 2 |
| 腿很短，尾巴长。 | 2 |
| 腿稍微长点，其他部位都比较短。 | 2 |
| 尾巴很长，头比较短。 | 2 |
| 腿很短，尾巴也比较短。 | 2 |
| 腿很长，尾巴比较短，其他部位更短。 | 2 |
| 头比较短，其他部位都是中等长度。 | 2 |
| 四个部位都很短。 | 2 |
| 腿和头都很长。 | 2 |
| 尾巴长，腿短。 | 2 |
| 头和腿都比较短，其他部位中等长度。 | 2 |
| 尾巴很短，其他部位都很长。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 选错了。 | 3 | meta | S1T146, S1T156, S1T157 |
| 只有头比较短，其他三个部位都是中等偏长。 | 1 | count_abstract | S1T32 |
| 腿很短，头和脖子很长，尾巴不是特别长。 | 1 | negation | S1T47 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位都比较短。 | 1 | 0.000 | absolute_short:脖子 < 0.50; absolute_short:头 < 0.50; absolute_short:腿 < 0.50; absolute_short:尾巴 < 0.50 | S1T33 |
| 头和脖子都比较长。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T13 |
| 尾巴略微长一点。 | 1 | 0.000 | absolute_long:尾巴 > 0.50 | S1T16 |
| 尾巴短，腿短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50 | S1T153 |
| 尾巴特别短，其他都比较长。 | 1 | 0.250 | complement:脖子 > 0.50; complement:头 > 0.50; complement:腿 > 0.50 | S1T17 |
| 腿比较短，头和尾巴略微长一点点。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T38 |

### S324

- trial 数: 512; 非空文本: 512; fidelity 可评分率: 1.000; 平均 fidelity: 0.941; 完全忠实率: 0.869; 低 fidelity 率: 0.008.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.008.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 512 | 1.000 |
| comparison | 16 | 0.031 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴较长，腿较短。 | 52 |
| 尾巴较短，头较长。 | 45 |
| 尾巴较短，头较短。 | 44 |
| 尾巴短，头长。 | 42 |
| 尾巴长，腿短。 | 36 |
| 尾巴短，头短。 | 32 |
| 尾巴较长，腿较长。 | 30 |
| 尾巴长，腿长。 | 24 |
| 尾巴较长，头较长。 | 12 |
| 尾巴较短，腿较短。 | 12 |
| 尾巴较长，头较短。 | 10 |
| 尾巴较短，腿较长。 | 9 |
| 尾巴较长，脖子较短。 | 8 |
| 四个部位长度中等。 | 6 |
| 尾巴较短，脖子较长。 | 4 |
| 尾巴短，腿短。 | 3 |
| 头和脖子较长，腿和尾巴较短。 | 3 |
| 腿和脖子较长，头和尾巴长度中等。 | 3 |
| 头和脖子较长，腿和尾巴中等。 | 3 |
| 头和腿较长，脖子和尾巴较短。 | 2 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴较短，头较短。 | 2 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50 | S2T97, S2T112 |
| 尾巴短，头短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:头 < 0.50 | S2T48 |
| 尾巴短，腿短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_short:腿 < 0.50 | S1T312 |

### S325

- trial 数: 512; 非空文本: 512; fidelity 可评分率: 0.992; 平均 fidelity: 0.925; 完全忠实率: 0.775; 低 fidelity 率: 0.014.
- 旧版 region 覆盖率: 0.992; 旧版 region 有未处理片段率: 0.008.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 512 | 1.000 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴长，头短。 | 38 |
| 尾巴长，头长。 | 36 |
| 尾巴短，腿长。 | 26 |
| 尾巴短，腿短。 | 25 |
| 四个都长。 | 16 |
| 只有腿短，其他都长。 | 14 |
| 只有脖子短，其他都长。 | 13 |
| 只有头短，其他都长。 | 12 |
| 只有脖子长，其他都短。 | 12 |
| 只有腿短。 | 11 |
| 只有腿长。 | 9 |
| 只有尾巴短，其他都长。 | 9 |
| 头长，脖子短，腿短，尾巴短。 | 8 |
| 只有脖子和尾巴长，其他都短。 | 8 |
| 只有脖子短。 | 8 |
| 只有头长，其他都短。 | 8 |
| 只有尾巴长，其他都短。 | 8 |
| 头长，脖子长，腿短，尾巴短。 | 7 |
| 头短，腿短，脖子长，尾巴长。 | 6 |
| 只有头和脖子长，其他都短。 | 6 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 只有腿短，其他都长。 | 2 | 0.429 | exclusive_case:头 > 0.50; complement:头 > 0.50; exclusive_case:尾巴 > 0.50; complement:尾巴 > 0.50 | S1T187, S2T23 |
| 只有脖子短。 | 1 | 0.000 | exclusive_case:脖子 < 0.50; exclusive_case:头 > 0.50; exclusive_case:腿 > 0.50; exclusive_case:尾巴 > 0.50 | S1T136 |
| 尾巴长，头长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:头 > 0.50 | S2T88 |
| 尾巴长，腿也长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:腿 > 0.50 | S1T146 |
| 头长，脖子短，腿长，尾巴长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T71 |
| 头长，脖子长，腿长，尾巴长。 | 1 | 0.250 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T81 |

### S326

- trial 数: 512; 非空文本: 503; fidelity 可评分率: 0.951; 平均 fidelity: 0.864; 完全忠实率: 0.721; 低 fidelity 率: 0.043.
- 旧版 region 覆盖率: 0.951; 旧版 region 有未处理片段率: 0.100.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 483 | 0.943 |
| comparison | 68 | 0.133 |
| body_ref | 55 | 0.107 |
| equality | 21 | 0.041 |
| superlative | 21 | 0.041 |
| other | 15 | 0.029 |
| empty | 9 | 0.018 |
| ranking | 3 | 0.006 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿长，头长。 | 42 |
| 腿长，头短。 | 40 |
| 腿短，尾巴长。 | 36 |
| 腿短，尾巴短。 | 24 |
| 腿短。 | 23 |
| 腿短，脖子长。 | 21 |
| 尾巴长，头长。 | 18 |
| 腿短，头长。 | 17 |
| 腿长。 | 10 |
| 腿长，尾巴长。 | 9 |
| 腿短，头短。 | 8 |
| 腿长，头比躯干低。 | 8 |
| 尾巴和头长。 | 8 |
| 尾巴最长。 | 7 |
| 腿长，脖子长。 | 7 |
| 腿短，脖子短。 | 7 |
| 头长，尾巴长。 | 6 |
| 假设脖子无关。 | 6 |
| 腿和脖子长。 | 5 |
| 头短，腿长。 | 4 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿长，头比躯干低。 | 8 | body_ref | S1T51, S1T56, S1T57, S1T58, S1T59, S1T61, S1T63, S1T64 |
| 假设脖子无关。 | 6 | other | S1T243, S1T244, S1T245, S1T246, S1T247, S1T248 |
| 朝右。 | 4 | other | S1T81, S1T82, S1T83, S1T84 |
| 头比躯干高，腿长。 | 3 | body_ref | S1T194, S1T196, S1T206 |
| 腿短，头比躯干高。 | 3 | body_ref | S1T52, S1T60, S1T62 |
| 假设尾巴无关。 | 2 | other | S1T241, S1T242 |
| 头比躯干低，尾巴比脖子高。 | 2 | body_ref | S1T203, S1T204 |
| 尾巴长，头比躯干高。 | 2 | body_ref | S1T119, S1T120 |
| 腿和躯干差不多，头比脖子长。 | 2 | equality, body_ref | S1T32, S1T33 |
| 腿比尾巴短，头在躯干以下。 | 2 | body_ref | S1T125, S1T126 |
| 腿长，头比躯干高。 | 2 | body_ref | S1T53, S1T55 |
| 分布均匀，像猫。 | 1 | equality | S1T222 |
| 头和脖子一样长，尾巴也差不多长，头在躯干以下。 | 1 | equality, body_ref | S1T8 |
| 头和脖子差不多长，尾巴短。 | 1 | equality | S1T7 |
| 头和脖子很长，尾巴很短，头在躯干以上。 | 1 | body_ref | S1T9 |
| 头和脖子比尾巴长，头在躯干以下。 | 1 | body_ref | S1T11 |
| 头在躯干以上，腿长，尾巴短。 | 1 | body_ref | S1T40 |
| 头在躯干以下，尾巴和腿差不多。 | 1 | equality, body_ref | S1T47 |
| 头很长，脖子、尾巴很短，头在躯干以下。 | 1 | body_ref | S1T10 |
| 头比尾巴长，头比躯干低。 | 1 | body_ref | S1T65 |
| 头比躯干低，头较长。 | 1 | body_ref | S1T49 |
| 头比躯干低，腿短。 | 1 | body_ref | S1T195 |
| 头比躯干低，腿长。 | 1 | body_ref | S1T193 |
| 头比躯干高，头较短。 | 1 | body_ref | S1T50 |
| 头比躯干高，尾巴比脖子长。 | 1 | body_ref | S1T117 |
| 头比躯干高，尾巴长。 | 1 | body_ref | S1T181 |
| 头比较短，脖子和尾巴一样长。 | 1 | equality | S1T13 |
| 头长，尾巴短，脖子短，头在躯干以下。 | 1 | body_ref | S1T6 |
| 头长，尾巴短，脖子长，头在躯干以上。 | 1 | body_ref | S1T5 |
| 头长，脖子短，尾巴短，头在躯干以下、腿以上。 | 1 | body_ref | S1T1 |
| 头长，脖子长，尾巴短，头在躯干和腿之间，右朝向。 | 1 | body_ref | S1T3 |
| 尾巴和头差不多长。 | 1 | equality | S1T182 |
| 尾巴和躯干一样长。 | 1 | equality, body_ref | S1T160 |
| 尾巴比脖子高，头比躯干高。 | 1 | body_ref | S1T205 |
| 尾巴长，头在躯干和腿之间。 | 1 | body_ref | S1T122 |
| 左朝向，头在躯干以下、腿以上，脖子和尾巴一般。 | 1 | body_ref | S1T2 |
| 朝左。 | 1 | other | S1T80 |
| 狗状。 | 1 | other | S1T295 |
| 脖子和头一样长，头在躯干以上。 | 1 | equality, body_ref | S1T16 |
| 脖子和头差不多长，尾巴和脖子差不多长。 | 1 | equality | S1T114 |
| 脖子和尾巴一样长，头在躯干以下。 | 1 | equality, body_ref | S1T15 |
| 脖子和尾巴一样长，头很短。 | 1 | equality | S1T12 |
| 脖子长，头和尾巴一样长。 | 1 | equality | S1T14 |
| 脖子长，头和腿差不多。 | 1 | equality | S1T95 |
| 腿和尾巴、脖子一样长，头比较短。 | 1 | equality | S1T26 |
| 腿和尾巴差不多长，脖子比头长。 | 1 | equality | S1T30 |
| 腿比尾巴短，头比躯干高。 | 1 | body_ref | S1T66 |
| 腿比较短，头在躯干以下。 | 1 | body_ref | S1T48 |
| 腿比较长，头在躯干以上，尾巴最长。 | 1 | body_ref | S1T39 |
| 腿比较长，头在躯干以下。 | 1 | body_ref | S1T38 |
| 腿短 尾巴次之，脖子、头最长。 | 1 | ranking | S1T35 |
| 腿短，头在躯干以上，尾巴长。 | 1 | body_ref | S1T46 |
| 腿短，尾巴和脖子差不多长，头短。 | 1 | equality | S1T202 |
| 腿短，尾巴次之，脖子、头长。 | 1 | ranking | S1T36 |
| 腿长，四个部位都差不多长。 | 1 | equality | S1T265 |
| 腿长，头和躯干一样高。 | 1 | equality, body_ref | S1T54 |
| 腿长，头在躯干以下，尾巴长。 | 1 | body_ref | S1T41 |
| 腿长，头在躯干以下，脖子、尾巴长。 | 1 | body_ref | S1T42 |
| 腿长，尾巴和腿一样长。 | 1 | equality | S1T27 |
| 腿长，脖子最短，然后是头和尾巴。 | 1 | ranking | S1T21 |
| 蜥蜴状。 | 1 | other | S1T294 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短。 | 3 | 0.000 | absolute_short:腿 < 0.50 | S2T7, S2T21, S2T26 |
| 头在躯干以下，尾巴和腿差不多。 | 1 | 0.000 | equality_range:尾巴+腿 = | S1T47 |
| 头比躯干高，腿长。 | 1 | 0.000 | body_ref:头 > 0.50; absolute_long:腿 > 0.50 | S1T196 |
| 头短。 | 1 | 0.000 | absolute_short:头 < 0.50 | S1T292 |
| 头短，腿长。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_long:腿 > 0.50 | S1T239 |
| 尾巴和头差不多长。 | 1 | 0.000 | equality_range:尾巴+头 = | S1T182 |
| 尾巴和躯干一样长。 | 1 | 0.000 | body_ref:尾巴 = 0.50 | S1T160 |
| 尾巴短，脖子长。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; absolute_long:脖子 > 0.50 | S1T287 |
| 尾巴长，脖子长。 | 1 | 0.000 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50 | S1T284 |
| 脖子和头一样长，头在躯干以上。 | 1 | 0.000 | equality_range:脖子+头 = | S1T16 |
| 脖子和头差不多长，尾巴和脖子差不多长。 | 1 | 0.000 | equality_range:脖子+头 =; equality_range:尾巴+脖子 = | S1T114 |
| 脖子和尾巴一样长，头在躯干以下。 | 1 | 0.000 | equality_range:脖子+尾巴 = | S1T15 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T148 |
| 腿长。 | 1 | 0.000 | absolute_long:腿 > 0.50 | S2T25 |
| 腿长，头比躯干低。 | 1 | 0.000 | absolute_long:腿 > 0.50; body_ref:头 < 0.50 | S1T63 |
| 腿长，头长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S1T86 |
| 腿长，尾巴长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:尾巴 > 0.50 | S1T313 |
| 腿长，脖子长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S1T304 |
| 头比较短，脖子、尾巴长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T207 |
| 尾巴、脖子、头较短。 | 1 | 0.333 | absolute_short:尾巴 < 0.50; absolute_short:脖子 < 0.50 | S1T19 |

### S327

- trial 数: 128; 非空文本: 128; fidelity 可评分率: 1.000; 平均 fidelity: 0.951; 完全忠实率: 0.883; 低 fidelity 率: 0.000.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.008.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 128 | 1.000 |
| comparison | 8 | 0.062 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，尾巴短。 | 24 |
| 头短，腿长。 | 19 |
| 头短，腿短。 | 19 |
| 头长，尾巴长。 | 7 |
| 头短，尾巴长。 | 4 |
| 头短，腿很长。 | 3 |
| 头长，尾巴较长。 | 3 |
| 头和尾巴长。 | 2 |
| 头短，尾巴短。 | 2 |
| 头短，腿长，尾巴短。 | 2 |
| 头、脖子、腿长度中等，尾巴较短。 | 1 |
| 脖子比较长，头、腿和尾巴都处于中等或中等偏下。 | 1 |
| 脖子比较长，头、腿和尾巴都中等。 | 1 |
| 头、脖子、腿长度中等，尾巴较长。 | 1 |
| 头长，脖子较短，腿和尾巴中等。 | 1 |
| 头和腿较短。 | 1 |
| 头、脖子、腿长度中等，尾巴比较长。 | 1 |
| 头、脖子较长，腿长度中等，尾巴也较长。 | 1 |
| 头、脖子、腿和尾巴都较短，脖子和尾巴略长于头和腿。 | 1 |
| 头较长，脖子和腿长度中等，尾巴较短。 | 1 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
无。

### S328

- trial 数: 704; 非空文本: 702; fidelity 可评分率: 0.997; 平均 fidelity: 0.915; 完全忠实率: 0.780; 低 fidelity 率: 0.009.
- 旧版 region 覆盖率: 0.997; 旧版 region 有未处理片段率: 0.009.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 702 | 0.997 |
| comparison | 440 | 0.625 |
| equality | 35 | 0.050 |
| group_sum | 11 | 0.016 |
| negation | 4 | 0.006 |
| empty | 2 | 0.003 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 尾巴长，头比脖子长。 | 39 |
| 尾巴短，头比脖子长。 | 38 |
| 尾巴长，头比脖子短。 | 34 |
| 尾巴短，头比脖子短。 | 34 |
| 尾巴长，头加脖子长。 | 33 |
| 尾巴长，头长。 | 33 |
| 尾巴长，头比腿短。 | 32 |
| 尾巴长，头短。 | 31 |
| 尾巴长，头比腿长。 | 30 |
| 尾巴短，脖子长。 | 29 |
| 尾巴短，脖子比腿长。 | 23 |
| 尾巴短，脖子比腿短。 | 23 |
| 尾巴短，脖子短。 | 19 |
| 尾巴长，头加脖子短。 | 15 |
| 尾巴短，头加脖子长。 | 12 |
| 尾巴长，脖子比腿短。 | 9 |
| 尾巴短，头加脖子短。 | 8 |
| 尾巴短，头和脖子比腿短。 | 7 |
| 尾巴短，脖子和腿一样长。 | 5 |
| 尾巴长，头和脖子一样长。 | 5 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 尾巴短，脖子和腿一样长。 | 5 | equality | S2T132, S2T141, S2T153, S2T163, S2T167 |
| 尾巴长，头和脖子一样长。 | 5 | equality | S2T91, S2T96, S2T117, S2T122, S2T151 |
| 尾巴短，头和脖子差不多长。 | 4 | equality | S1T226, S2T8, S2T20, S2T34 |
| 尾巴比腿短，头和脖子加起来长。 | 2 | group_sum | S1T54, S1T120 |
| 尾巴短，头和脖子一样长。 | 2 | equality | S2T50, S2T56 |
| 尾巴短，头和脖子差不多。 | 2 | equality | S1T159, S1T310 |
| 尾巴不算短，尾巴比腿短，头和脖子加在一起比较长。 | 1 | negation | S1T24 |
| 尾巴和腿差不多长，头加脖子很长。 | 1 | equality | S1T63 |
| 尾巴和腿差不多长，尾巴比腿稍长一点，头比脖子长。 | 1 | equality | S1T66 |
| 尾巴和腿差不多长，尾巴特别长，头和脖子加在一起比较长。 | 1 | equality | S1T26 |
| 尾巴和腿都比较长，差不多，头加脖子比较短。 | 1 | equality | S1T95 |
| 尾巴比腿短，头和脖子加起来特别长，脖子比较长。 | 1 | group_sum | S1T48 |
| 尾巴比腿长，头和脖子加起来比较短。 | 1 | group_sum | S1T55 |
| 尾巴比腿长，头和脖子加起来比较长，脖子比头短。 | 1 | group_sum | S1T47 |
| 尾巴比腿长，头很长，脖子也很长，头和脖子差不多长。 | 1 | equality | S1T35 |
| 尾巴比腿长，尾巴很长，头和脖子差不多。 | 1 | equality | S1T92 |
| 尾巴特别短，尾巴和腿差不多短，头和脖子加在一起特别长。 | 1 | equality | S1T22 |
| 尾巴特别长，尾巴和腿差不多长，头加脖子比较长。 | 1 | equality | S1T21 |
| 尾巴短，脖子和腿差不多。 | 1 | equality | S2T73 |
| 尾巴短，脖子比腿短，头和腿一样长。 | 1 | equality | S2T124 |
| 尾巴短，腿也短，尾巴和腿差不多长，脖子长，头不算长，头和脖子加一块很长。 | 1 | equality, negation | S1T99 |
| 尾巴长，头和腿一样长。 | 1 | equality | S2T217 |
| 尾巴长，头和腿差不多长，脖子比腿长。 | 1 | equality | S2T43 |
| 尾巴长，脖子和腿一样长。 | 1 | equality | S2T49 |
| 尾巴长，腿也长，头加脖子不算特别长。 | 1 | negation | S1T98 |
| 尾巴长，腿长，头和脖子加起来短。 | 1 | group_sum | S1T119 |
| 脖子比头短，脖子加头比较长，尾巴和腿差不多长。 | 1 | equality | S1T45 |
| 腿和尾巴差不多长。 | 1 | equality | S1T50 |
| 腿和尾巴差不多长，都是中等长度，头和脖子都比较长。 | 1 | equality | S1T52 |
| 腿短，头和脖子加起来不算长。 | 1 | group_sum, negation | S1T121 |
| 腿短，头和脖子加起来很长。 | 1 | group_sum | S1T79 |
| 腿短，头和脖子加起来长。 | 1 | group_sum | S1T122 |
| 腿长，头长，脖子短，头和脖子加起来长。 | 1 | group_sum | S1T78 |
| 腿长，尾巴长，头和脖子加起来短。 | 1 | group_sum | S1T124 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴短，脖子和腿差不多。 | 1 | 0.000 | absolute_short:尾巴 < 0.50; equality_range:脖子+腿 = | S2T73 |
| 腿长，尾巴长，脖子长，头短。 | 1 | 0.250 | absolute_long:尾巴 > 0.50; absolute_long:脖子 > 0.50; absolute_short:头 < 0.50 | S1T86 |
| 头加脖子比尾巴短，头比脖子短。 | 1 | 0.333 | group_sum:头+脖子 < 尾巴; comparison:脖子 < 尾巴 | S1T254 |
| 尾巴短，头加脖子长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T260 |
| 尾巴长，头加脖子长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S2T12 |
| 腿和尾巴差不多长，都是中等长度，头和脖子都比较长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T52 |

### S329

- trial 数: 384; 非空文本: 384; fidelity 可评分率: 0.992; 平均 fidelity: 0.921; 完全忠实率: 0.836; 低 fidelity 率: 0.023.
- 旧版 region 覆盖率: 0.992; 旧版 region 有未处理片段率: 0.010.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 382 | 0.995 |
| ranking | 10 | 0.026 |
| comparison | 8 | 0.021 |
| superlative | 5 | 0.013 |
| other | 2 | 0.005 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，尾巴短。 | 68 |
| 头长，尾巴长。 | 61 |
| 头短，脖子长。 | 46 |
| 头短，脖子短。 | 19 |
| 腿长，脖子短。 | 12 |
| 脖子长。 | 10 |
| 腿长，脖子长。 | 8 |
| 腿长。 | 7 |
| 头长。 | 7 |
| 脖子长，尾巴长。 | 7 |
| 脖子长，其他部分中等长度。 | 5 |
| 尾巴最短。 | 4 |
| 尾巴长。 | 4 |
| 腿长，脖子长，尾巴长，头短。 | 3 |
| 头短，脖子短，腿长。 | 3 |
| 头短，腿长。 | 3 |
| 头长，腿长。 | 3 |
| 头长，尾巴长，脖子较长，腿短。 | 2 |
| 头、脖子、尾巴长，腿短。 | 2 |
| 头长，脖子长，尾巴长，腿短。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 按错了。 | 2 | other | S1T185, S1T251 |
| 头长，其次是脖子和尾巴，最后是腿。 | 1 | ranking | S1T123 |
| 头长，其次是脖子和尾巴，腿短。 | 1 | ranking | S1T162 |
| 尾巴长，其次是腿和脖子，头短。 | 1 | ranking | S1T143 |
| 尾巴长，其次是腿，然后是头和脖子。 | 1 | ranking | S1T87 |
| 脖子、尾巴长，腿其次，头短。 | 1 | ranking | S1T94 |
| 脖子长，头和尾巴其次，腿短。 | 1 | ranking | S1T91 |
| 脖子长，尾巴其次，头和腿短。 | 1 | ranking | S1T112 |
| 腿长，头、脖子其次，尾巴短。 | 1 | ranking | S1T104 |
| 腿长，头和尾巴其次，脖子短。 | 1 | ranking | S1T172 |
| 腿长，尾巴长，头其次，脖子短。 | 1 | ranking | S1T95 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头长，尾巴长。 | 4 | 0.000 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T62, S1T68, S1T238, S2T62 |
| 头长，脖子短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_short:脖子 < 0.50 | S1T129 |
| 头长，腿短。 | 1 | 0.000 | absolute_long:头 > 0.50; absolute_short:腿 < 0.50 | S1T101 |
| 脖子长。 | 1 | 0.000 | absolute_long:脖子 > 0.50 | S1T117 |
| 腿长，脖子长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:脖子 > 0.50 | S1T204 |
| 头长，脖子短，尾巴长。 | 1 | 0.333 | absolute_long:头 > 0.50; absolute_long:尾巴 > 0.50 | S1T257 |

### S330

- trial 数: 256; 非空文本: 255; fidelity 可评分率: 0.996; 平均 fidelity: 0.956; 完全忠实率: 0.910; 低 fidelity 率: 0.008.
- 旧版 region 覆盖率: 0.996; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 255 | 0.996 |
| superlative | 28 | 0.109 |
| comparison | 11 | 0.043 |
| body_ref | 2 | 0.008 |
| empty | 1 | 0.004 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 腿短，脖子短。 | 41 |
| 腿长，头长。 | 34 |
| 腿短，脖子长。 | 34 |
| 腿长，头短。 | 33 |
| 腿长。 | 13 |
| 腿短。 | 11 |
| 头最长。 | 10 |
| 腿长，脖子长。 | 8 |
| 尾巴长。 | 6 |
| 腿长，脖子短。 | 6 |
| 腿最长。 | 4 |
| 头长。 | 4 |
| 脖子长。 | 4 |
| 尾巴最长。 | 4 |
| 尾巴最短。 | 3 |
| 脖子比头长。 | 3 |
| 腿短，尾巴长。 | 3 |
| 腿长，尾巴短。 | 3 |
| 头最短。 | 3 |
| 腿短，头长。 | 2 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 腿长，头比躯干短。 | 2 | body_ref | S1T141, S1T149 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头长。 | 1 | 0.000 | absolute_long:头 > 0.50 | S1T13 |
| 头最长。 | 1 | 0.333 | superlative:头 > 脖子; superlative:头 > 尾巴 | S1T52 |

### S331

- trial 数: 640; 非空文本: 612; fidelity 可评分率: 0.955; 平均 fidelity: 0.890; 完全忠实率: 0.787; 低 fidelity 率: 0.053.
- 旧版 region 覆盖率: 0.955; 旧版 region 有未处理片段率: 0.005.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 604 | 0.944 |
| empty | 28 | 0.044 |
| superlative | 19 | 0.030 |
| equality | 14 | 0.022 |
| body_ref | 1 | 0.002 |
| other | 1 | 0.002 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头长，脖子长。 | 90 |
| 头长，脖子短。 | 64 |
| 头短，尾巴长。 | 61 |
| 头短，尾巴短。 | 43 |
| 头长。 | 39 |
| 脖子长。 | 38 |
| 腿长。 | 33 |
| 头长，腿长。 | 16 |
| 头长，尾巴长。 | 16 |
| 腿长，脖子长。 | 13 |
| 腿长，头长。 | 13 |
| 头短，脖子长。 | 11 |
| 脖子长，头长。 | 11 |
| 尾巴长。 | 10 |
| 脖子长，腿长。 | 10 |
| 腿最长。 | 10 |
| 头短。 | 9 |
| 腿长，尾巴长。 | 8 |
| 头短，脖子短。 | 7 |
| 脖子长，尾巴长。 | 7 |

非常规风格对应试次：
| text | count | nonstandard_styles | sample_trials |
| --- | --- | --- | --- |
| 四个部位很均衡。 | 4 | equality | S1T33, S1T35, S1T38, S1T52 |
| 四个部位均衡。 | 3 | equality | S1T10, S1T57, S1T96 |
| 腿和头一样长。 | 2 | equality | S1T28, S1T40 |
| 四个部位均衡，腿最长。 | 1 | equality | S1T58 |
| 均等。 | 1 | other | S1T154 |
| 头和腿差不多长。 | 1 | equality | S1T53 |
| 腿、头和脖子一样长。 | 1 | equality | S1T43 |
| 腿短，四个部位均衡。 | 1 | equality | S1T224 |
| 腿长，四个部位均衡。 | 1 | equality | S1T32 |
| 躯干长，头短。 | 1 | body_ref | S1T29 |

低忠实率对应试次（fidelity < 0.5）：
| text | count | mean_fidelity | sample_failed_claims | sample_trials |
| --- | --- | --- | --- | --- |
| 头长。 | 6 | 0.000 | absolute_long:头 > 0.50 | S1T140, S1T219, S1T278, S2T41, S2T52, S2T153 |
| 脖子长。 | 4 | 0.000 | absolute_long:脖子 > 0.50 | S1T245, S1T269, S1T270, S1T274 |
| 四个部位均衡。 | 3 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T10, S1T57, S1T96 |
| 四个部位很均衡。 | 3 | 0.000 | equality_range:脖子+头+腿+尾巴 = | S1T35, S1T38, S1T52 |
| 头长，脖子长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:脖子 > 0.50 | S1T171, S2T195, S2T246 |
| 头长，腿长。 | 3 | 0.000 | absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T207, S1T265, S1T315 |
| 尾巴长。 | 3 | 0.000 | absolute_long:尾巴 > 0.50 | S1T299, S1T300, S1T303 |
| 头短。 | 2 | 0.000 | absolute_short:头 < 0.50 | S1T151, S2T45 |
| 头和腿差不多长。 | 1 | 0.000 | equality_range:头+腿 = | S1T53 |
| 头短，尾巴长。 | 1 | 0.000 | absolute_short:头 < 0.50; absolute_long:尾巴 > 0.50 | S2T176 |
| 尾巴短。 | 1 | 0.000 | absolute_short:尾巴 < 0.50 | S2T17 |
| 腿、头和脖子一样长。 | 1 | 0.000 | equality_range:腿+头+脖子 = | S1T43 |
| 腿长，头长。 | 1 | 0.000 | absolute_long:腿 > 0.50; absolute_long:头 > 0.50 | S1T141 |
| 四个部位都长。 | 1 | 0.250 | absolute_long:脖子 > 0.50; absolute_long:头 > 0.50; absolute_long:腿 > 0.50 | S1T90 |
| 脖子、头长，尾巴长。 | 1 | 0.333 | absolute_long:脖子 > 0.50; absolute_long:尾巴 > 0.50 | S1T62 |

### S332

- trial 数: 128; 非空文本: 128; fidelity 可评分率: 1.000; 平均 fidelity: 0.975; 完全忠实率: 0.945; 低 fidelity 率: 0.000.
- 旧版 region 覆盖率: 1.000; 旧版 region 有未处理片段率: 0.000.

汇报风格标签：
| style | count | rate |
| --- | --- | --- |
| direct_absolute | 128 | 1.000 |
| comparison | 1 | 0.008 |

典型说法 Top 20：
| text | count |
| --- | --- |
| 头较长，脖子较短。 | 8 |
| 脖子较长，腿较短。 | 8 |
| 头较短，脖子短。 | 8 |
| 脖子长，腿长。 | 7 |
| 脖子较长，腿较长。 | 6 |
| 头长，脖子短。 | 6 |
| 头较短，脖子较短。 | 6 |
| 头长，脖子较短。 | 4 |
| 脖子长，腿较长。 | 4 |
| 脖子长，腿较短。 | 3 |
| 头短，脖子短。 | 3 |
| 头较长，脖子短。 | 3 |
| 脖子较长，腿长。 | 2 |
| 脖子长，腿短。 | 2 |
| 头中等长度，脖子短，腿较短。 | 2 |
| 头中等偏长，脖子较长，腿中等偏短。 | 1 |
| 头较长，脖子短，腿中等长度。 | 1 |
| 头短，脖子中等偏短，腿长。 | 1 |
| 头较短，脖子短，腿短。 | 1 |
| 头长，脖子短，腿长。 | 1 |

非常规风格对应试次：
无。

低忠实率对应试次（fidelity < 0.5）：
无。

