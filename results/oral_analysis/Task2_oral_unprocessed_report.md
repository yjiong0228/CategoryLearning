# Task2 未编码文本分析报告

## 说明

- 本报告只分析 `Task2_oral_center_intermediate.csv` 与 `Task2_oral_region_intermediate.csv` 中 `un_pro` 非空的 trial。
- `un_pro` 表示当前 parser/encoder 没有正式编码的文本片段；同一 trial 可能已有部分内容被编码。
- 目前 center 与 region 共享同一个语义 parser，因此二者的 `un_pro` 理论上应高度一致；差异只应来自后续投影策略。

## 总览

| mode | unprocessed_trials | subjects | unique_texts | top_categories |
| --- | --- | --- | --- | --- |
| center | 2411 | 82 | 1468 | other_unparsed:1389; count_abstract:358; body_geometry:221; global_balance:200; vague_size:132; meta_or_uncertain:74; proportion_or_ratio:62; disjoint_inequality:43 |
| region | 2411 | 82 | 1468 | other_unparsed:1389; count_abstract:358; body_geometry:221; global_balance:200; vague_size:132; meta_or_uncertain:74; proportion_or_ratio:62; disjoint_inequality:43 |

## Center 与 Region 差异

本轮 center 与 region 的 `un_pro` trial 清单完全一致，说明残留主要来自共享语义解析层，而不是投影到 center 或 region 的差异。

## Center 未编码分析

### 被试摘要

| iSub | unprocessed_trials | unique_texts | top_categories | top_texts |
| --- | --- | --- | --- | --- |
| 102 | 49 | 38 | other_unparsed:45; count_abstract:2; meta_or_uncertain:1; global_balance:1 | 脖子比头长，但是是最短的两个。:4; 脖子比头短，但是是最长的两个。:3; 脖子比头短，都短于腿。:3; 脖子比头短，它们两个是最长的。:3; 脖子比头短，都长于腿。:2 |
| 103 | 12 | 12 | other_unparsed:12 | 尾巴很长，长于头和脖子，腿较短。:1; 腿极长，尾巴中等长度，长于头和脖子。:1; 腿很长，脖子和头中等长度，长于较短的尾巴。:1; 腿较短，尾巴较长，长于头和脖子。:1; 腿很短，脖子很长，长于尾巴，尾巴和头中等长度。:1 |
| 104 | 2 | 2 | global_balance:2 | 头长，较均匀。:1; 比较均匀。:1 |
| 105 | 1 | 1 | other_unparsed:1 | 腿短于阈值。:1 |
| 106 | 28 | 18 | body_geometry:20; other_unparsed:7; count_abstract:1 | 躯体下方比上面高。:7; 头离地面近。:2; 下面的部分更高一些。:2; 腿不是最长的。:2; 躯体下方没有上面高。:2 |
| 107 | 5 | 5 | other_unparsed:3; proportion_or_ratio:1; global_balance:1 | 头和尾巴很长，脖子也挺长，但相对短一些，腿很短。:1; 腿、脖子和头比例比较协调，尾巴短。:1; 腿、脖子和头长度差不多，都不算很长。:1; 尾巴、腿、脖子和头长度差不多，都还挺长的。:1; 腿、脖子和头不算特别长，且长度相当。:1 |
| 108 | 21 | 10 | other_unparsed:19; meta_or_uncertain:2 | 脖子不是最长的。:8; 腿比脖子短，比尾巴长。:3; 选错了。:2; 腿比脖子和尾巴短，比头长。:2; 头和脖子显著比腿和尾巴长。:1 |
| 109 | 2 | 2 | other_unparsed:2 | 脖子，头较长，腿一般。:1; 腿较长，脖子一般，头较小。:1 |
| 110 | 1 | 1 | other_unparsed:1 | 头短，脖子腿，尾巴长。:1 |
| 111 | 1 | 1 | other_unparsed:1 | 腿短，尾巴、头、脖子长度相同。:1 |
| 112 | 6 | 6 | other_unparsed:5; disjoint_inequality:1 | 尾巴和腿之间关系。:1; 尾巴比头长一点，比腿和脖子都短。:1; 尾巴跟腿之间的关系以及跟脖子的关系。:1; 尾巴跟脖子不一样长。:1; 脖子和腿之间的关系。:1 |
| 113 | 26 | 23 | other_unparsed:23; meta_or_uncertain:2; global_balance:1 | 脖子长，比头长。:3; 头比脖子长，都很长。:2; 脖子比头长，头比腿长，腿比尾巴长，都有些长。:1; 头最长，尾巴最短，脖子和腿差不多长，都比较长。:1; 尾巴比脖子长，但是都很短，脖子和腿很长，腿特别长。:1 |
| 114 | 3 | 2 | global_balance:2; other_unparsed:1 | 都差不多长。:2; 都不长。:1 |
| 116 | 5 | 5 | other_unparsed:5 | 四个部位都比较长，其中腿最明显。:1; 腿的长度最突出。:1; 脖子的长度很突出。:1; 腿和脖子都比较长，但腿是最明显的。:1; 嗯，脖子比较长。:1 |
| 118 | 65 | 27 | count_abstract:50; other_unparsed:10; extreme_endpoint:3; global_balance:2 | 三个部位很长。:17; 只有一个部位很长。:7; 两个部位很长。:6; 一个部位很长。:5; 只有两个部位很长。:3 |
| 119 | 205 | 36 | other_unparsed:150; count_abstract:52; body_geometry:37 | 头短于中间值。:62; 头长于中间值。:53; 头不是最长。:14; 有一个部位长于躯干。:13; 有三个部位长于躯干。:7 |
| 120 | 29 | 20 | body_geometry:10; other_reference:9; count_abstract:7; other_unparsed:6; global_balance:4 | 有奇数个部位长于躯干。:4; 有偶数个部位长于躯干。:3; 腿明显短于其他部位。:2; 头明显长于其他部位。:2; 脖子和尾巴的长度不一样。:2 |
| 121 | 27 | 20 | body_geometry:12; other_unparsed:8; count_abstract:6; global_balance:1 | 腿不是所有部位里最短的。:4; 头在躯干上方，腿比较长。:3; 腿不是最短的部位。:2; 头不是所有部位里最短的。:2; 头比脖子长一点。腿和尾巴差不多，都很短。:1 |
| 123 | 7 | 7 | other_unparsed:6; count_abstract:1; body_geometry:1 | 腿长，脖子，尾巴、腿都偏短。:1; 腿短，脖，脖子和头长，尾巴短。:1; 头、脖子、腿、尾巴、躯干这些部位的长度进行描述，但不一定用的是这几个词语，且可能会涉及到这几个部位之间的比较，包括大小和长短关系。:1; 腿和头。头短，尾巴和脖子长。:1; 头和脖外，头和腿长。:1 |
| 125 | 16 | 14 | other_unparsed:15; proportion_or_ratio:1 | 腿非常短，几乎是最短。:2; 腿很长，基本是最长。:2; 头长，脖子长，腿较长，尾巴长，脖子比腿长一点，头和脖子占的比重比较大。:1; 头长，脖子长，腿长，尾巴短，头和脖子的比重比较大，脖子比腿长一点。:1; 头长，脖子较短，腿短，尾巴较短，头、脖子、尾巴均比腿长。:1 |
| 127 | 2 | 2 | other_unparsed:2 | 腿和头比较长，脖子后尾巴稍微短，一点。:1; 差不多，都一样长。:1 |
| 129 | 62 | 19 | body_geometry:22; count_abstract:21; disjoint_inequality:15; other_unparsed:10 | 一个部位很长。:10; 头在躯干之下。:9; 尾巴和脖子不一样长。:7; 有两个部位一样长。:6; 低头。:4 |
| 130 | 38 | 10 | other_unparsed:28; proportion_or_ratio:10 | 腿和尾巴都比脖子长。:16; 头身比例比较协调。:5; 尾巴和腿都比脖子短。:5; 头身比例不协调。:4; 腿和尾巴都比脖子短。:2 |
| 131 | 39 | 29 | other_unparsed:23; global_balance:15; vague_size:13 | 体型分布的不均匀。:4; 体型分布的很均匀。:3; 朝左边。:3; 朝右边。:2; 腿比头长，朝右边。:2 |
| 132 | 14 | 4 | other_unparsed:14 | 尾巴不是最短的。:8; 尾巴不是最长的。:3; 尾巴短于某个数值。:2; 尾巴长于某个数值。:1 |
| 202 | 1 | 1 | meta_or_uncertain:1 | 选错了。:1 |
| 203 | 11 | 11 | other_unparsed:8; body_geometry:2; global_balance:1; meta_or_uncertain:1 | 身体各个部位都很匀称。:1; 腿很长，颈部也很短。:1; 腿很长，头很短，脖子和尾巴长度对称。:1; 头和尾巴均长于脖子，腿很短。:1; 头显著短、短于脖子，尾巴很长，腿很短。:1 |
| 204 | 14 | 14 | other_unparsed:7; global_balance:4; vague_size:2; meta_or_uncertain:1; body_geometry:1 | 头、脖子、尾巴都很长，腿比它们稍微短一点。:1; 头和尾巴都比脖子长。:1; 四个部位几乎一样长，都挺长。:1; 脖子很长，头很小。:1; 头很小，尾巴很长，脖子也很长。:1 |
| 205 | 4 | 4 | other_unparsed:3; global_balance:1 | 头、脖子和腿，差不多长。:1; 腿，头最长。:1; 腿，尾巴最长。:1; 腿，脖子最长。:1 |
| 206 | 23 | 21 | other_unparsed:21; global_balance:1; vague_size:1 | 脖子比头长，腿也比尾巴长。:2; 脖子比头短，腿也比尾巴短。:2; 短长。:1; 头和脖子。:1; 脖子很短，腿、尾巴，头很长。:1 |
| 207 | 27 | 23 | vague_size:14; other_unparsed:12; count_abstract:1; global_balance:1 | 四个部位都很长，体型很大。:4; 四个部位长度差不多，都比较长。:2; 尾巴最长，其他三个部位一样长，都比较长。:1; 头和脖子一样长，都是最长的，其他两个比较短。:1; 四个部位都比较长，长度接近。:1 |
| 208 | 22 | 21 | other_unparsed:16; global_balance:4; meta_or_uncertain:1; count_abstract:1 | 四者都很长。:2; 脖子很长，尾巴稍短一些，腿和腿一样长。:1; 头和，四个部位都长。:1; 四者都很短，尤其是腿很短。:1; 腿偏短，脖子、尾巴和头都偏长，一样长。:1 |
| 209 | 28 | 25 | other_unparsed:26; global_balance:2 | 头和尾巴长，脖子和腿短，头明显比脖子长。:2; 头最长，脖子短，腿和尾巴中等，头明显比脖子长。:2; 头最长，腿中等，脖子和尾巴短，头明显比脖子长。:2; 所有部位都是中等偏长，而且长度差不多。:1; 脖子很长，头，中等长度，尾巴和腿比较短。:1 |
| 210 | 121 | 59 | count_abstract:85; other_unparsed:26; body_geometry:7; global_balance:4; ordinal_or_secondary:1 | 有三个部位几乎一样长。:13; 三个部位几乎一样长。:11; 有两个部位几乎一样长。:9; 有三个部位长度一样。:7; 有三个部位一样长。:6 |
| 211 | 6 | 6 | other_unparsed:6 | 头最长，比脖子和尾巴都长。:1; 头，和腿基本一样长。:1; 头和脖子，尾巴和腿中有三个是一样长。:1; 头和尾巴都比较长，一样长，腿也比较长。:1; 四个部位长度差不多，比较长。:1 |
| 212 | 136 | 47 | other_unparsed:103; global_balance:32; vague_size:1 | 四个部位较均等。:17; 头比脖子短，比尾巴短。:10; 尾巴比脖子长，比头长。:7; 头比脖子长，比尾巴短。:7; 头比脖子长，比尾巴长。:7 |
| 213 | 31 | 20 | other_unparsed:18; count_abstract:12; global_balance:1 | 有两个部位比较长。:7; 有三个部位比较长。:3; 腿较短，头和尾巴均长于脖子。:2; 脖子比头长，脖子不是最长。:2; 有一个部位比较长。:2 |
| 214 | 53 | 11 | meta_or_uncertain:39; other_unparsed:8; global_balance:3; count_abstract:3 | 选错了。:39; 两长两短。:3; 差不多。:3; 头，和，腿、头和腿略短，其他中等长度。:1; 腿长，脖，脖子中等长度，尾巴略长，头略短。:1 |
| 215 | 39 | 15 | count_abstract:34; disjoint_inequality:3; other_unparsed:2 | 两个部位长，两个部位短。:7; 三个部位长，一个部位短。:6; 三个部位长。:6; 三长一短。:3; 三个部位短，一个部位长。:3 |
| 216 | 18 | 17 | other_unparsed:15; global_balance:3 | 尾巴和腿差不多长，都比脖子长。:2; 头、脖子、腿都很长，尾巴中等，微微偏短。:1; 每个部位。:1; 尾巴和腿差不多长，偏短。:1; 尾巴和腿都一样长，偏短。:1 |
| 217 | 6 | 5 | other_unparsed:3; count_abstract:2; body_geometry:1 | 尾巴长，有两个部位短。:2; 由长到短是头和腿，脖子和尾巴。:1; 由长到短是头、腿、脖子，尾巴。:1; 头比和脖子比较长。:1; 都比躯干短。:1 |
| 218 | 108 | 91 | other_unparsed:100; other_reference:8; count_abstract:2 | 脖子和尾巴都较长，明显长于腿。:4; 脖子和尾巴都较短，明显短于腿。:3; 脖子、尾巴都长于腿。:3; 头、尾巴长度明显长于脖子、腿。:2; 尾巴明显长于其余三部位。:2 |
| 219 | 51 | 26 | global_balance:30; other_unparsed:18; meta_or_uncertain:2; vague_size:1 | 比较均衡。:19; 头和尾巴。:3; 稍微短。:3; 均衡。:3; 选错了。:2 |
| 220 | 82 | 76 | other_unparsed:61; disjoint_inequality:8; other_reference:7; global_balance:4; count_abstract:4 | 脖子和尾巴长于腿，长于头。:3; 头和尾巴不一样长，且腿长。:3; 头和尾巴一样长，且脖子和腿不一样长。:2; 头、脖子和腿都比尾巴长。:2; 头、脖子、尾巴和腿都比尾巴短。:1 |
| 221 | 4 | 4 | other_unparsed:2; vague_size:1; meta_or_uncertain:1 | 脖子和尾巴相差比较大，腿和头相差比较小。:1; 腿不是最长，头不是最短。:1; 脖子，最长。:1; 选错了。:1 |
| 222 | 110 | 97 | vague_size:69; other_unparsed:30; count_abstract:7; other_reference:2; global_balance:2 | 头很大。:4; 体型中等，尾巴和脖子比腿长。:3; 四个部位差不多长，体型偏大。:2; 体型中等，四个部位差不多长。:2; 体型大，四个部位都差不多。:2 |
| 223 | 2 | 2 | other_unparsed:2 | 头适中，脖子短，腿长，尾巴短，脖子比腿短，头也比腿短。:1; 头适中，脖子短，腿适中，尾巴短，头比脖子长，头也比腿短。:1 |
| 224 | 28 | 28 | other_unparsed:19; count_abstract:4; other_reference:3; proportion_or_ratio:2; global_balance:2 | 头比尾巴长，比脖子长。:1; 头最长，尾巴，腿很短。:1; 头和脖子相对。:1; 头比脖子长，比尾巴长。:1; 四个部位都差不多长，腿最长，长度都适中。:1 |
| 225 | 2 | 2 | other_unparsed:2 | 脖子远比头长。:1; 头远比脖子长。:1 |
| 226 | 108 | 73 | other_unparsed:62; count_abstract:27; global_balance:12; proportion_or_ratio:5; body_geometry:2 | 两长两短。:12; 三长一短。:11; 三个差不多长。:4; 两短两长。:3; 三短一长。:3 |
| 227 | 21 | 11 | disjoint_inequality:10; other_unparsed:8; body_geometry:2; global_balance:1; count_abstract:1 | 四个部位长度各不相同。:7; 脖子和腿长度近似。:2; 四个部位长度不一。:2; 尾巴和腿长度近似。:2; 四个部位长度不同。:2 |
| 228 | 20 | 7 | global_balance:15; other_unparsed:3; count_abstract:1; body_geometry:1 | 比较均匀。:10; 均匀。:5; 三长一短。:1; 脖子不是最短。:1; 整体都比较长。:1 |
| 231 | 23 | 21 | other_unparsed:12; global_balance:10; ordinal_or_secondary:1 | 肘比腿长，脖子比尾巴长。:2; 各个部分差不多长。:2; 脖子最短，腿和尾巴最长，并且差不多。:1; 尾巴最长，其他部位稍短，并且长度差不多。:1; 尾巴最长，然后是脖子，其他两个部位差不多。:1 |
| 301 | 36 | 34 | other_unparsed:35; global_balance:1 | 脖子和头都很长，长度相近。:3; 腿极长，头、脖子和尾巴相近，稍短一些。:1; 腿极短，头、脖子、尾巴都较长，长度相近。:1; 腿很长，头和脖子相近，也都比较长。:1; 腿很长，其余三个部位较长，长度相近。:1 |
| 302 | 13 | 12 | other_unparsed:7; global_balance:6; other_reference:1 | 腿长，其他部位都不是很长，比较匀称。:2; 脖子，脖子长，尾巴短。:1; 上半身很长，腿很短。:1; 脖子长，尾巴长，头短，腿中等，整体比较匀称。:1; 上半身比较长，腿相对短。:1 |
| 303 | 1 | 1 | other_unparsed:1 | 头，脖子、腿都短，尾巴也短、比其他部位长一点。:1 |
| 304 | 3 | 2 | other_unparsed:2; other_reference:1; body_geometry:1 | 头和腿长度相同。:2; 头和躯干长度相同，其他部位长度是躯干的0.7倍。:1 |
| 305 | 33 | 11 | body_geometry:28; vague_size:28; other_unparsed:4; meta_or_uncertain:1 | 挺高大。:8; 身材高大。:7; 很高大。:6; 挺高大，脖子短。:5; 很高。:1 |
| 306 | 37 | 37 | other_unparsed:26; body_geometry:11 | 腿和尾巴一样长，较长，脖子非常长，头较短。:1; 腿和头差不多一样长，长度较长，尾巴和脖子相对来说较短。:1; 脖子比头长，和腿差不多，比腿稍长，尾巴最短。:1; 较躯干来说，腿较短，其他部位均较长，脖子比头长。:1; 较躯干来说，腿适中，其他部位较长一些。:1 |
| 307 | 9 | 9 | other_unparsed:8; vague_size:1 | 腿、脖子和尾巴，四个部位都短，头较长。:1; 头和脖子差不多长，都是较长，尾巴中等，腿较长。:1; 头和脖子差不多长，它们都是中等，腿较长，尾巴较长。:1; 脖子是头的两倍。:1; 头和脖子，腿较短，尾巴很短。:1 |
| 308 | 27 | 16 | other_unparsed:27 | 脖子不是最长。:12; 腿与尾巴，最短。:1; 脖子与尾巴长度相似，且最长。:1; 脖子与腿长度相似，且长度长于头。:1; 头与尾巴长度相似，比脖子长。:1 |
| 309 | 10 | 8 | global_balance:6; other_unparsed:2; other_reference:1; meta_or_uncertain:1 | 整体都较为均匀。:2; 比较均衡。:2; 尾巴非常短，其他部位正常。:1; 腿、头、尾巴都比脖子长。:1; 尾巴短，小小。:1 |
| 310 | 37 | 36 | other_unparsed:15; body_geometry:12; global_balance:11; other_reference:3; ordinal_or_secondary:1 | 头短，躯干长。:2; 头最长，腿最短，相差较大。:1; 脖子长，头和尾巴，中间腿最短。:1; 头很长，脖子和腿，长度差不多。:1; 腿最长，头、尾巴、脖子较短，且长度差不多。:1 |
| 311 | 44 | 37 | other_unparsed:43; ordinal_or_secondary:1 | 腿最长，头、尾巴和脖子。:2; 脖子最长，尾巴、头和腿。:2; 腿最长，尾巴、头和脖子。:2; 脖子最长，腿、头和尾巴。:2; 头最长，尾巴、脖子和腿。:2 |
| 312 | 9 | 9 | other_unparsed:7; meta_or_uncertain:1; global_balance:1 | 一样长。:1; 选错了。:1; 差不多长。:1; 都略短，头比尾巴长。:1; 头、腿、尾巴，很短。:1 |
| 313 | 7 | 7 | other_unparsed:6; body_geometry:1 | 尾巴，巴长，腿短，头短。:1; 头很长，尾巴、腿、脖子差不多长，都比头稍微短一点。:1; 腿长，尾巴长，头和脖子差不多长，头和脖子都比尾巴和腿短。:1; 腿短，尾巴长，头较短，和脖子相比。:1; 尾巴短，腿短，脖子比头短，都比尾巴和腿长。:1 |
| 314 | 23 | 12 | other_unparsed:19; body_geometry:4; count_abstract:3 | 躯干长于脖子，且短于腿。:6; 朝向向左。:4; 有一个部位比躯干长。:3; 躯干短于脖子，且长于尾巴。:2; 脖子长于头、长于躯干，长于尾巴、长于腿。:1 |
| 315 | 1 | 1 | other_unparsed:1 | 脖。:1 |
| 316 | 52 | 44 | other_unparsed:50; other_reference:2 | 腿短，且是最短，头比脖子长。:4; 腿短，且是最短，头比脖子短。:3; 腿长，尾巴明显比腿短。:2; 腿短，且脖子不是头、脖子、尾巴里最短。:2; 腿短，尾巴不比脖子长。:2 |
| 317 | 62 | 34 | extreme_endpoint:31; count_abstract:26; other_unparsed:21; body_geometry:5 | 有部位达到最长或最短长度。:6; 腿不是最短的部位。:5; 脖子不是最长的部位。:5; 腿没有达到最大长度。:4; 没有部位达到最长或最短长度。:4 |
| 318 | 17 | 17 | other_unparsed:12; body_geometry:5; global_balance:1 | 脖子最短，尾巴比腿短，比脖子长，头也比较长。:1; 腿最短，脖子和尾巴一样长，比腿长，头最长。:1; 尾巴比腿长，头和脖子，脖子更长。:1; 尾巴比腿长，躯干差距很大。:1; 尾巴比腿长，差距不大，脖子和头都很长。:1 |
| 319 | 73 | 29 | proportion_or_ratio:43; other_unparsed:10; global_balance:7; meta_or_uncertain:6; count_abstract:5 | 头和脖子的比例大于腿和尾巴。:26; 头和脖子的比例小于腿和尾巴。:10; 选错了。:6; 头和脖子都比腿长。:3; 四个部位长度比较平衡。:2 |
| 321 | 20 | 19 | other_unparsed:20 | 腿不是最短的，头比脖子长。:2; 腿较，尾巴和脖子都长，头也长。:1; 腿超级短，头、脖子和尾巴都比腿长。:1; 头比脖子短，腿略微比尾巴短。:1; 头比脖子短，尾巴也比腿短。:1 |
| 322 | 56 | 36 | other_unparsed:48; body_geometry:5; disjoint_inequality:3 | 头不是最长。:21; 脖子和腿，头比脖子和腿长，尾巴很短。:1; 脖子和头非常长，腿略短于二者，尾巴最短。:1; 脖子和尾巴一样长，头略短于二者，腿最短。:1; 脖子和腿差不多长，头和尾巴差不多长，且长于前二者。:1 |
| 323 | 3 | 1 | meta_or_uncertain:3 | 选错了。:3 |
| 324 | 4 | 4 | other_unparsed:4 | 腿较短，前已经较长。:1; 腿和脖子较长，头和尾巴长中等，较短。:1; 腿和头较长，脖子长度中，脖子长度中等，尾巴较短。:1; 腿较长，下巴为中等。:1 |
| 325 | 4 | 2 | other_unparsed:4 | 每个都长。:3; 每一个都很长。:1 |
| 326 | 51 | 40 | body_geometry:24; other_unparsed:17; meta_or_uncertain:8; global_balance:3; ordinal_or_secondary:1 | 假设脖子无关。:6; 朝右。:4; 假设尾巴无关。:2; 腿和躯干差不多，头比脖子长。:2; 腿比尾巴短，头在躯干以下。:2 |
| 327 | 1 | 1 | other_unparsed:1 | 头短，尾巴短，脖子长，腿中上。:1 |
| 328 | 6 | 6 | other_unparsed:5; global_balance:1 | 尾巴特别短，尾巴明显比腿短，头和脖子加在一起比较长。:1; 腿和尾巴差不多长，都是中等长度，头和脖子都比较长。:1; 尾巴和腿都比较长，差不多，头加脖子比较短。:1; 尾巴长，头和脖子都很短，都比腿短。:1; 尾巴短，头和脖子都比腿长。:1 |
| 329 | 4 | 3 | other_unparsed:3; ordinal_or_secondary:1 | 按错了。:2; 全身都长。:1; 尾巴长，其次是腿，然后是头和脖子。:1 |
| 331 | 3 | 3 | other_unparsed:1; body_geometry:1; global_balance:1 | 头长，尾巴长，腿也还行。:1; 躯干长，头短。:1; 均等。:1 |

### 高频未编码文本 Top 80

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短于中间值。 | 62 | 头短于中间值 | other_unparsed | S3T14, S3T17, S3T20, S3T21, S3T22, S3T23, S3T26, S3T27 |
| 选错了。 | 61 | 选错了 | meta_or_uncertain | S1T225, S2T69, S1T308, S1T209, S2T112, S5T316, S1T97, S1T131 |
| 头长于中间值。 | 53 | 头长于中间值 | other_unparsed | S3T15, S3T16, S3T18, S3T19, S3T24, S3T25, S3T33, S3T36 |
| 头不是最长。 | 35 | 头不是最长 | other_unparsed | S2T46, S2T47, S2T48, S2T164, S2T165, S2T166, S2T168, S2T169 |
| 头和脖子的比例大于腿和尾巴。 | 26 | 头和脖子的比例大于腿和尾巴 | proportion_or_ratio | S1T88, S1T100, S1T103, S1T108, S1T109, S1T111, S1T113, S1T115 |
| 比较均衡。 | 21 | 比较均衡 | global_balance | S1T63, S1T69, S1T75, S1T76, S1T78, S1T160, S1T189, S1T256 |
| 两长两短。 | 19 | 两长两短 | count_abstract | S1T275, S1T297, S3T220, S3T221, S3T222, S2T173, S2T174, S1T46 |
| 三个部位很长。 | 17 | 三个部位很长 | count_abstract | S1T123, S1T124, S1T125, S1T126, S1T134, S1T135, S1T136, S1T137 |
| 四个部位较均等。 | 17 | 四个部位较均等 | global_balance | S1T61, S1T65, S1T68, S1T72, S1T93, S1T119, S1T124, S1T126 |
| 腿和尾巴都比脖子长。 | 16 | 腿和尾巴都比脖子长 | other_unparsed | S1T180, S1T244, S1T245, S1T248, S1T250, S1T252, S1T253, S1T256 |
| 一个部位很长。 | 15 | 一个部位很长 | count_abstract | S1T147, S1T148, S1T150, S1T153, S1T155, S1T32, S1T33, S1T36 |
| 三长一短。 | 15 | 三长一短 | count_abstract | S2T175, S2T238, S2T240, S1T26, S1T32, S1T45, S1T50, S1T53 |
| 有一个部位长于躯干。 | 13 | 有一个部位长于躯干 | count_abstract, body_geometry | S2T281, S2T284, S2T293, S2T295, S2T296, S2T298, S2T299, S2T301 |
| 有三个部位几乎一样长。 | 13 | 有三个部位几乎一样长 | count_abstract | S1T29, S1T106, S1T108, S1T142, S1T193, S1T257, S1T260, S1T262 |
| 比较均匀。 | 12 | 比较均匀 | global_balance | S1T139, S1T146, S2T79, S2T82, S2T88, S2T96, S2T98, S2T102 |
| 脖子不是最长。 | 12 | 脖子不是最长 | other_unparsed | S2T235, S2T242, S2T243, S2T244, S2T257, S2T258, S2T273, S2T280 |
| 三个部位几乎一样长。 | 11 | 三个部位几乎一样长 | count_abstract | S1T170, S1T190, S1T223, S1T224, S1T241, S1T290, S1T310, S1T314 |
| 两个部位长，两个部位短。 | 10 | 两个部位长; 两个部位短 | count_abstract | S1T109, S1T110, S1T112, S2T52, S2T62, S2T63, S2T87, S2T225 |
| 头和脖子的比例小于腿和尾巴。 | 10 | 头和脖子的比例小于腿和尾巴 | proportion_or_ratio | S1T98, S1T101, S1T102, S1T104, S1T112, S1T118, S1T122, S1T123 |
| 头比脖子短，比尾巴短。 | 10 | 比尾巴短 | other_unparsed | S4T138, S4T139, S4T145, S4T146, S4T151, S4T152, S4T154, S4T155 |
| 头比脖子长，比尾巴长。 | 10 | 比尾巴长 | other_unparsed | S4T141, S4T144, S4T149, S4T150, S4T153, S4T168, S4T195, S1T155 |
| 头在躯干之下。 | 9 | 头在躯干之下 | body_geometry | S1T108, S1T111, S1T114, S1T116, S1T117, S1T118, S1T119, S1T121 |
| 有两个部位一样长。 | 9 | 有两个部位一样长 | count_abstract | S1T15, S1T59, S1T60, S1T61, S1T62, S1T64, S1T66, S1T40 |
| 有两个部位几乎一样长。 | 9 | 有两个部位几乎一样长 | count_abstract | S1T30, S1T38, S1T135, S1T138, S1T141, S1T144, S1T167, S1T259 |
| 三个部位长。 | 8 | 三个部位长 | count_abstract | S1T121, S1T122, S2T35, S2T50, S2T184, S2T188, S2T189, S2T194 |
| 尾巴不是最短的。 | 8 | 尾巴不是最短的 | other_unparsed | S1T118, S1T144, S1T145, S1T146, S1T147, S1T149, S1T151, S1T154 |
| 挺高大。 | 8 | 挺高大 | body_geometry, vague_size | S1T50, S1T51, S1T52, S1T54, S1T55, S1T56, S1T59, S1T61 |
| 脖子不是最长的。 | 8 | 脖子不是最长的 | other_unparsed | S2T34, S2T35, S2T36, S2T37, S2T38, S2T39, S2T43, S2T44 |
| 三个部位长，一个部位短。 | 7 | 三个部位长; 一个部位短 | count_abstract | S1T111, S2T51, S2T57, S2T58, S2T64, S2T231, S2T236 |
| 只有一个部位很长。 | 7 | 只有一个部位很长 | count_abstract | S1T118, S1T129, S1T132, S1T133, S1T140, S1T141, S1T142 |
| 四个部位长度各不相同。 | 7 | 四个部位长度各不相同 | disjoint_inequality | S1T97, S1T98, S1T100, S1T101, S1T103, S1T132, S1T133 |
| 头比脖子长，比尾巴短。 | 7 | 比尾巴短 | other_unparsed | S4T93, S4T140, S4T142, S4T147, S4T148, S4T163, S4T166 |
| 尾巴和脖子不一样长。 | 7 | 尾巴和脖子不一样长 | disjoint_inequality | S1T26, S1T27, S1T54, S1T55, S1T57, S1T58, S1T155 |
| 尾巴比脖子长，比头长。 | 7 | 比头长 | other_unparsed | S3T258, S3T261, S3T271, S3T272, S4T88, S4T91, S4T92 |
| 有三个部位长于躯干。 | 7 | 有三个部位长于躯干 | count_abstract, body_geometry | S2T282, S2T283, S2T286, S2T288, S2T292, S2T300, S2T306 |
| 有三个部位长度一样。 | 7 | 有三个部位长度一样 | count_abstract | S3T189, S3T190, S3T207, S3T210, S3T216, S3T236, S4T216 |
| 有两个部位比较长。 | 7 | 有两个部位比较长 | count_abstract | S2T306, S2T309, S2T310, S2T312, S2T314, S2T317, S2T318 |
| 有两个部位长于躯干。 | 7 | 有两个部位长于躯干 | count_abstract, body_geometry | S2T280, S2T285, S2T290, S2T291, S2T303, S2T307, S2T310 |
| 腿不是最短的部位。 | 7 | 腿不是最短的部位 | other_unparsed | S2T56, S2T57, S2T185, S2T186, S2T187, S2T189, S2T190 |
| 身材高大。 | 7 | 身材高大 | body_geometry, vague_size | S1T70, S1T80, S1T82, S1T89, S1T92, S1T94, S1T96 |
| 躯体下方比上面高。 | 7 | 躯体下方比上面高 | body_geometry | S1T179, S1T181, S1T183, S1T184, S1T185, S1T186, S1T187 |
| 两个部位很长。 | 6 | 两个部位很长 | count_abstract | S1T119, S1T120, S1T127, S1T128, S1T145, S1T146 |
| 假设脖子无关。 | 6 | 假设脖子无关 | meta_or_uncertain | S1T243, S1T244, S1T245, S1T246, S1T247, S1T248 |
| 四个部位较匀称。 | 6 | 四个部位较匀称 | global_balance | S1T182, S1T242, S1T246, S1T255, S1T259, S2T24 |
| 尾巴比脖子短，比头短。 | 6 | 比头短 | other_unparsed | S3T257, S3T262, S3T265, S4T87, S4T89, S4T90 |
| 很高大。 | 6 | 很高大 | body_geometry, vague_size | S1T1, S1T12, S1T23, S1T24, S1T31, S1T46 |
| 有三个部位一样长。 | 6 | 有三个部位一样长 | count_abstract | S1T64, S1T210, S2T199, S3T23, S3T51, S3T108 |
| 有部位达到最长或最短长度。 | 6 | 有部位达到最长或最短长度 | count_abstract, extreme_endpoint | S2T109, S2T110, S2T113, S2T316, S2T317, S2T319 |
| 脖子比头短，比尾巴短。 | 6 | 比尾巴短 | other_unparsed | S4T96, S4T97, S4T98, S4T101, S4T105, S4T110 |
| 躯干长于脖子，且短于腿。 | 6 | 且短于腿 | other_unparsed | S3T137, S3T139, S3T141, S3T142, S3T145, S3T146 |
| 两个部位一样长。 | 5 | 两个部位一样长 | count_abstract | S1T86, S1T90, S1T93, S1T179, S1T180 |
| 两个部位几乎一样长。 | 5 | 两个部位几乎一样长 | count_abstract | S1T164, S2T9, S2T76, S2T112, S2T116 |
| 均匀。 | 5 | 均匀 | global_balance | S2T104, S2T107, S2T110, S2T112, S2T116 |
| 头身比例比较协调。 | 5 | 头身比例比较协调 | proportion_or_ratio | S1T30, S1T31, S1T32, S1T33, S1T36 |
| 尾巴和腿都比脖子短。 | 5 | 尾巴和腿都比脖子短 | other_unparsed | S1T280, S1T281, S1T282, S1T284, S1T285 |
| 挺高大，脖子短。 | 5 | 挺高大 | body_geometry, vague_size | S1T37, S1T40, S1T42, S1T45, S1T47 |
| 有一个部位比躯干长。 | 5 | 有一个部位比躯干长 | count_abstract, body_geometry | S1T130, S1T131, S1T43, S1T44, S1T45 |
| 脖子不是最长的部位。 | 5 | 脖子不是最长的部位 | other_unparsed | S2T176, S2T177, S2T178, S2T179, S2T181 |
| 脖子比头长，比尾巴长。 | 5 | 比尾巴长 | other_unparsed | S4T94, S4T99, S4T103, S4T106, S4T109 |
| 腿不是最短。 | 5 | 腿不是最短 | other_unparsed | S2T157, S2T158, S2T159, S2T161, S2T162 |
| 腿比脖子短，比尾巴短。 | 5 | 比尾巴短 | other_unparsed | S4T170, S4T173, S4T174, S4T176, S2T35 |
| 都差不多长。 | 5 | 都差不多长 | global_balance | S1T237, S2T73, S2T242, S2T312, S2T214 |
| 三个差不多长。 | 4 | 三个差不多长 | global_balance | S1T67, S1T72, S1T78, S1T151 |
| 三个部位短，一个部位长。 | 4 | 三个部位短; 一个部位长 | count_abstract | S2T59, S2T86, S2T237, S1T74 |
| 三个部位长度一样。 | 4 | 三个部位长度一样 | count_abstract | S3T82, S3T83, S3T120, S3T122 |
| 三短一长。 | 4 | 三短一长 | other_unparsed | S2T239, S1T44, S1T59, S1T262 |
| 两个部位比中间值长。 | 4 | 两个部位比中间值长 | count_abstract | S2T149, S2T150, S2T151, S2T152 |
| 低头。 | 4 | 低头 | other_unparsed | S1T94, S1T95, S1T97, S1T98 |
| 体型分布的不均匀。 | 4 | 体型分布的不均匀 | global_balance, vague_size | S1T70, S1T71, S1T79, S1T81 |
| 四个部位都很长，体型很大。 | 4 | 体型很大 | vague_size | S1T157, S1T184, S1T195, S1T210 |
| 头在躯干之上。 | 4 | 头在躯干之上 | body_geometry | S1T110, S1T112, S1T113, S1T126 |
| 头很大。 | 4 | 头很大 | other_unparsed | S2T125, S2T155, S2T162, S2T243 |
| 头比脖子短，比尾巴长。 | 4 | 比尾巴长 | other_unparsed | S4T137, S4T161, S4T167, S4T194 |
| 头身比例不协调。 | 4 | 头身比例不协调 | proportion_or_ratio | S1T29, S1T34, S1T35, S1T37 |
| 尾巴长，头的位置高。 | 4 | 头的位置高 | other_unparsed | S6T75, S6T76, S6T77, S6T81 |
| 差不多长。 | 4 | 差不多长 | global_balance | S3T237, S1T77, S2T117, S2T254 |
| 有三个部位比较长。 | 4 | 有三个部位比较长 | count_abstract | S2T96, S2T305, S2T311, S2T316 |
| 有奇数个部位长于躯干。 | 4 | 有奇数个部位长于躯干 | count_abstract, body_geometry | S1T153, S1T154, S1T155, S1T159 |
| 朝右。 | 4 | 朝右 | other_unparsed | S1T81, S1T82, S1T83, S1T84 |
| 朝向向左。 | 4 | 朝向向左 | other_unparsed | S1T37, S1T39, S1T40, S1T42 |

### 逐被试未编码文本

#### S102

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子比头长，但是是最短的两个。 | 4 | 但是是最短的两个 | other_unparsed | S1T149, S1T150, S1T160, S1T172 |
| 脖子比头短，但是是最长的两个。 | 3 | 但是是最长的两个 | other_unparsed | S1T178, S1T179, S1T197 |
| 脖子比头短，它们两个是最长的。 | 3 | 它们两个是最长的 | other_unparsed | S1T204, S1T205, S1T206 |
| 脖子比头短，都短于腿。 | 3 | 都短于腿 | other_unparsed | S1T310, S1T311, S1T313 |
| 脖子比头短，都长于腿。 | 2 | 都长于腿 | other_unparsed | S1T307, S1T308 |
| 脖子短于头，但它们是最小的。 | 2 | 但它们是最小的 | other_unparsed | S1T231, S1T244 |
| 三长一中。 | 1 | 三长一中 | count_abstract | S1T286 |
| 两长两中。 | 1 | 两长两中 | other_unparsed | S1T279 |
| 两长两短。 | 1 | 两长两短 | count_abstract | S1T275 |
| 脖子小于头，但它们是最大的。 | 1 | 但它们是最大的 | other_unparsed | S1T251 |
| 脖子微短于头，腿适中，尾巴短。 | 1 | 脖子微短于头 | other_unparsed | S1T109 |
| 脖子比头短，但是是最短的两个。 | 1 | 但是是最短的两个 | other_unparsed | S1T176 |
| 脖子比头短，但是都小于腿。 | 1 | 但是都小于腿 | other_unparsed | S1T301 |
| 脖子比头短，但是都长于腿。 | 1 | 但是都长于腿 | other_unparsed | S1T299 |
| 脖子比头短，但都比尾巴短。 | 1 | 但都比尾巴短 | other_unparsed | S1T182 |
| 脖子比头短，而且不是最长的两个。 | 1 | 而且不是最长的两个 | other_unparsed | S1T184 |
| 脖子比头短，而且是最长的两个。 | 1 | 而且是最长的两个 | other_unparsed | S1T186 |
| 脖子比头短，腿和尾巴差不多，都适中。 | 1 | 都适中 | other_unparsed | S1T138 |
| 脖子比头短，都大于腿。 | 1 | 都大于腿 | other_unparsed | S1T303 |
| 脖子比头长，不确定是不是最短的两个。 | 1 | 不确定是不是最短的两个 | meta_or_uncertain | S1T157 |
| 脖子比头长，但是它们两个是最长的。 | 1 | 但是它们两个是最长的 | other_unparsed | S1T201 |
| 脖子比头长，但是是最长的两个。 | 1 | 但是是最长的两个 | other_unparsed | S1T174 |
| 脖子比头长，但是都长于腿。 | 1 | 但是都长于腿 | other_unparsed | S1T298 |
| 脖子比头长，但都比较短。 | 1 | 但都比较短 | other_unparsed | S1T199 |
| 脖子比头长，都大于腿。 | 1 | 都大于腿 | other_unparsed | S1T309 |
| 脖子比头长，都短于腿。 | 1 | 都短于腿 | other_unparsed | S1T305 |
| 脖子比头长，都长于腿。 | 1 | 都长于腿 | other_unparsed | S1T306 |
| 脖子略微短于头，腿适中，尾巴适中。 | 1 | 脖子略微短于头 | other_unparsed | S1T84 |
| 脖子短于头，但它们是最小的两个。 | 1 | 但它们是最小的两个 | other_unparsed | S1T222 |
| 脖子短于头，但它们是最长的两个。 | 1 | 但它们是最长的两个 | other_unparsed | S1T214 |
| 脖子短于头，但是两个都很长。 | 1 | 但是两个都很长 | other_unparsed | S1T237 |
| 脖子短于头，但是它们的平均长度长于尾巴。 | 1 | 但是它们的平均长度长于尾巴 | global_balance | S1T209 |
| 脖子短于头，腿和尾巴都比它们短。 | 1 | 腿和尾巴都比它们短 | other_unparsed | S1T91 |
| 脖子长于头，但它们是最大的两个。 | 1 | 但它们是最大的两个 | other_unparsed | S1T219 |
| 脖子长于头，但它们是最小的两个。 | 1 | 但它们是最小的两个 | other_unparsed | S1T218 |
| 脖子长于头，但它们是最短的两个。 | 1 | 但它们是最短的两个 | other_unparsed | S1T211 |
| 腿长，尾巴适中，脖子略微长于头。 | 1 | 脖子略微长于头 | other_unparsed | S1T78 |
| 腿长，尾巴长，脖子略微长于头。 | 1 | 脖子略微长于头 | other_unparsed | S1T79 |

#### S103

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、尾巴都中等，略长一点，腿较短。 | 1 | 略长一点 | other_unparsed | S1T95 |
| 尾巴很长，长于头和脖子，且腿中等偏短。 | 1 | 长于头和脖子 | other_unparsed | S1T143 |
| 尾巴很长，长于头和脖子，头、脖子、腿都很短。 | 1 | 长于头和脖子 | other_unparsed | S1T35 |
| 尾巴很长，长于头和脖子，腿较短。 | 1 | 长于头和脖子 | other_unparsed | S1T4 |
| 尾巴较长，长于脖子和头。 | 1 | 长于脖子和头 | other_unparsed | S1T185 |
| 脖子和尾巴较长，长于头和腿。 | 1 | 长于头和腿 | other_unparsed | S1T202 |
| 腿中等，头较长，尾巴较长，明显长于脖子。 | 1 | 明显长于脖子 | other_unparsed | S1T44 |
| 腿很短，脖子很长，长于尾巴，尾巴和头中等长度。 | 1 | 长于尾巴 | other_unparsed | S1T33 |
| 腿很长，脖子和头中等长度，长于较短的尾巴。 | 1 | 长于较短的尾巴 | other_unparsed | S1T13 |
| 腿极长，尾巴中等长度，长于头和脖子。 | 1 | 长于头和脖子 | other_unparsed | S1T5 |
| 腿较短，尾巴较长，长于头和脖子。 | 1 | 长于头和脖子 | other_unparsed | S1T21 |
| 腿较长，脖子中等较长，长于尾巴。 | 1 | 长于尾巴 | other_unparsed | S1T52 |

#### S104

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头长，较均匀。 | 1 | 较均匀 | global_balance | S1T109 |
| 比较均匀。 | 1 | 比较均匀 | global_balance | S1T139 |

#### S105

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短于阈值。 | 1 | 腿短于阈值 | other_unparsed | S1T17 |

#### S106

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 躯体下方比上面高。 | 7 | 躯体下方比上面高 | body_geometry | S1T179, S1T181, S1T183, S1T184, S1T185, S1T186, S1T187 |
| 下面的部分更高一些。 | 2 | 下面的部分更高一些 | body_geometry | S1T174, S1T175 |
| 头离地面近。 | 2 | 头离地面近 | other_unparsed | S1T32, S1T33 |
| 腿不是最长的。 | 2 | 腿不是最长的 | other_unparsed | S1T55, S1T58 |
| 躯体下方没有上面高。 | 2 | 躯体下方没有上面高 | body_geometry | S1T180, S1T182 |
| 上面的高度大于下面。 | 1 | 上面的高度大于下面 | body_geometry | S1T173 |
| 上面的高度小于下面。 | 1 | 上面的高度小于下面 | body_geometry | S1T172 |
| 下半身腿比上面的高。 | 1 | 下半身腿比上面的高 | body_geometry | S1T170 |
| 下面的部分高于上面。 | 1 | 下面的部分高于上面 | body_geometry | S1T176 |
| 下面的高度比上面高。 | 1 | 下面的高度比上面高 | body_geometry | S1T178 |
| 下面的高度高于上面。 | 1 | 下面的高度高于上面 | body_geometry | S1T177 |
| 头离地面远。 | 1 | 头离地面远 | other_unparsed | S1T34 |
| 头距离地面比较近。 | 1 | 头距离地面比较近 | other_unparsed | S1T35 |
| 有两个部位一样长。 | 1 | 有两个部位一样长 | count_abstract | S1T15 |
| 点错了。 | 1 | 点错了 | other_unparsed | S1T105 |
| 腿离地面距离不如躯体最高点的距离高。 | 1 | 腿离地面距离不如躯体最高点的距离高 | body_geometry | S1T171 |
| 躯体下方比上面高一些。 | 1 | 躯体下方比上面高一些 | body_geometry | S1T190 |
| 躯体下方没有上面高，腿太短。 | 1 | 躯体下方没有上面高 | body_geometry | S1T188 |

#### S107

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴很长，脖子也挺长，但相对短一些，腿很短。 | 1 | 但相对短一些 | other_unparsed | S1T45 |
| 尾巴、腿、脖子和头长度差不多，都还挺长的。 | 1 | 都还挺长的 | other_unparsed | S1T164 |
| 腿、脖子和头不算特别长，且长度相当。 | 1 | 且长度相当 | global_balance | S1T191 |
| 腿、脖子和头比例比较协调，尾巴短。 | 1 | 腿、脖子和头比例比较协调 | proportion_or_ratio | S1T95 |
| 腿、脖子和头长度差不多，都不算很长。 | 1 | 都不算很长 | other_unparsed | S1T111 |

#### S108

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子不是最长的。 | 8 | 脖子不是最长的 | other_unparsed | S2T34, S2T35, S2T36, S2T37, S2T38, S2T39, S2T43, S2T44 |
| 腿比脖子短，比尾巴长。 | 3 | 比尾巴长 | other_unparsed | S1T26, S1T42, S1T164 |
| 腿比脖子和尾巴短，比头长。 | 2 | 比头长 | other_unparsed | S1T53, S1T54 |
| 选错了。 | 2 | 选错了 | meta_or_uncertain | S1T225, S2T69 |
| 头和脖子显著比腿和尾巴长。 | 1 | 头和脖子显著比腿和尾巴长 | other_unparsed | S1T14 |
| 脖子不是最长的，腿是最长的。 | 1 | 脖子不是最长的 | other_unparsed | S2T48 |
| 脖子和尾巴显著比腿长。 | 1 | 脖子和尾巴显著比腿长 | other_unparsed | S1T8 |
| 腿比尾巴，腿比脖子短。 | 1 | 腿比尾巴 | other_unparsed | S1T166 |
| 腿比脖子短，比头和尾巴长。 | 1 | 比头和尾巴长 | other_unparsed | S1T29 |
| 腿比脖子长，比尾巴短。 | 1 | 比尾巴短 | other_unparsed | S1T31 |

#### S109

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子，头较长，腿一般。 | 1 | 脖子 | other_unparsed | S1T211 |
| 腿较长，脖子一般，头较小。 | 1 | 头较小 | other_unparsed | S1T213 |

#### S110

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短，脖子腿，尾巴长。 | 1 | 脖子腿 | other_unparsed | S1T169 |

#### S111

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短，尾巴、头、脖子长度相同。 | 1 | 尾巴、头、脖子长度相同 | other_unparsed | S1T18 |

#### S112

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴和头之间有交集。 | 1 | 尾巴和头之间有交集 | other_unparsed | S1T136 |
| 尾巴和腿之间关系。 | 1 | 尾巴和腿之间关系 | other_unparsed | S1T29 |
| 尾巴比头长一点，比腿和脖子都短。 | 1 | 比腿和脖子都短 | other_unparsed | S1T34 |
| 尾巴跟脖子不一样长。 | 1 | 尾巴跟脖子不一样长 | disjoint_inequality | S1T39 |
| 尾巴跟腿之间的关系以及跟脖子的关系。 | 1 | 尾巴跟腿之间的关系以及跟脖子的关系 | other_unparsed | S1T37 |
| 脖子和腿之间的关系。 | 1 | 脖子和腿之间的关系 | other_unparsed | S1T47 |

#### S113

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子长，比头长。 | 3 | 比头长 | other_unparsed | S1T293, S1T294, S1T318 |
| 头比脖子长，都很长。 | 2 | 都很长 | other_unparsed | S1T135, S1T137 |
| 不知道。 | 1 | 不知道 | meta_or_uncertain | S1T218 |
| 头和尾巴很长，脖子短一些，腿第二短，但也比较长。 | 1 | 但也比较长 | other_unparsed | S1T24 |
| 头和脖子差不多长，都很长，尾巴和腿比较短。 | 1 | 都很长 | other_unparsed | S1T62 |
| 头和脖子差不多长，都比较长，腿和尾巴很短。 | 1 | 都比较长 | other_unparsed | S1T80 |
| 头和腿一样长，都比较长，脖子和尾巴比较短。 | 1 | 都比较长 | other_unparsed | S1T4 |
| 头最长，尾巴最短，脖子和腿差不多长，都比较长。 | 1 | 都比较长 | other_unparsed | S1T13 |
| 头比脖子短，都有些短。 | 1 | 都有些短 | other_unparsed | S1T136 |
| 头比脖子长，尾巴和腿适中，比较长。 | 1 | 比较长 | other_unparsed | S1T66 |
| 头比脖子长，尾巴和腿适中，短一些。 | 1 | 短一些 | other_unparsed | S1T87 |
| 尾巴和腿很短，头比适中，脖子有些长。 | 1 | 头比适中 | other_unparsed | S1T30 |
| 尾巴最长，头最短，脖子和腿适中，比较长。 | 1 | 比较长 | other_unparsed | S1T38 |
| 尾巴比头长，都比较长，脖子和腿比较短，腿比脖子短。 | 1 | 都比较长 | other_unparsed | S1T3 |
| 尾巴比脖子长，但是都很短，脖子和腿很长，腿特别长。 | 1 | 但是都很短 | other_unparsed | S1T20 |
| 点错了。 | 1 | 点错了 | other_unparsed | S2T32 |
| 脖子中等长度，比头短。 | 1 | 比头短 | other_unparsed | S1T307 |
| 脖子明显比头短。 | 1 | 脖子明显比头短 | other_unparsed | S1T252 |
| 脖子比头长，头比腿长，腿比尾巴长，都有些长。 | 1 | 都有些长 | other_unparsed | S1T1 |
| 脖子短，但是和头的长度差不多。 | 1 | 但是和头的长度差不多 | global_balance | S1T262 |
| 脖子短，比头短。 | 1 | 比头短 | other_unparsed | S1T304 |
| 脖子长，比头短。 | 1 | 比头短 | other_unparsed | S1T319 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S1T308 |

#### S114

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 都差不多长。 | 2 | 都差不多长 | global_balance | S1T237, S2T73 |
| 都不长。 | 1 | 都不长 | other_unparsed | S1T96 |

#### S116

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 嗯，脖子比较长。 | 1 | 嗯 | other_unparsed | S1T48 |
| 四个部位都比较长，其中腿最明显。 | 1 | 其中腿最明显 | other_unparsed | S1T24 |
| 脖子的长度很突出。 | 1 | 脖子的长度很突出 | other_unparsed | S1T28 |
| 腿和脖子都比较长，但腿是最明显的。 | 1 | 但腿是最明显的 | other_unparsed | S1T36 |
| 腿的长度最突出。 | 1 | 腿的长度最突出 | other_unparsed | S1T27 |

#### S118

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 三个部位很长。 | 17 | 三个部位很长 | count_abstract | S1T123, S1T124, S1T125, S1T126, S1T134, S1T135, S1T136, S1T137 |
| 只有一个部位很长。 | 7 | 只有一个部位很长 | count_abstract | S1T118, S1T129, S1T132, S1T133, S1T140, S1T141, S1T142 |
| 两个部位很长。 | 6 | 两个部位很长 | count_abstract | S1T119, S1T120, S1T127, S1T128, S1T145, S1T146 |
| 一个部位很长。 | 5 | 一个部位很长 | count_abstract | S1T147, S1T148, S1T150, S1T153, S1T155 |
| 三个部位都很长。 | 3 | 三个部位都很长 | count_abstract | S1T107, S1T117, S1T130 |
| 两个部位长，两个部位短。 | 3 | 两个部位长; 两个部位短 | count_abstract | S1T109, S1T110, S1T112 |
| 只有两个部位很长。 | 3 | 只有两个部位很长 | count_abstract | S1T131, S1T143, S1T159 |
| 三个部位长。 | 2 | 三个部位长 | count_abstract | S1T121, S1T122 |
| 三个部位都很短。 | 1 | 三个部位都很短 | count_abstract | S1T113 |
| 三个部位都很长，只有头是最短的。 | 1 | 三个部位都很长 | count_abstract | S1T108 |
| 三个部位长，一个部位短。 | 1 | 三个部位长; 一个部位短 | count_abstract | S1T111 |
| 两个部位很长，两个部位很短。 | 1 | 两个部位很长; 两个部位很短 | count_abstract | S1T114 |
| 头、脖子、尾巴、腿都很短，都差不多长度。 | 1 | 都差不多长度 | global_balance | S1T52 |
| 头和尾巴一样长，都很长，脖子稍微比头和尾巴短一点，尾巴比腿长，脖子和腿差不多长。 | 1 | 都很长; 脖子稍微比头和尾巴短一点 | other_unparsed | S1T13 |
| 头和脖子都是它们长度范围的1/2，头和脖子一样长，尾巴也非常长，腿也比较长，但没有达到最大长度。 | 1 | 但没有达到最大长度 | extreme_endpoint | S1T26 |
| 头是最长的，脖子第二长，大概在最长长度的1/2，腿也是在它最大长度的1/2，尾巴比较短。 | 1 | 大概在最长长度的1/2 | extreme_endpoint | S1T4 |
| 头短，脖子短，尾巴最长，腿很短，是中等长度以下。 | 1 | 是中等长度以下 | other_unparsed | S1T42 |
| 头短，脖子短，尾巴稍微长一点，是最长的部位，腿短。 | 1 | 是最长的部位 | other_unparsed | S1T40 |
| 头长，脖子长，腿长，尾巴长，都很长。 | 1 | 都很长 | other_unparsed | S1T87 |
| 尾巴、腿、脖子、头都挺长的，都差不多长。 | 1 | 都差不多长 | global_balance | S1T32 |
| 尾巴、腿都非常长，达到了它们的最长长度，脖子也达到了最长长度，头是最短的，但也很长，可能是它自身最长长度的1/2。 | 1 | 但也很长 | other_unparsed | S1T9 |
| 尾巴和腿都很短，尾巴在1/4到1/5之间，腿在1/4到1/5之间，但是脖子和头都比较长。 | 1 | 尾巴在1/4到1/5之间; 腿在1/4到1/5之间 | other_unparsed | S1T3 |
| 尾巴比较大。 | 1 | 尾巴比较大 | other_unparsed | S1T256 |
| 按错了。 | 1 | 按错了 | other_unparsed | S1T248 |
| 脖子最长，腿第二长，尾巴比较短，大概是它最长长度的1/3，头和尾巴差不多长。 | 1 | 大概是它最长长度的1/3 | extreme_endpoint | S1T11 |
| 腿非常短，是最短的部位，也是是它自身最短长度，尾巴是第二长的，头和脖子都比较长。 | 1 | 是最短的部位 | other_unparsed | S1T21 |
| 腿非常短，达到了最小长度，头比腿长，脖子也比腿长，尾巴也比腿长，尾巴达到了最长长度。 | 1 | 脖子也比腿长; 尾巴也比腿长 | other_unparsed | S1T8 |

#### S119

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短于中间值。 | 62 | 头短于中间值 | other_unparsed | S3T14, S3T17, S3T20, S3T21, S3T22, S3T23, S3T26, S3T27 |
| 头长于中间值。 | 53 | 头长于中间值 | other_unparsed | S3T15, S3T16, S3T18, S3T19, S3T24, S3T25, S3T33, S3T36 |
| 头不是最长。 | 14 | 头不是最长 | other_unparsed | S2T46, S2T47, S2T48, S2T164, S2T165, S2T166, S2T168, S2T169 |
| 有一个部位长于躯干。 | 13 | 有一个部位长于躯干 | count_abstract, body_geometry | S2T281, S2T284, S2T293, S2T295, S2T296, S2T298, S2T299, S2T301 |
| 有三个部位长于躯干。 | 7 | 有三个部位长于躯干 | count_abstract, body_geometry | S2T282, S2T283, S2T286, S2T288, S2T292, S2T300, S2T306 |
| 有两个部位长于躯干。 | 7 | 有两个部位长于躯干 | count_abstract, body_geometry | S2T280, S2T285, S2T290, S2T291, S2T303, S2T307, S2T310 |
| 腿不是最短。 | 5 | 腿不是最短 | other_unparsed | S2T157, S2T158, S2T159, S2T161, S2T162 |
| 两个部位比中间值长。 | 4 | 两个部位比中间值长 | count_abstract | S2T149, S2T150, S2T151, S2T152 |
| 腿长于中间值。 | 4 | 腿长于中间值 | other_unparsed | S2T156, S3T11, S3T12, S3T13 |
| 两个部位长于中间值。 | 3 | 两个部位长于中间值 | count_abstract | S2T100, S2T102, S2T103 |
| 有两个部位长于中间值。 | 3 | 有两个部位长于中间值 | count_abstract | S2T97, S2T98, S2T268 |
| 没有部位长于躯干。 | 3 | 没有部位长于躯干 | count_abstract, body_geometry | S2T287, S2T294, S2T297 |
| 一个部位比中间值长。 | 2 | 一个部位比中间值长 | count_abstract | S2T153, S2T154 |
| 一个部位长于中间值。 | 2 | 一个部位长于中间值 | count_abstract | S2T99, S2T101 |
| 脖子比中间值长。 | 2 | 脖子比中间值长 | other_unparsed | S2T145, S2T147 |
| 三个部位长于中间值。 | 1 | 三个部位长于中间值 | count_abstract | S2T215 |
| 只有一个部位长于躯干。 | 1 | 只有一个部位长于躯干 | count_abstract, body_geometry | S1T35 |
| 大部分长于躯干。 | 1 | 大部分长于躯干 | body_geometry | S2T183 |
| 头比中间值长。 | 1 | 头比中间值长 | other_unparsed | S2T146 |
| 头没有长于腿。 | 1 | 头没有长于腿 | other_unparsed | S2T67 |
| 少于两个部位长于中间值。 | 1 | 少于两个部位长于中间值 | count_abstract | S2T269 |
| 尾巴和脖子不是最短的。 | 1 | 尾巴和脖子不是最短的 | other_unparsed | S2T193 |
| 有一个部位长于中间值。 | 1 | 有一个部位长于中间值 | count_abstract | S2T267 |
| 有一个部位长长于躯干。 | 1 | 有一个部位长长于躯干 | count_abstract, body_geometry | S2T289 |
| 有三个部位比较长。 | 1 | 有三个部位比较长 | count_abstract | S2T96 |
| 脖子和腿长度差异比较大。 | 1 | 脖子和腿长度差异比较大 | other_unparsed | S1T16 |
| 脖子没有长于尾巴。 | 1 | 脖子没有长于尾巴 | other_unparsed | S2T69 |
| 脖子长于中间值。 | 1 | 脖子长于中间值 | other_unparsed | S2T272 |
| 腿不是最长。 | 1 | 腿不是最长 | other_unparsed | S2T111 |
| 腿和尾巴长于中间值。 | 1 | 腿和尾巴长于中间值 | other_unparsed | S2T273 |
| 腿比中间值长。 | 1 | 腿比中间值长 | other_unparsed | S2T148 |
| 腿短于中间值。 | 1 | 腿短于中间值 | other_unparsed | S3T10 |
| 超过两个等于躯干。 | 1 | 超过两个等于躯干 | body_geometry | S2T182 |
| 超过两个部位短于躯干。 | 1 | 超过两个部位短于躯干 | count_abstract, body_geometry | S2T180 |
| 超过两个部位长于躯干。 | 1 | 超过两个部位长于躯干 | count_abstract, body_geometry | S1T34 |
| 都短于躯干。 | 1 | 都短于躯干 | body_geometry | S2T181 |

#### S120

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 有奇数个部位长于躯干。 | 4 | 有奇数个部位长于躯干 | count_abstract, body_geometry | S1T153, S1T154, S1T155, S1T159 |
| 有偶数个部位长于躯干。 | 3 | 有偶数个部位长于躯干 | count_abstract, body_geometry | S1T156, S1T157, S1T158 |
| 头明显长于其他部位。 | 2 | 头明显长于其他部位 | other_reference | S1T28, S1T33 |
| 脖子和尾巴的长度不一样。 | 2 | 脖子和尾巴的长度不一样 | disjoint_inequality | S1T42, S1T43 |
| 脖子明显短于其他部位。 | 2 | 脖子明显短于其他部位 | other_reference | S1T109, S1T186 |
| 腿明显短于其他部位。 | 2 | 腿明显短于其他部位 | other_reference | S1T35, S1T36 |
| 五个部位的长度都差不多。 | 1 | 五个部位的长度都差不多 | global_balance | S1T25 |
| 头和尾巴加起来短于脖子和躯干，也短于脖子和腿。 | 1 | 也短于脖子和腿 | other_unparsed | S1T58 |
| 头和尾巴长度大于脖子和腿的长度。 | 1 | 头和尾巴长度大于脖子和腿的长度 | other_unparsed | S1T63 |
| 头明显长于其他四个部位。 | 1 | 头明显长于其他四个部位 | other_reference | S1T26 |
| 头比较长的，长于尾巴，脖子长于腿。 | 1 | 长于尾巴 | other_unparsed | S1T74 |
| 头比较长，和躯干差不多。 | 1 | 和躯干差不多 | body_geometry, global_balance | S1T5 |
| 尾巴和脖子长度不一样。 | 1 | 尾巴和脖子长度不一样 | disjoint_inequality | S1T41 |
| 点错了。 | 1 | 点错了 | other_unparsed | S1T310 |
| 脖子明显短于其他四个部位。 | 1 | 脖子明显短于其他四个部位 | other_reference | S1T59 |
| 脖子明显长于其他的四个部位。 | 1 | 脖子明显长于其他的四个部位 | other_reference | S1T107 |
| 脖子比较长，跟腿差不多，也跟躯干差不多。 | 1 | 跟腿差不多; 也跟躯干差不多 | body_geometry, global_balance | S1T9 |
| 脖子比较长，长于头。 | 1 | 长于头 | other_unparsed | S1T13 |
| 脖子非常长，跟躯干差不多。 | 1 | 跟躯干差不多 | body_geometry, global_balance | S1T2 |
| 腿比较长，长于头和尾巴。 | 1 | 长于头和尾巴 | other_unparsed | S1T6 |

#### S121

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿不是所有部位里最短的。 | 4 | 腿不是所有部位里最短的 | count_abstract | S2T30, S2T34, S2T35, S2T36 |
| 头在躯干上方，腿比较长。 | 3 | 头在躯干上方 | body_geometry | S1T291, S1T294, S1T296 |
| 头不是所有部位里最短的。 | 2 | 头不是所有部位里最短的 | count_abstract | S2T38, S2T39 |
| 腿不是最短的部位。 | 2 | 腿不是最短的部位 | other_unparsed | S2T56, S2T57 |
| 头和脖子差不多，都比较长，腿较短。 | 1 | 都比较长 | other_unparsed | S1T237 |
| 头在躯干上方，腿比较短。 | 1 | 头在躯干上方 | body_geometry | S1T293 |
| 头在躯干上方，腿比较长，尾巴很短。 | 1 | 头在躯干上方 | body_geometry | S1T295 |
| 头在躯干上方，腿较短。 | 1 | 头在躯干上方 | body_geometry | S1T299 |
| 头在躯干上方，腿较长。 | 1 | 头在躯干上方 | body_geometry | S1T298 |
| 头在躯干下方。 | 1 | 头在躯干下方 | body_geometry | S1T246 |
| 头在躯干下方，腿比较短。 | 1 | 头在躯干下方 | body_geometry | S1T292 |
| 头在躯干下方，腿较短。 | 1 | 头在躯干下方 | body_geometry | S1T297 |
| 头在躯干下方，腿较长。 | 1 | 头在躯干下方 | body_geometry | S1T300 |
| 头在躯干的上方。 | 1 | 头在躯干的上方 | body_geometry | S1T289 |
| 头比脖子短，腿和尾巴差不多，都很长。 | 1 | 都很长 | other_unparsed | S1T207 |
| 头比脖子短，腿比尾巴，差不多。 | 1 | 腿比尾巴; 差不多 | global_balance | S1T212 |
| 头比脖子长一点。腿和尾巴差不多，都很短。 | 1 | 都很短 | other_unparsed | S1T145 |
| 头比脖子长，腿远远长于头和脖子，尾巴较短。 | 1 | 腿远远长于头和脖子 | other_unparsed | S1T127 |
| 头脖子，尾巴，头脖子腿差不多，尾巴较短。 | 1 | 头脖子; 尾巴 | other_unparsed | S1T140 |
| 脖子和腿差别比较大。 | 1 | 脖子和腿差别比较大 | other_unparsed | S1T21 |

#### S123

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、腿、尾巴、躯干这些部位的长度进行描述，但不一定用的是这几个词语，且可能会涉及到这几个部位之间的比较，包括大小和长短关系。 | 1 | 头、脖子、腿、尾巴、躯干这些部位的长度进行描述; 但不一定用的是这几个词语; 且可能会涉及到这几个部位之间的比较; 包括大小和长短关系 | count_abstract, body_geometry | S1T73 |
| 头和脖外，头和腿长。 | 1 | 头和脖子外 | other_unparsed | S2T63 |
| 头，脖子，腿，尾巴长。 | 1 | 头; 脖子; 腿 | other_unparsed | S2T86 |
| 头，腿长。 | 1 | 头 | other_unparsed | S2T105 |
| 腿和头。头短，尾巴和脖子长。 | 1 | 腿和头 | other_unparsed | S1T91 |
| 腿短，脖，脖子和头长，尾巴短。 | 1 | 脖子 | other_unparsed | S1T60 |
| 腿长，脖子，尾巴、腿都偏短。 | 1 | 脖子 | other_unparsed | S1T25 |

#### S125

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿很长，基本是最长。 | 2 | 基本是最长 | other_unparsed | S1T93, S1T101 |
| 腿非常短，几乎是最短。 | 2 | 几乎是最短 | other_unparsed | S1T92, S1T109 |
| 头短，脖子较短，腿较短，尾巴短，整体看起来比较短小。 | 1 | 整体看起来比较短小 | other_unparsed | S1T39 |
| 头短，脖子较长，腿较长，尾巴较短，整体看起来比较修长。 | 1 | 整体看起来比较修长 | other_unparsed | S1T50 |
| 头长度适中，脖子短，腿短，尾巴较短，整体看起来比较短小。 | 1 | 整体看起来比较短小 | other_unparsed | S1T51 |
| 头长，脖子短，腿短，尾巴较短，整体看起来比较短小。 | 1 | 整体看起来比较短小 | other_unparsed | S1T43 |
| 头长，脖子较短，腿短，尾巴较短，头、脖子、尾巴均比腿长。 | 1 | 头、脖子、尾巴均比腿长 | other_unparsed | S1T8 |
| 头长，脖子长，腿较长，尾巴长，脖子比腿长一点，头和脖子占的比重比较大。 | 1 | 头和脖子占的比重比较大 | other_unparsed | S1T40 |
| 头长，脖子长，腿长，尾巴短，头和脖子的比重比较大，脖子比腿长一点。 | 1 | 头和脖子的比重比较大 | other_unparsed | S1T42 |
| 腿很短，几乎是最短。 | 1 | 几乎是最短 | other_unparsed | S1T114 |
| 腿长度适中，头和脖子比较长，尾巴长度适中，在整体比例中腿显得比较短。 | 1 | 在整体比例中腿显得比较短 | proportion_or_ratio | S1T58 |
| 腿长度适中，脖子比腿更长，整体很修长。 | 1 | 整体很修长 | other_unparsed | S1T102 |
| 腿非常短，基本是最短。 | 1 | 基本是最短 | other_unparsed | S1T95 |
| 腿非常长，基本是最长。 | 1 | 基本是最长 | other_unparsed | S1T121 |

#### S127

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 差不多，都一样长。 | 1 | 差不多; 都一样长 | other_unparsed | S1T90 |
| 腿和头比较长，脖子后尾巴稍微短，一点。 | 1 | 一点 | other_unparsed | S1T17 |

#### S129

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 一个部位很长。 | 10 | 一个部位很长 | count_abstract | S1T32, S1T33, S1T36, S1T37, S1T38, S1T39, S1T43, S1T44 |
| 头在躯干之下。 | 9 | 头在躯干之下 | body_geometry | S1T108, S1T111, S1T114, S1T116, S1T117, S1T118, S1T119, S1T121 |
| 尾巴和脖子不一样长。 | 7 | 尾巴和脖子不一样长 | disjoint_inequality | S1T26, S1T27, S1T54, S1T55, S1T57, S1T58, S1T155 |
| 有两个部位一样长。 | 6 | 有两个部位一样长 | count_abstract | S1T59, S1T60, S1T61, S1T62, S1T64, S1T66 |
| 低头。 | 4 | 低头 | other_unparsed | S1T94, S1T95, S1T97, S1T98 |
| 头在躯干之上。 | 4 | 头在躯干之上 | body_geometry | S1T110, S1T112, S1T113, S1T126 |
| 四个部位和躯干都不一样长。 | 3 | 四个部位和躯干都不一样长 | body_geometry, disjoint_inequality | S1T158, S1T159, S1T160 |
| 头在腿之上。 | 3 | 头在腿之上 | body_geometry | S1T89, S1T91, S1T101 |
| 头和尾巴不一样长。 | 2 | 头和尾巴不一样长 | disjoint_inequality | S1T23, S1T24 |
| 头在腿上。 | 2 | 头在腿上 | other_unparsed | S1T92, S1T93 |
| 头朝左。 | 2 | 头朝左 | other_unparsed | S1T6, S1T20 |
| 有一个部位比躯干长。 | 2 | 有一个部位比躯干长 | count_abstract, body_geometry | S1T130, S1T131 |
| 没有两个部位一样长。 | 2 | 没有两个部位一样长 | count_abstract | S1T63, S1T67 |
| 一个部位很短。 | 1 | 一个部位很短 | count_abstract | S1T34 |
| 抬头。 | 1 | 抬头 | other_unparsed | S1T96 |
| 脖子和尾巴不一样长。 | 1 | 脖子和尾巴不一样长 | disjoint_inequality | S1T165 |
| 脖子尾巴不一样长。 | 1 | 脖子尾巴不一样长 | disjoint_inequality | S1T166 |
| 腿和躯干不一样长。 | 1 | 腿和躯干不一样长 | body_geometry, disjoint_inequality | S1T171 |
| 腿短，低头。 | 1 | 低头 | other_unparsed | S1T100 |

#### S130

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿和尾巴都比脖子长。 | 16 | 腿和尾巴都比脖子长 | other_unparsed | S1T180, S1T244, S1T245, S1T248, S1T250, S1T252, S1T253, S1T256 |
| 头身比例比较协调。 | 5 | 头身比例比较协调 | proportion_or_ratio | S1T30, S1T31, S1T32, S1T33, S1T36 |
| 尾巴和腿都比脖子短。 | 5 | 尾巴和腿都比脖子短 | other_unparsed | S1T280, S1T281, S1T282, S1T284, S1T285 |
| 头身比例不协调。 | 4 | 头身比例不协调 | proportion_or_ratio | S1T29, S1T34, S1T35, S1T37 |
| 像爬行类的动物。 | 2 | 像爬行类的动物 | other_unparsed | S1T113, S1T114 |
| 腿和尾巴都比脖子短。 | 2 | 腿和尾巴都比脖子短 | other_unparsed | S1T276, S1T288 |
| 像直立行走的动物。 | 1 | 像直立行走的动物 | other_unparsed | S1T122 |
| 头和尾巴都比脖子长。 | 1 | 头和尾巴都比脖子长 | other_unparsed | S1T184 |
| 头和脖子都比腿短。 | 1 | 头和脖子都比腿短 | other_unparsed | S1T8 |
| 比例看起来不是很协调。 | 1 | 比例看起来不是很协调 | proportion_or_ratio | S1T20 |

#### S131

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 体型分布的不均匀。 | 4 | 体型分布的不均匀 | global_balance, vague_size | S1T70, S1T71, S1T79, S1T81 |
| 体型分布的很均匀。 | 3 | 体型分布的很均匀 | global_balance, vague_size | S1T74, S1T75, S1T77 |
| 朝左边。 | 3 | 朝左边 | other_unparsed | S1T27, S1T51, S1T53 |
| 它像一个小型的动物。 | 2 | 它像一个小型的动物 | other_unparsed | S1T65, S1T67 |
| 朝右边。 | 2 | 朝右边 | other_unparsed | S1T52, S1T54 |
| 腿比头长，朝右边。 | 2 | 朝右边 | other_unparsed | S1T42, S1T47 |
| 体型分布得不均匀。 | 1 | 体型分布得不均匀 | global_balance, vague_size | S1T80 |
| 体型分布得很均匀。 | 1 | 体型分布得很均匀 | global_balance, vague_size | S1T76 |
| 体型分布的不均匀，且方向是朝左。 | 1 | 体型分布的不均匀; 且方向是朝左 | global_balance, vague_size | S1T82 |
| 体型分布的还算均匀，头很长。 | 1 | 体型分布的还算均匀 | global_balance, vague_size | S1T72 |
| 体型比较大且分布均匀。 | 1 | 体型比较大且分布均匀 | global_balance, vague_size | S1T69 |
| 体型看上去很大，各个部位都很长。 | 1 | 体型看上去很大 | vague_size | S1T15 |
| 各部位分布均匀，都很长。 | 1 | 都很长 | other_unparsed | S1T64 |
| 头长，腿很长，分布均匀。 | 1 | 分布均匀 | global_balance | S1T68 |
| 它像一个大型的动物。 | 1 | 它像一个大型的动物 | other_unparsed | S1T66 |
| 尾巴很长，朝右侧。 | 1 | 朝右侧 | other_unparsed | S1T25 |
| 尾巴长，腿短，是个大型动物。 | 1 | 是个大型动物 | other_unparsed | S1T130 |
| 是个大型动物。 | 1 | 是个大型动物 | other_unparsed | S1T131 |
| 朝右边，腿长。 | 1 | 朝右边 | other_unparsed | S1T55 |
| 朝左侧，尾巴很短。 | 1 | 朝左侧 | other_unparsed | S1T24 |
| 朝左边，头长。 | 1 | 朝左边 | other_unparsed | S1T56 |
| 朝左边，腿长，头长。 | 1 | 朝左边 | other_unparsed | S1T57 |
| 腿很长，个子很高。 | 1 | 个子很高 | other_unparsed | S1T83 |
| 腿很长，分布得很均匀。 | 1 | 分布得很均匀 | global_balance | S1T73 |
| 腿很长，尾巴很短，像一条狗。 | 1 | 像一条狗 | other_unparsed | S1T84 |
| 腿比头短，朝右边。 | 1 | 朝右边 | other_unparsed | S1T48 |
| 腿比头短，朝左边。 | 1 | 朝左边 | other_unparsed | S1T43 |
| 腿比头长，朝左边。 | 1 | 朝左边 | other_unparsed | S1T41 |
| 腿短，不均匀。 | 1 | 不均匀 | global_balance | S1T63 |

#### S132

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴不是最短的。 | 8 | 尾巴不是最短的 | other_unparsed | S1T118, S1T144, S1T145, S1T146, S1T147, S1T149, S1T151, S1T154 |
| 尾巴不是最长的。 | 3 | 尾巴不是最长的 | other_unparsed | S1T93, S1T152, S1T153 |
| 尾巴短于某个数值。 | 2 | 尾巴短于某个数值 | other_unparsed | S2T14, S2T15 |
| 尾巴长于某个数值。 | 1 | 尾巴长于某个数值 | other_unparsed | S2T16 |

#### S202

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S1T209 |

#### S203

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴均长于脖子，腿很短。 | 1 | 头和尾巴均长于脖子 | other_unparsed | S1T97 |
| 头和尾巴很长，长于脖子，长于腿。 | 1 | 长于脖子; 长于腿 | other_unparsed | S1T137 |
| 头显著短、短于脖子，尾巴很长，腿很短。 | 1 | 头显著短、短于脖子 | other_unparsed | S1T107 |
| 尾巴很长，头最短，躯干中等。 | 1 | 躯干中等 | body_geometry | S1T155 |
| 所有部位都中等偏长，中等偏短。 | 1 | 中等偏短 | other_unparsed | S2T28 |
| 脖子长于头和尾巴，但都它们都很长，腿很短。 | 1 | 但都它们都很长 | other_unparsed | S1T109 |
| 腿很长，头很短，脖子和尾巴长度对称。 | 1 | 脖子和尾巴长度对称 | other_unparsed | S1T37 |
| 腿很长，尾巴很短，体很长，体中等。 | 1 | 体很长; 体中等 | other_unparsed | S1T303 |
| 腿很长，颈部也很短。 | 1 | 颈部也很短 | other_unparsed | S1T30 |
| 身体各个部位都很匀称。 | 1 | 身体各个部位都很匀称 | body_geometry, global_balance | S1T15 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S2T112 |

#### S204

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位几乎一样长，都挺长。 | 1 | 都挺长 | other_unparsed | S1T130 |
| 四个部位都比较长，长度都差不多。 | 1 | 长度都差不多 | global_balance | S1T203 |
| 头、脖子、尾巴都很长，腿比它们稍微短一点。 | 1 | 腿比它们稍微短一点 | other_unparsed | S1T1 |
| 头和尾巴比较短，脖子和腿长，选错了。 | 1 | 选错了 | meta_or_uncertain | S1T186 |
| 头和尾巴都比脖子长。 | 1 | 头和尾巴都比脖子长 | other_unparsed | S1T113 |
| 头很小，尾巴很长，脖子也很长。 | 1 | 头很小 | vague_size | S1T179 |
| 头很长，腿差不多正好。 | 1 | 腿差不多正好 | other_unparsed | S1T241 |
| 头比尾巴长，脖子也比尾巴长。 | 1 | 脖子也比尾巴长 | other_unparsed | S1T223 |
| 头短，脖子差不多刚好。 | 1 | 脖子差不多刚好 | other_unparsed | S2T91 |
| 头短，脖子长，脖子和躯干差不多。 | 1 | 脖子和躯干差不多 | body_geometry, global_balance | S1T276 |
| 头长，腿长，很均衡。 | 1 | 很均衡 | global_balance | S1T311 |
| 尾巴不比脖子长。 | 1 | 尾巴不比脖子长 | other_unparsed | S1T259 |
| 尾巴相当短，腿相当长。 | 1 | 腿相当长 | global_balance | S1T187 |
| 脖子很长，头很小。 | 1 | 头很小 | vague_size | S1T159 |

#### S205

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子和腿，差不多长。 | 1 | 头、脖子和腿; 差不多长 | global_balance | S1T126 |
| 腿，头最长。 | 1 | 腿 | other_unparsed | S2T224 |
| 腿，尾巴最长。 | 1 | 腿 | other_unparsed | S2T235 |
| 腿，脖子最长。 | 1 | 腿 | other_unparsed | S3T7 |

#### S206

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子比头短，腿也比尾巴短。 | 2 | 腿也比尾巴短 | other_unparsed | S3T82, S3T85 |
| 脖子比头长，腿也比尾巴长。 | 2 | 腿也比尾巴长 | other_unparsed | S3T35, S3T37 |
| 像小狗。 | 1 | 像小狗 | other_unparsed | S2T144 |
| 像小狗一样。 | 1 | 像小狗一样 | other_unparsed | S2T86 |
| 像食蚁兽。 | 1 | 像食蚁兽 | other_unparsed | S2T92 |
| 像食蚁兽，脖子很长。 | 1 | 像食蚁兽 | other_unparsed | S2T64 |
| 头和尾巴比较长，尾巴，腿比较短，脖子也很长。 | 1 | 尾巴 | other_unparsed | S2T7 |
| 头和脖子。 | 1 | 头和脖子 | other_unparsed | S1T27 |
| 头和脖子都比腿短。 | 1 | 头和脖子都比腿短 | other_unparsed | S2T51 |
| 头和脖子都比较小，腿很长。 | 1 | 头和脖子都比较小 | vague_size | S4T74 |
| 头明显比脖子长，腿和尾巴都很长。 | 1 | 头明显比脖子长 | other_unparsed | S4T32 |
| 短长。 | 1 | 短长 | other_unparsed | S1T92 |
| 脖子和头比，脖子很短，腿比尾巴要长。 | 1 | 脖子和头比 | other_unparsed | S3T281 |
| 脖子很短，腿、尾巴，头很长。 | 1 | 腿、尾巴 | other_unparsed | S1T8 |
| 脖子很长，比头长，腿比尾巴长。 | 1 | 比头长 | other_unparsed | S3T184 |
| 脖子比头略长，腿比尾巴略长，它们都很长。 | 1 | 它们都很长 | other_unparsed | S3T305 |
| 腿和脖子，腿比较短，脖子比较长，头比较短。 | 1 | 腿和脖子 | other_unparsed | S3T127 |
| 腿很短，头和脖子，较为长。 | 1 | 头和脖子; 较为长 | other_unparsed | S5T119 |
| 腿比较短，像小狗。 | 1 | 像小狗 | other_unparsed | S2T65 |
| 较为均衡。 | 1 | 较为均衡 | global_balance | S1T158 |
| 较为等长。 | 1 | 较为等长 | other_unparsed | S1T101 |

#### S207

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位都很长，体型很大。 | 4 | 体型很大 | vague_size | S1T157, S1T184, S1T195, S1T210 |
| 四个部位长度差不多，都比较长。 | 2 | 都比较长 | other_unparsed | S1T63, S1T114 |
| 四个部位都很长，体型非常大。 | 1 | 体型非常大 | vague_size | S1T198 |
| 四个部位都比较长，体型比较大。 | 1 | 体型比较大 | vague_size | S1T145 |
| 四个部位都比较长，长度接近。 | 1 | 长度接近 | other_unparsed | S1T101 |
| 四个部位长度相仿。 | 1 | 四个部位长度相仿 | other_unparsed | S1T8 |
| 头、尾巴和腿都非常长，脖子明显比这三个短。 | 1 | 脖子明显比这三个短 | other_unparsed | S1T152 |
| 头、脖子和腿都很长，体型相对比较大。 | 1 | 体型相对比较大 | vague_size | S1T160 |
| 头和脖子一样长，都是最长的，其他两个比较短。 | 1 | 都是最长的 | other_unparsed | S1T40 |
| 头和脖子都非常长，体型比较大。 | 1 | 体型比较大 | vague_size | S1T151 |
| 头和腿很长，比脖子长。 | 1 | 比脖子长 | other_unparsed | S1T247 |
| 头最长，腿和尾巴也很长，和头比较接近，脖子很短。 | 1 | 和头比较接近 | other_unparsed | S1T136 |
| 尾巴很长，和前三个部位长度差不多。 | 1 | 和前三个部位长度差不多 | count_abstract, global_balance | S1T166 |
| 尾巴最长，其他三个部位一样长，都比较长。 | 1 | 都比较长 | other_unparsed | S1T34 |
| 脖子和腿都很长，体型很大。 | 1 | 体型很大 | vague_size | S1T199 |
| 脖子明显比头长很多。 | 1 | 脖子明显比头长很多 | other_unparsed | S2T4 |
| 脖子最长，体型很大。 | 1 | 体型很大 | vague_size | S1T206 |
| 脖子最长，体型比较大。 | 1 | 体型比较大 | vague_size | S1T187 |
| 脖子最长，体型相对比较小。 | 1 | 体型相对比较小 | vague_size | S1T177 |
| 脖子最长，比头长很多。 | 1 | 比头长很多 | other_unparsed | S2T6 |
| 脖子长比头要长一点。 | 1 | 脖子长比头要长一点 | other_unparsed | S3T106 |
| 脖子非常长，四个部位都比较长，体型很大。 | 1 | 体型很大 | vague_size | S1T134 |
| 除了头，都非常长，体型很大。 | 1 | 体型很大 | vague_size | S1T188 |

#### S208

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四者都很长。 | 2 | 四者都很长 | other_unparsed | S1T113, S1T177 |
| 三个长，一个短。 | 1 | 三个长; 一个短 | other_unparsed | S1T172 |
| 两长两短。 | 1 | 两长两短 | count_abstract | S1T297 |
| 四者均匀的很长。 | 1 | 四者均匀的很长 | global_balance | S1T195 |
| 四者有三个很长。 | 1 | 四者有三个很长 | other_unparsed | S1T114 |
| 四者都差不多长。 | 1 | 四者都差不多长 | global_balance | S1T182 |
| 四者都很短，尤其是腿很短。 | 1 | 四者都很短 | other_unparsed | S1T71 |
| 四者都比较长。 | 1 | 四者都比较长 | other_unparsed | S1T111 |
| 四者都较短。 | 1 | 四者都较短 | other_unparsed | S1T120 |
| 均匀的长。 | 1 | 均匀的长 | global_balance | S1T283 |
| 头和尾巴，头和脖子较短，腿和尾巴长。 | 1 | 头和尾巴 | other_unparsed | S1T85 |
| 头和脖子，较短。 | 1 | 头和脖子; 较短 | other_unparsed | S1T284 |
| 头和，四个部位都长。 | 1 | 头和 | other_unparsed | S1T38 |
| 头短，脖，脖子、腿和尾巴都长。 | 1 | 脖子 | other_unparsed | S1T92 |
| 就都挺短。 | 1 | 就都挺短 | other_unparsed | S1T210 |
| 就都挺长。 | 1 | 就都挺长 | other_unparsed | S1T209 |
| 我真的不知道，没什么区别。 | 1 | 我真的不知道; 没什么区别 | meta_or_uncertain | S1T160 |
| 脖子很长，尾巴稍短一些，腿和腿一样长。 | 1 | 腿和腿一样长 | other_unparsed | S1T32 |
| 腿偏短，脖子、尾巴和头都偏长，一样长。 | 1 | 一样长 | other_unparsed | S1T9 |
| 都非常短。 | 1 | 都非常短 | other_unparsed | S1T289 |
| 长度比较均匀。 | 1 | 长度比较均匀 | global_balance | S1T244 |

#### S209

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴长，脖子和腿短，头明显比脖子长。 | 2 | 头明显比脖子长 | other_unparsed | S1T88, S1T160 |
| 头最长，脖子短，腿和尾巴中等，头明显比脖子长。 | 2 | 头明显比脖子长 | other_unparsed | S1T152, S1T198 |
| 头最长，腿中等，脖子和尾巴短，头明显比脖子长。 | 2 | 头明显比脖子长 | other_unparsed | S1T157, S1T201 |
| 头、脖子、腿都长尾巴，中等长度。 | 1 | 中等长度 | other_unparsed | S1T40 |
| 头、脖子和腿，尾巴长，短。 | 1 | 头、脖子和腿; 短 | other_unparsed | S1T58 |
| 头和尾巴中等偏长，脖子和腿短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T139 |
| 头和脖子，所有部位都差不多长，其中头和脖子长度接近，并且是中等偏长。 | 1 | 头和脖子; 并且是中等偏长 | other_unparsed | S1T107 |
| 头最长，尾巴中等，脖子中等偏短，腿短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T186 |
| 头最长，脖子、腿中等，尾巴短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T163 |
| 头最长，脖子中等，尾巴和腿短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T141 |
| 头最长，腿和尾巴中等偏长，脖子短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T153 |
| 头长，尾巴长，脖子和腿中等偏短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T90 |
| 头长，脖子中等，偏长。 | 1 | 偏长 | other_unparsed | S2T124 |
| 头长，脖子短，头、腿、尾巴中等长度，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T169 |
| 头长，脖子短，腿和尾巴中等，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T144 |
| 所有部位都是中等偏长，而且长度差不多。 | 1 | 而且长度差不多 | global_balance | S1T18 |
| 所有部位都是中等长度，而且都差不多长。 | 1 | 而且都差不多长 | global_balance | S1T78 |
| 脖子、腿、尾巴长，头中等偏短，头明显比脖子短。 | 1 | 头明显比脖子短 | other_unparsed | S1T135 |
| 脖子很长，头，中等长度，尾巴和腿比较短。 | 1 | 头; 中等长度 | other_unparsed | S1T21 |
| 脖子长，头中等，总体来说比较长。 | 1 | 总体来说比较长 | other_unparsed | S1T320 |
| 脖子长，腿长，头中等，尾巴短，腿明显比尾巴长。 | 1 | 腿明显比尾巴长 | other_unparsed | S1T191 |
| 腿长，头短，脖子和尾巴中等，并且接近。 | 1 | 并且接近 | other_unparsed | S1T117 |
| 腿长，头短，脖子和尾巴中等，腿明显比尾巴长。 | 1 | 腿明显比尾巴长 | other_unparsed | S1T156 |
| 腿长，尾巴，中等偏短。 | 1 | 尾巴; 中等偏短 | other_unparsed | S1T272 |
| 腿长，脖子长，头和尾巴短，腿明显比尾巴长。 | 1 | 腿明显比尾巴长 | other_unparsed | S1T190 |

#### S210

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 有三个部位几乎一样长。 | 13 | 有三个部位几乎一样长 | count_abstract | S1T29, S1T106, S1T108, S1T142, S1T193, S1T257, S1T260, S1T262 |
| 三个部位几乎一样长。 | 11 | 三个部位几乎一样长 | count_abstract | S1T170, S1T190, S1T223, S1T224, S1T241, S1T290, S1T310, S1T314 |
| 有两个部位几乎一样长。 | 9 | 有两个部位几乎一样长 | count_abstract | S1T30, S1T38, S1T135, S1T138, S1T141, S1T144, S1T167, S1T259 |
| 有三个部位长度一样。 | 7 | 有三个部位长度一样 | count_abstract | S3T189, S3T190, S3T207, S3T210, S3T216, S3T236, S4T216 |
| 有三个部位一样长。 | 6 | 有三个部位一样长 | count_abstract | S1T64, S1T210, S2T199, S3T23, S3T51, S3T108 |
| 两个部位一样长。 | 5 | 两个部位一样长 | count_abstract | S1T86, S1T90, S1T93, S1T179, S1T180 |
| 两个部位几乎一样长。 | 5 | 两个部位几乎一样长 | count_abstract | S1T164, S2T9, S2T76, S2T112, S2T116 |
| 三个部位长度一样。 | 4 | 三个部位长度一样 | count_abstract | S3T82, S3T83, S3T120, S3T122 |
| 躯干最长。 | 3 | 躯干最长 | body_geometry | S1T75, S1T78, S1T81 |
| 三个部位一样长。 | 2 | 三个部位一样长 | count_abstract | S2T272, S2T278 |
| 三个部位长度相似。 | 2 | 三个部位长度相似 | count_abstract | S1T278, S4T51 |
| 两两长度一样。 | 2 | 两两长度一样 | other_unparsed | S2T214, S2T217 |
| 头。 | 2 | 头 | other_unparsed | S4T294, S5T134 |
| 有两个部位一样长。 | 2 | 有两个部位一样长 | count_abstract | S1T40, S1T255 |
| 躯干是最长。 | 2 | 躯干是最长 | body_geometry | S1T39, S1T73 |
| 都差不多长。 | 2 | 都差不多长 | global_balance | S2T242, S2T312 |
| 长度两两相似。 | 2 | 长度两两相似 | other_unparsed | S3T10, S3T118 |
| 三个部位一样长，头最小。 | 1 | 三个部位一样长 | count_abstract | S4T300 |
| 三个部位明显长。 | 1 | 三个部位明显长 | count_abstract | S1T161 |
| 三个部位相似，脖子最长。 | 1 | 三个部位相似 | count_abstract | S4T84 |
| 三个部位相似，腿最短。 | 1 | 三个部位相似 | count_abstract | S4T83 |
| 三个部位，长最长，尾巴最短。 | 1 | 三个部位; 长最长 | count_abstract | S4T82 |
| 两两一样长。 | 1 | 两两一样长 | other_unparsed | S2T244 |
| 两两长度相似。 | 1 | 两两长度相似 | other_unparsed | S3T162 |
| 两部位一样长。 | 1 | 两部位一样长 | other_unparsed | S1T127 |
| 两部位长相似。 | 1 | 两部位长相似 | other_unparsed | S1T303 |
| 几个部位几乎一样长。 | 1 | 几个部位几乎一样长 | count_abstract | S1T235 |
| 几个部位的长度差不多。 | 1 | 几个部位的长度差不多 | count_abstract, global_balance | S3T196 |
| 几个部位都比躯干短。 | 1 | 几个部位都比躯干短 | count_abstract, body_geometry | S1T58 |
| 几个部位长度一样。 | 1 | 几个部位长度一样 | count_abstract | S4T140 |
| 几个部位长度差别不大。 | 1 | 几个部位长度差别不大 | count_abstract | S3T31 |
| 几个部位长度相差不大。 | 1 | 几个部位长度相差不大 | count_abstract | S3T145 |
| 又有两个部位一样长。 | 1 | 又有两个部位一样长 | count_abstract | S1T32 |
| 各部位都差不多，较短，尾巴最短。 | 1 | 较短 | other_unparsed | S5T74 |
| 四个部位都很长，且几乎一样长。 | 1 | 且几乎一样长 | other_unparsed | S5T16 |
| 头和尾巴一样长，都是最长。 | 1 | 都是最长 | other_unparsed | S3T102 |
| 头和尾巴长度一样，都最短，腿第二短，脖子很长。 | 1 | 都最短 | other_unparsed | S4T9 |
| 头和脖子最长，长于腿、长于尾巴。 | 1 | 长于腿、长于尾巴 | other_unparsed | S4T30 |
| 头和脖子，腿和尾巴比较短。 | 1 | 头和脖子 | other_unparsed | S5T14 |
| 头和腿几乎一样长，而且最短。 | 1 | 而且最短 | other_unparsed | S2T32 |
| 头比较长，脖子，尾巴最短。 | 1 | 脖子 | other_unparsed | S5T155 |
| 头，头比脖子短。 | 1 | 头 | other_unparsed | S2T158 |
| 头，尾巴最短。 | 1 | 头 | other_unparsed | S3T168 |
| 头，第三长。 | 1 | 头; 第三长 | count_abstract, ordinal_or_secondary | S2T101 |
| 尾巴非常短，其他部位都，也比较短。 | 1 | 其他部位都; 也比较短 | other_reference | S5T33 |
| 有三个部位比躯干长。 | 1 | 有三个部位比躯干长 | count_abstract, body_geometry | S1T137 |
| 有三个部位都一样长。 | 1 | 有三个部位都一样长 | count_abstract | S3T2 |
| 有三个部位长度一致。 | 1 | 有三个部位长度一致 | count_abstract | S3T172 |
| 有两个部位长度一样。 | 1 | 有两个部位长度一样 | count_abstract | S3T80 |
| 有两个部位长得相似。 | 1 | 有两个部位长得相似 | count_abstract | S1T299 |
| 脖子、尾巴、腿的长度，一样。 | 1 | 脖子、尾巴、腿的长度; 一样 | other_unparsed | S3T199 |
| 脖子。 | 1 | 脖子 | other_unparsed | S2T39 |
| 脖子和尾巴，几乎一样长。 | 1 | 脖子和尾巴; 几乎一样长 | other_unparsed | S2T36 |
| 腿和尾巴很长，另两个部位非常短。 | 1 | 另两个部位非常短 | count_abstract | S5T77 |
| 腿，头和尾巴一样长。 | 1 | 腿 | other_unparsed | S3T146 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S5T316 |
| 都是中等长度。 | 1 | 都是中等长度 | other_unparsed | S5T51 |
| 长度差不多。 | 1 | 长度差不多 | global_balance | S3T229 |
| 长度相似，脖子最短。 | 1 | 长度相似 | other_unparsed | S4T79 |

#### S211

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度差不多，比较长。 | 1 | 比较长 | other_unparsed | S3T57 |
| 头和尾巴都比较长，一样长，腿也比较长。 | 1 | 一样长 | other_unparsed | S2T175 |
| 头和脖子，尾巴和腿中有三个是一样长。 | 1 | 头和脖子 | other_unparsed | S1T57 |
| 头最长，比脖子和尾巴都长。 | 1 | 比脖子和尾巴都长 | other_unparsed | S1T1 |
| 头，和腿基本一样长。 | 1 | 头; 和腿基本一样长 | other_unparsed | S1T55 |
| 都不太长，只有腿最短。 | 1 | 都不太长 | other_unparsed | S3T58 |

#### S212

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位较均等。 | 17 | 四个部位较均等 | global_balance | S1T61, S1T65, S1T68, S1T72, S1T93, S1T119, S1T124, S1T126 |
| 头比脖子短，比尾巴短。 | 10 | 比尾巴短 | other_unparsed | S4T138, S4T139, S4T145, S4T146, S4T151, S4T152, S4T154, S4T155 |
| 头比脖子长，比尾巴短。 | 7 | 比尾巴短 | other_unparsed | S4T93, S4T140, S4T142, S4T147, S4T148, S4T163, S4T166 |
| 头比脖子长，比尾巴长。 | 7 | 比尾巴长 | other_unparsed | S4T141, S4T144, S4T149, S4T150, S4T153, S4T168, S4T195 |
| 尾巴比脖子长，比头长。 | 7 | 比头长 | other_unparsed | S3T258, S3T261, S3T271, S3T272, S4T88, S4T91, S4T92 |
| 四个部位较匀称。 | 6 | 四个部位较匀称 | global_balance | S1T182, S1T242, S1T246, S1T255, S1T259, S2T24 |
| 尾巴比脖子短，比头短。 | 6 | 比头短 | other_unparsed | S3T257, S3T262, S3T265, S4T87, S4T89, S4T90 |
| 脖子比头短，比尾巴短。 | 6 | 比尾巴短 | other_unparsed | S4T96, S4T97, S4T98, S4T101, S4T105, S4T110 |
| 脖子比头长，比尾巴长。 | 5 | 比尾巴长 | other_unparsed | S4T94, S4T99, S4T103, S4T106, S4T109 |
| 头比脖子短，比尾巴长。 | 4 | 比尾巴长 | other_unparsed | S4T137, S4T161, S4T167, S4T194 |
| 尾巴长，头的位置高。 | 4 | 头的位置高 | other_unparsed | S6T75, S6T76, S6T77, S6T81 |
| 腿比脖子短，比尾巴短。 | 4 | 比尾巴短 | other_unparsed | S4T170, S4T173, S4T174, S4T176 |
| 腿比脖子长，比尾巴长。 | 4 | 比尾巴长 | other_unparsed | S4T165, S4T172, S4T175, S4T177 |
| 四个部位较为匀称。 | 3 | 四个部位较为匀称 | global_balance | S1T236, S1T274, S1T278 |
| 四个部位长度较均等。 | 3 | 四个部位长度较均等 | global_balance | S1T8, S1T41, S1T57 |
| 尾巴比脖子短，比头长。 | 3 | 比头长 | other_unparsed | S3T259, S3T264, S4T86 |
| 尾巴长，头的位置低。 | 3 | 头的位置低 | other_unparsed | S6T74, S6T78, S6T80 |
| 脖子比头长，比尾巴短。 | 3 | 比尾巴短 | other_unparsed | S4T95, S4T100, S4T104 |
| 尾巴比脖子长，比头短。 | 2 | 比头短 | other_unparsed | S3T260, S3T263 |
| 尾巴比躯干短，比脖子短。 | 2 | 比脖子短 | other_unparsed | S3T316, S3T318 |
| 脖子和尾巴都比腿长。 | 2 | 脖子和尾巴都比腿长 | other_unparsed | S2T226, S2T228 |
| 脖子比尾巴短，比腿长。 | 2 | 比腿长 | other_unparsed | S4T190, S4T191 |
| 腿比脖子长，比尾巴短。 | 2 | 比尾巴短 | other_unparsed | S4T169, S4T171 |
| 上身长，腿短。 | 1 | 上身长 | other_unparsed | S2T294 |
| 四个部位均等。 | 1 | 四个部位均等 | global_balance | S1T9 |
| 四个部位的长度较均等。 | 1 | 四个部位的长度较均等 | global_balance | S1T6 |
| 四个部位较均等，腿较短。 | 1 | 四个部位较均等 | global_balance | S1T125 |
| 四个部位都很小。 | 1 | 四个部位都很小 | vague_size | S1T85 |
| 头比脖子长，比腿长。 | 1 | 比腿长 | other_unparsed | S4T143 |
| 尾巴比头短，比脖子短。 | 1 | 比脖子短 | other_unparsed | S3T270 |
| 尾巴比躯干短，比脖子长。 | 1 | 比脖子长 | other_unparsed | S3T319 |
| 尾巴短，头位置高。 | 1 | 头位置高 | other_unparsed | S6T73 |
| 尾巴短，头的位置低。 | 1 | 头的位置低 | other_unparsed | S6T79 |
| 尾巴长，头低。 | 1 | 头低 | other_unparsed | S6T72 |
| 脖子和尾巴一样长，比腿长。 | 1 | 比腿长 | other_unparsed | S2T243 |
| 脖子和尾巴短，比头长。 | 1 | 比头长 | other_unparsed | S3T255 |
| 脖子和尾巴都比腿短。 | 1 | 脖子和尾巴都比腿短 | other_unparsed | S2T225 |
| 脖子比头短，比尾巴长。 | 1 | 比尾巴长 | other_unparsed | S4T107 |
| 脖子比头长，比腿长，比尾巴长。 | 1 | 比腿长; 比尾巴长 | other_unparsed | S4T102 |
| 脖子比尾巴短，比头短。 | 1 | 比头短 | other_unparsed | S3T269 |
| 脖子比尾巴短，比头长。 | 1 | 比头长 | other_unparsed | S3T256 |
| 脖子比尾巴短，比腿短。 | 1 | 比腿短 | other_unparsed | S4T192 |
| 脖子比尾巴长，比头长。 | 1 | 比头长 | other_unparsed | S3T254 |
| 脖子比腿和尾巴短，比头短。 | 1 | 比头短 | other_unparsed | S4T108 |
| 腿比尾巴短，比脖子短。 | 1 | 比脖子短 | other_unparsed | S4T179 |
| 腿比脖子短，比尾巴长。 | 1 | 比尾巴长 | other_unparsed | S4T178 |
| 腿长，上身短。 | 1 | 上身短 | other_unparsed | S2T293 |

#### S213

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 有两个部位比较长。 | 7 | 有两个部位比较长 | count_abstract | S2T306, S2T309, S2T310, S2T312, S2T314, S2T317, S2T318 |
| 有三个部位比较长。 | 3 | 有三个部位比较长 | count_abstract | S2T305, S2T311, S2T316 |
| 有一个部位比较长。 | 2 | 有一个部位比较长 | count_abstract | S2T308, S2T315 |
| 脖子比头长，脖子不是最长。 | 2 | 脖子不是最长 | other_unparsed | S2T297, S2T301 |
| 腿较短，头和尾巴均长于脖子。 | 2 | 头和尾巴均长于脖子 | other_unparsed | S2T43, S2T44 |
| 四部位差不多长。 | 1 | 四部位差不多长 | global_balance | S1T56 |
| 头和尾巴比较长，长于脖子。 | 1 | 长于脖子 | other_unparsed | S2T65 |
| 头和尾巴都比脖子长。 | 1 | 头和尾巴都比脖子长 | other_unparsed | S1T99 |
| 头明显比脖子长，尾巴比较长。 | 1 | 头明显比脖子长 | other_unparsed | S3T81 |
| 头显著比脖子长。 | 1 | 头显著比脖子长 | other_unparsed | S4T18 |
| 头比尾巴短，比脖子短。 | 1 | 比脖子短 | other_unparsed | S1T131 |
| 头比脖子长，头不是最长。 | 1 | 头不是最长 | other_unparsed | S2T298 |
| 头比脖子长，脖子不是最短。 | 1 | 脖子不是最短 | other_unparsed | S2T300 |
| 头长，脖子长，尾巴长，腿和尾巴。 | 1 | 腿和尾巴 | other_unparsed | S1T91 |
| 脖子显著比头长。 | 1 | 脖子显著比头长 | other_unparsed | S4T17 |
| 腿和脖子都比尾巴长。 | 1 | 腿和脖子都比尾巴长 | other_unparsed | S3T161 |
| 腿比躯干长，头不比脖子短。 | 1 | 头不比脖子短 | other_unparsed | S2T24 |
| 腿比较长，头和脖子无明显差距。 | 1 | 头和脖子无明显差距 | other_unparsed | S3T167 |
| 腿比较长，头和脖子，四个部位都比较短。 | 1 | 头和脖子 | other_unparsed | S4T171 |
| 腿较长，头和脖子均长于尾巴。 | 1 | 头和脖子均长于尾巴 | other_unparsed | S2T42 |

#### S214

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 选错了。 | 39 | 选错了 | meta_or_uncertain | S1T97, S1T131, S1T148, S1T166, S1T249, S1T311, S2T27, S2T53 |
| 两长两短。 | 3 | 两长两短 | count_abstract | S3T220, S3T221, S3T222 |
| 差不多。 | 3 | 差不多 | global_balance | S2T35, S2T110, S4T18 |
| 三个中等，尾巴短。 | 1 | 三个中等 | other_unparsed | S3T139 |
| 四个部位都还行。 | 1 | 四个部位都还行 | other_unparsed | S5T246 |
| 头，和，腿、头和腿略短，其他中等长度。 | 1 | 头; 和 | other_unparsed | S1T60 |
| 头，脖子特别长，腿特别短。 | 1 | 头 | other_unparsed | S1T188 |
| 差距不大。 | 1 | 差距不大 | other_unparsed | S3T189 |
| 是长。 | 1 | 是长 | other_unparsed | S2T184 |
| 腿长，脖，脖子中等长度，尾巴略长，头略短。 | 1 | 脖子 | other_unparsed | S1T44 |
| 腿，最短。 | 1 | 腿; 最短 | other_unparsed | S3T180 |

#### S215

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 两个部位长，两个部位短。 | 7 | 两个部位长; 两个部位短 | count_abstract | S2T52, S2T62, S2T63, S2T87, S2T225, S2T232, S2T233 |
| 三个部位长。 | 6 | 三个部位长 | count_abstract | S2T35, S2T50, S2T184, S2T188, S2T189, S2T194 |
| 三个部位长，一个部位短。 | 6 | 三个部位长; 一个部位短 | count_abstract | S2T51, S2T57, S2T58, S2T64, S2T231, S2T236 |
| 三个部位短，一个部位长。 | 3 | 三个部位短; 一个部位长 | count_abstract | S2T59, S2T86, S2T237 |
| 三长一短。 | 3 | 三长一短 | count_abstract | S2T175, S2T238, S2T240 |
| 两个部位长。 | 3 | 两个部位长 | count_abstract | S2T185, S2T193, S2T195 |
| 两长两短。 | 2 | 两长两短 | count_abstract | S2T173, S2T174 |
| 只有一个部位长。 | 2 | 只有一个部位长 | count_abstract | S2T183, S2T187 |
| 一个部位长，三个部位短。 | 1 | 一个部位长; 三个部位短 | count_abstract | S2T229 |
| 三短一长。 | 1 | 三短一长 | other_unparsed | S2T239 |
| 三长一短，头最短。 | 1 | 三长一短 | count_abstract | S2T176 |
| 头和尾巴的长度超过腿。 | 1 | 头和尾巴的长度超过腿 | other_unparsed | S1T17 |
| 腿和尾巴不一样长，头比脖子长。 | 1 | 腿和尾巴不一样长 | disjoint_inequality | S2T282 |
| 腿和尾巴不一样，头和脖子不一样长。 | 1 | 腿和尾巴不一样; 头和脖子不一样长 | disjoint_inequality | S2T284 |
| 腿和尾巴都长，头和脖子不一样长。 | 1 | 头和脖子不一样长 | disjoint_inequality | S2T280 |

#### S216

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴和腿差不多长，都比脖子长。 | 2 | 都比脖子长 | other_unparsed | S2T121, S2T122 |
| 头、脖子、腿都很长，尾巴中等，微微偏短。 | 1 | 微微偏短 | other_unparsed | S1T24 |
| 尾巴和腿差不多长，偏短。 | 1 | 偏短 | other_unparsed | S1T319 |
| 尾巴和腿都一样长，偏短。 | 1 | 偏短 | other_unparsed | S1T294 |
| 尾巴最长，头和脖子都比尾巴短。 | 1 | 头和脖子都比尾巴短 | other_unparsed | S2T252 |
| 尾巴最长，比腿长也比脖子长。 | 1 | 比腿长也比脖子长 | other_unparsed | S2T115 |
| 尾巴比腿长，也比脖子长。 | 1 | 也比脖子长 | other_unparsed | S2T73 |
| 每个部位。 | 1 | 每个部位 | other_unparsed | S1T105 |
| 每个部分都差不多长。 | 1 | 每个部分都差不多长 | global_balance | S2T267 |
| 每个部分都挺长。 | 1 | 每个部分都挺长 | other_unparsed | S2T253 |
| 每个部分都短。 | 1 | 每个部分都短 | other_unparsed | S2T276 |
| 脖子和尾巴都挺，腿都挺长。 | 1 | 脖子和尾巴都挺 | other_unparsed | S2T206 |
| 脖子比腿，腿和尾巴长。 | 1 | 脖子比腿 | other_unparsed | S2T83 |
| 腿和尾巴一样长，偏长。 | 1 | 偏长 | other_unparsed | S2T37 |
| 腿最长，尾巴最短，脖子居中。 | 1 | 脖子居中 | other_unparsed | S2T141 |
| 都差不多长。 | 1 | 都差不多长 | global_balance | S2T214 |
| 都差不多长，中等长度。 | 1 | 都差不多长; 中等长度 | global_balance | S2T292 |

#### S217

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴长，有两个部位短。 | 2 | 有两个部位短 | count_abstract | S4T74, S4T75 |
| 头比和脖子比较长。 | 1 | 头比和脖子比较长 | other_unparsed | S2T97 |
| 由长到短是头、腿、脖子，尾巴。 | 1 | 尾巴 | other_unparsed | S1T240 |
| 由长到短是头和腿，脖子和尾巴。 | 1 | 脖子和尾巴 | other_unparsed | S1T226 |
| 都比躯干短。 | 1 | 都比躯干短 | body_geometry | S4T126 |

#### S218

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子和尾巴都较长，明显长于腿。 | 4 | 明显长于腿 | other_unparsed | S3T119, S3T150, S3T165, S3T215 |
| 脖子、尾巴都长于腿。 | 3 | 脖子、尾巴都长于腿 | other_unparsed | S4T69, S4T74, S4T76 |
| 脖子和尾巴都较短，明显短于腿。 | 3 | 明显短于腿 | other_unparsed | S3T93, S3T118, S4T167 |
| 头、尾巴长度明显长于脖子、腿。 | 2 | 头、尾巴长度明显长于脖子、腿 | other_unparsed | S3T145, S3T151 |
| 尾巴明显长于其余三部位。 | 2 | 尾巴明显长于其余三部位 | other_reference | S4T282, S4T311 |
| 尾巴长度明显长于脖子。 | 2 | 尾巴长度明显长于脖子 | other_unparsed | S4T27, S4T28 |
| 脖子、尾巴较长，且明显长于腿。 | 2 | 且明显长于腿 | other_unparsed | S3T47, S3T86 |
| 脖子和尾巴较长，明显长于腿。 | 2 | 明显长于腿 | other_unparsed | S3T207, S3T208 |
| 脖子和尾巴都明显长于腿。 | 2 | 脖子和尾巴都明显长于腿 | other_unparsed | S3T193, S4T198 |
| 脖子和尾巴都较短，明显短于头。 | 2 | 明显短于头 | other_unparsed | S3T189, S4T160 |
| 脖子明显长于腿，略长于尾巴。 | 2 | 略长于尾巴 | other_unparsed | S4T127, S4T159 |
| 脖子最长，明显长于尾巴。 | 2 | 明显长于尾巴 | other_unparsed | S3T48, S3T68 |
| 脖子长度明显长于尾巴和腿。 | 2 | 脖子长度明显长于尾巴和腿 | other_unparsed | S3T155, S3T156 |
| 四个部位都略长，长度相近。 | 1 | 长度相近 | other_unparsed | S3T101 |
| 四个部位都较短，且长度较相近。 | 1 | 且长度较相近 | other_unparsed | S3T136 |
| 四个部位都较长，且几乎最长。 | 1 | 且几乎最长 | other_unparsed | S3T164 |
| 四个部位都较长，且尾巴微长于脖子。 | 1 | 且尾巴微长于脖子 | other_unparsed | S4T77 |
| 四个部位都较长，且长度较相近。 | 1 | 且长度较相近 | other_unparsed | S3T137 |
| 四个部位都较长，脖子最长，头，尾巴稍短。 | 1 | 头 | other_unparsed | S2T113 |
| 头、脖子、腿、尾巴长度相近，且长度中等。 | 1 | 且长度中等 | other_unparsed | S1T44 |
| 头和脖子较长，腿和尾巴较短，且几乎最短。 | 1 | 且几乎最短 | other_unparsed | S2T156 |
| 头和脖子较长，腿和尾巴较短，且最短。 | 1 | 且最短 | other_unparsed | S2T135 |
| 头明显长于其他三个部位。 | 1 | 头明显长于其他三个部位 | count_abstract, other_reference | S4T2 |
| 头明显长于其余三部位。 | 1 | 头明显长于其余三部位 | other_reference | S4T283 |
| 头明显长于脖子，略长于尾巴和腿。 | 1 | 略长于尾巴和腿 | other_unparsed | S4T128 |
| 头最短，尾巴轻，尾巴较短，脖子较短。 | 1 | 尾巴轻 | other_unparsed | S2T20 |
| 头最长，脖子、腿、尾巴较短，且几乎最短。 | 1 | 且几乎最短 | other_unparsed | S2T189 |
| 头最长，脖子略长于尾巴，腿。 | 1 | 腿 | other_unparsed | S4T72 |
| 头最长，脖子，腿较短，尾巴较长。 | 1 | 脖子 | other_unparsed | S1T92 |
| 头最长，脖子，腿，稍短。 | 1 | 脖子; 腿; 稍短 | other_unparsed | S1T282 |
| 头略长于其余三部位。 | 1 | 头略长于其余三部位 | other_reference | S3T285 |
| 头达到头最长，脖子略短，腿短于尾巴，且都较短。 | 1 | 且都较短 | other_unparsed | S2T239 |
| 头达到最，头较长，脖子、尾巴较短，腿略长于脖子。 | 1 | 头达到最 | other_unparsed | S2T235 |
| 头，尾巴最短，脖子和腿较长。 | 1 | 头 | other_unparsed | S1T128 |
| 头，尾巴较长，脖子稍短，腿最短。 | 1 | 头 | other_unparsed | S2T225 |
| 头，腿较长。 | 1 | 头 | other_unparsed | S1T293 |
| 尾巴、腿都较长，且略长于脖子。 | 1 | 且略长于脖子 | other_unparsed | S4T32 |
| 尾巴和腿略长于脖子，且都较长。 | 1 | 且都较长 | other_unparsed | S4T52 |
| 尾巴明显长于其他三个部位。 | 1 | 尾巴明显长于其他三个部位 | count_abstract, other_reference | S3T116 |
| 尾巴明显长于头，且长于脖子和腿。 | 1 | 且长于脖子和腿 | other_unparsed | S4T199 |
| 尾巴明显长于脖子，且长于腿。 | 1 | 且长于腿 | other_unparsed | S4T168 |
| 尾巴明显长于脖子，腿。 | 1 | 腿 | other_unparsed | S4T116 |
| 尾巴最短，体腿最长，最长。 | 1 | 最长 | other_unparsed | S1T167 |
| 尾巴最短，其余部位较长，且几乎最长。 | 1 | 且几乎最长 | other_unparsed | S2T55 |
| 尾巴最长，其余部位较短，且几乎最短。 | 1 | 且几乎最短 | other_unparsed | S2T116 |
| 尾巴最长，腿稍短，头，脖子较短。 | 1 | 头 | other_unparsed | S2T130 |
| 尾巴略长于脖子，且都明显短于腿。 | 1 | 且都明显短于腿 | other_unparsed | S3T250 |
| 尾巴约等于腿，远长于脖子。 | 1 | 远长于脖子 | other_unparsed | S4T43 |
| 尾巴长度明显长于脖子、腿。 | 1 | 尾巴长度明显长于脖子、腿 | other_unparsed | S3T157 |
| 尾巴，头最长，腿较短。 | 1 | 尾巴 | other_unparsed | S1T295 |
| 脖子、头、尾巴达到较长，腿最短，最短。 | 1 | 最短 | other_unparsed | S2T90 |
| 脖子、尾巴都明显短于腿，脖子略长。 | 1 | 脖子、尾巴都明显短于腿 | other_unparsed | S3T194 |
| 脖子、尾巴都短于腿。 | 1 | 脖子、尾巴都短于腿 | other_unparsed | S4T73 |
| 脖子、尾巴长度明显长于腿。 | 1 | 脖子、尾巴长度明显长于腿 | other_unparsed | S3T154 |
| 脖子、腿较长，明显长于尾巴。 | 1 | 明显长于尾巴 | other_unparsed | S3T94 |
| 脖子和尾巴明显略短于腿。 | 1 | 脖子和尾巴明显略短于腿 | other_unparsed | S3T203 |
| 脖子和尾巴明显短于其余部位。 | 1 | 脖子和尾巴明显短于其余部位 | other_reference | S4T312 |
| 脖子和尾巴明显长于腿，明显短于腿。 | 1 | 明显短于腿 | other_unparsed | S4T99 |
| 脖子和尾巴最长，明显长于腿。 | 1 | 明显长于腿 | other_unparsed | S4T29 |
| 脖子和尾巴略短于腿，且都较长，长度相近。 | 1 | 且都较长; 长度相近 | other_unparsed | S4T5 |
| 脖子和尾巴较短，明显短于腿。 | 1 | 明显短于腿 | other_unparsed | S3T229 |
| 脖子和尾巴较短，明显短于腿和头。 | 1 | 明显短于腿和头 | other_unparsed | S3T98 |
| 脖子和尾巴都短于腿。 | 1 | 脖子和尾巴都短于腿 | other_unparsed | S4T75 |
| 脖子和尾巴都较长，明显长于腿，头最长。 | 1 | 明显长于腿 | other_unparsed | S3T166 |
| 脖子和尾巴长度明显长于头和腿。 | 1 | 脖子和尾巴长度明显长于头和腿 | other_unparsed | S3T138 |
| 脖子和尾巴，最长，头和腿最短。 | 1 | 脖子和尾巴; 最长 | other_unparsed | S2T143 |
| 脖子和尾巴，略长于腿，头最长。 | 1 | 脖子和尾巴; 略长于腿 | other_unparsed | S3T283 |
| 脖子和腿长度明显长于尾巴。 | 1 | 脖子和腿长度明显长于尾巴 | other_unparsed | S3T153 |
| 脖子明显长于尾巴，且长于腿。 | 1 | 且长于腿 | other_unparsed | S3T127 |
| 脖子明显长于尾巴，尾巴长度约等于腿。 | 1 | 尾巴长度约等于腿 | other_unparsed | S4T6 |
| 脖子最短，其余部位都较长，且几乎最长。 | 1 | 且几乎最长 | other_unparsed | S2T75 |
| 脖子最短，尾巴略长于脖子，且都小于头。 | 1 | 且都小于头 | other_unparsed | S3T129 |
| 脖子最长，尾巴稍短，且明显长于腿。 | 1 | 且明显长于腿 | other_unparsed | S3T78 |
| 脖子最长，明显长于尾巴、腿。 | 1 | 明显长于尾巴、腿 | other_unparsed | S3T95 |
| 脖子最长，长于尾巴，腿稍短。 | 1 | 长于尾巴 | other_unparsed | S3T26 |
| 脖子略长于尾巴，且两者明显短于腿。 | 1 | 且两者明显短于腿 | other_unparsed | S3T173 |
| 脖子略长于尾巴，两者都明显短于腿。 | 1 | 两者都明显短于腿 | other_unparsed | S3T114 |
| 脖子短于尾巴，且短于腿。 | 1 | 且短于腿 | other_unparsed | S4T235 |
| 脖子短于尾巴，两者都短于腿。 | 1 | 两者都短于腿 | other_unparsed | S3T110 |
| 脖子短，尾巴短，且都最短。 | 1 | 且都最短 | other_unparsed | S3T8 |
| 脖子较长，腿，尾巴较长，头较短。 | 1 | 腿 | other_unparsed | S1T272 |
| 脖子长于尾巴，且明显长于腿。 | 1 | 且明显长于腿 | other_unparsed | S3T109 |
| 脖子长度明显短于尾巴，脖子也短于腿。 | 1 | 脖子长度明显短于尾巴; 脖子也短于腿 | other_unparsed | S3T204 |
| 脖子长度约等于腿，且明显长于尾巴。 | 1 | 脖子长度约等于腿; 且明显长于尾巴 | other_unparsed | S4T25 |
| 脖子长约等于腿，且明显大于尾巴。 | 1 | 脖子长约等于腿; 且明显大于尾巴 | other_unparsed | S4T134 |
| 腿最短，脖子、头、尾巴都较长，且几乎最长。 | 1 | 且几乎最长 | other_unparsed | S2T65 |
| 腿最短，脖子，尾巴较长。 | 1 | 脖子 | other_unparsed | S1T109 |
| 腿最长，头略短，脖子略长于尾巴，且都较短。 | 1 | 且都较短 | other_unparsed | S2T236 |
| 腿最长，头，尾巴较短。 | 1 | 头 | other_unparsed | S1T173 |
| 腿略短于其余三部位。 | 1 | 腿略短于其余三部位 | other_reference | S4T314 |
| 长度较相近且长度中等。 | 1 | 长度较相近且长度中等 | other_unparsed | S1T231 |

#### S219

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 比较均衡。 | 19 | 比较均衡 | global_balance | S1T63, S1T69, S1T75, S1T76, S1T78, S1T160, S1T189, S1T256 |
| 均衡。 | 3 | 均衡 | global_balance | S2T49, S2T104, S2T123 |
| 头和尾巴。 | 3 | 头和尾巴 | other_unparsed | S1T93, S1T124, S1T309 |
| 稍微短。 | 3 | 稍微短 | other_unparsed | S3T234, S3T235, S3T237 |
| 选错了。 | 2 | 选错了 | meta_or_uncertain | S3T97, S3T137 |
| 四个部位都中等，比较均匀。 | 1 | 比较均匀 | global_balance | S3T23 |
| 头位呀。 | 1 | 头位呀 | other_unparsed | S3T246 |
| 头和尾巴明显比脖子和腿长。 | 1 | 头和尾巴明显比脖子和腿长 | other_unparsed | S3T3 |
| 头和脖子很小。 | 1 | 头和脖子很小 | vague_size | S3T50 |
| 头和腿很短，其他很长，特别是尾巴。 | 1 | 特别是尾巴 | other_unparsed | S3T102 |
| 头是最长，明显比腿长很多。 | 1 | 明显比腿长很多 | other_unparsed | S3T8 |
| 头比较大。 | 1 | 头比较大 | other_unparsed | S1T84 |
| 头还行，腿很短。 | 1 | 头还行 | other_unparsed | S1T107 |
| 头长，脖子短，腿还行。 | 1 | 腿还行 | other_unparsed | S1T102 |
| 比较均衡，头有点儿短。 | 1 | 比较均衡 | global_balance | S1T98 |
| 比较均衡，头有点短。 | 1 | 比较均衡 | global_balance | S3T207 |
| 比较均衡，尾巴有点长。 | 1 | 比较均衡 | global_balance | S1T221 |
| 比较均衡，脖子有点儿短。 | 1 | 比较均衡 | global_balance | S1T264 |
| 比较均衡，脖子短。 | 1 | 比较均衡 | global_balance | S1T291 |
| 脖子、尾巴都。 | 1 | 脖子、尾巴都 | other_unparsed | S2T167 |
| 脖子。 | 1 | 脖子 | other_unparsed | S2T208 |
| 脖子和尾巴都。 | 1 | 脖子和尾巴都 | other_unparsed | S2T183 |
| 脖子和腿偏短，其他偏长，没有很长。 | 1 | 没有很长 | other_unparsed | S3T100 |
| 都不长，比较均衡。 | 1 | 都不长; 比较均衡 | global_balance | S3T280 |
| 都中等。 | 1 | 都中等 | other_unparsed | S3T262 |
| 都比较均衡，脖子和尾巴短。 | 1 | 都比较均衡 | global_balance | S1T195 |

#### S220

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴不一样长，且腿长。 | 3 | 头和尾巴不一样长 | disjoint_inequality | S3T262, S3T263, S3T266 |
| 脖子和尾巴长于腿，长于头。 | 3 | 长于头 | other_unparsed | S3T68, S3T69, S3T70 |
| 头、脖子和腿都比尾巴长。 | 2 | 头、脖子和腿都比尾巴长 | other_unparsed | S3T84, S3T101 |
| 头和尾巴一样长，且脖子和腿不一样长。 | 2 | 且脖子和腿不一样长 | disjoint_inequality | S3T167, S3T169 |
| 三个部位长于腿。 | 1 | 三个部位长于腿 | count_abstract | S3T33 |
| 四部位差不多长。 | 1 | 四部位差不多长 | global_balance | S3T260 |
| 头、尾巴和腿长，头和尾巴，脖子、尾巴和腿长，头短。 | 1 | 头和尾巴 | other_unparsed | S1T68 |
| 头、脖子、尾巴一样长，长于腿。 | 1 | 长于腿 | other_unparsed | S4T33 |
| 头、脖子、尾巴和腿都比尾巴短。 | 1 | 头、脖子、尾巴和腿都比尾巴短 | other_unparsed | S3T83 |
| 头、脖子、尾巴都很长，且长度差不多。 | 1 | 且长度差不多 | global_balance | S1T49 |
| 头、脖子、尾巴长于四条腿。 | 1 | 头、脖子、尾巴长于四条腿 | other_unparsed | S4T35 |
| 头、脖子、腿，和尾巴差不多长。 | 1 | 头、脖子、腿 | other_unparsed | S2T73 |
| 头、脖子和尾巴差不多长，长于腿。 | 1 | 长于腿 | other_unparsed | S3T54 |
| 头、脖子和尾巴比，头、腿和尾巴比脖子长。 | 1 | 头、脖子和尾巴比 | other_unparsed | S3T96 |
| 头、脖子和尾巴都很长，长于腿。 | 1 | 长于腿 | other_unparsed | S3T53 |
| 头、脖子和尾巴长度相等，都比腿长。 | 1 | 都比腿长 | other_unparsed | S3T13 |
| 头、脖子和腿都会比尾巴更长。 | 1 | 头、脖子和腿都会比尾巴更长 | other_unparsed | S2T247 |
| 头、脖子和腿都是尾巴长度的至少两倍。 | 1 | 头、脖子和腿都是尾巴长度的至少两倍 | other_unparsed | S1T226 |
| 头、脖子和腿长度较长，比尾巴长很多。 | 1 | 比尾巴长很多 | other_unparsed | S3T14 |
| 头、脖子长，腿中等，较短，尾巴短。 | 1 | 较短 | other_unparsed | S1T253 |
| 头、腿和尾巴都是脖子长度的至少两倍。 | 1 | 头、腿和尾巴都是脖子长度的至少两倍 | other_unparsed | S1T225 |
| 头、腿和尾巴都比脖子长，脖子最短。 | 1 | 头、腿和尾巴都比脖子长 | other_unparsed | S1T89 |
| 头和尾巴一样短，脖子和腿最长，且一样长。 | 1 | 且一样长 | other_unparsed | S1T92 |
| 头和尾巴不一样长，且脖子和腿一样长。 | 1 | 头和尾巴不一样长 | disjoint_inequality | S3T264 |
| 头和尾巴不一样长，且腿短。 | 1 | 头和尾巴不一样长 | disjoint_inequality | S3T265 |
| 头和尾巴中等偏长一点，腿中等偏短一点，脖子。 | 1 | 脖子 | other_unparsed | S2T149 |
| 头和脖子一样长，比腿和尾巴都长。 | 1 | 比腿和尾巴都长 | other_unparsed | S1T47 |
| 头和脖子一样长，腿最短，尾巴中等，较短。 | 1 | 较短 | other_unparsed | S1T71 |
| 头和脖子中等，腿比中等短一点点，尾巴较长。 | 1 | 腿比中等短一点点 | other_unparsed | S2T108 |
| 头和脖子中等，较短，尾巴短，腿较长。 | 1 | 较短 | other_unparsed | S2T36 |
| 头和脖子较短，尾巴和腿较长，中等。 | 1 | 中等 | other_unparsed | S2T132 |
| 头和脖子长，尾巴和腿。 | 1 | 尾巴和腿 | other_unparsed | S3T109 |
| 头和腿长，脖子和尾巴。 | 1 | 脖子和尾巴 | other_unparsed | S3T259 |
| 头最长，四个都不一样长。 | 1 | 四个都不一样长 | disjoint_inequality | S1T46 |
| 头短，尾巴短，脖子和腿差不多长，都是中等长度。 | 1 | 都是中等长度 | other_unparsed | S2T6 |
| 头超长，尾巴超长，脖子较短，腿中等，躯干长度的一半。 | 1 | 躯干长度的一半 | body_geometry | S1T2 |
| 头长，脖子，尾巴和腿短。 | 1 | 脖子 | other_unparsed | S1T111 |
| 头长，腿中腿差不多，腿中等长，脖子和尾巴短。 | 1 | 腿中腿差不多 | global_balance | S1T67 |
| 尾巴和脖子中等，头和腿。 | 1 | 头和腿 | other_unparsed | S2T30 |
| 尾巴和腿最长，比头和脖子长。 | 1 | 比头和脖子长 | other_unparsed | S2T294 |
| 尾巴和腿，和头较长，脖子中等。 | 1 | 尾巴和腿 | other_unparsed | S2T112 |
| 尾巴大于腿，大于头，大于脖子。 | 1 | 大于头; 大于脖子 | other_unparsed | S3T66 |
| 尾巴明显短于其他三个。 | 1 | 尾巴明显短于其他三个 | other_reference | S2T299 |
| 尾巴长于头和腿，长于脖子。 | 1 | 长于脖子 | other_unparsed | S3T76 |
| 尾巴长于头，长于脖子，长于腿。 | 1 | 长于脖子; 长于腿 | other_unparsed | S3T75 |
| 尾巴长于脖子长于头，和腿。 | 1 | 和腿 | other_unparsed | S4T34 |
| 差不多长。 | 1 | 差不多长 | global_balance | S3T237 |
| 我说脖子明显短于其他三个。 | 1 | 我说脖子明显短于其他三个 | other_reference | S2T300 |
| 有两个部位显著的短于另外两个部位。 | 1 | 有两个部位显著的短于另外两个部位 | count_abstract, other_reference | S3T81 |
| 脖子、头，脖子和尾巴和腿差不多长。 | 1 | 脖子、头 | other_unparsed | S2T287 |
| 脖子、尾巴和头，腿长稍短。 | 1 | 脖子、尾巴和头 | other_unparsed | S1T216 |
| 脖子和尾巴一样长，长于头和腿。 | 1 | 长于头和腿 | other_unparsed | S3T230 |
| 脖子和尾巴中等，尾巴比中等长一点点，头中等，腿较短。 | 1 | 尾巴比中等长一点点 | other_unparsed | S2T114 |
| 脖子和尾巴中等，腿比中等稍长一点点，头比中等稍短一点点。 | 1 | 腿比中等稍长一点点; 头比中等稍短一点点 | other_unparsed | S2T113 |
| 脖子和尾巴长度差不多，比腿和头都长。 | 1 | 比腿和头都长 | other_unparsed | S3T12 |
| 脖子和尾巴，头和腿最短。 | 1 | 脖子和尾巴 | other_unparsed | S2T70 |
| 脖子和腿一样长，比头和尾巴长。 | 1 | 比头和尾巴长 | other_unparsed | S3T295 |
| 脖子和腿中等，头和尾巴比中等偏长一点点。 | 1 | 头和尾巴比中等偏长一点点 | other_unparsed | S2T41 |
| 脖子和腿都比头和尾巴长。 | 1 | 脖子和腿都比头和尾巴长 | other_unparsed | S3T85 |
| 脖子和腿，最长头和尾巴最短。 | 1 | 脖子和腿 | other_unparsed | S1T135 |
| 脖子和腿，最长，头和尾巴最短。 | 1 | 脖子和腿; 最长 | other_unparsed | S2T135 |
| 脖子显著的长于其他三个。 | 1 | 脖子显著的长于其他三个 | other_reference | S2T298 |
| 脖子显著短于其他三个部位。 | 1 | 脖子显著短于其他三个部位 | count_abstract, other_reference | S3T44 |
| 脖子显著长于其他三个。 | 1 | 脖子显著长于其他三个 | other_reference | S2T272 |
| 脖子长于三其他三个部位。 | 1 | 脖子长于三其他三个部位 | count_abstract, other_reference | S4T100 |
| 脖子长于头，长于腿和尾巴。 | 1 | 长于腿和尾巴 | other_unparsed | S3T74 |
| 腿会比头长很多。 | 1 | 腿会比头长很多 | other_unparsed | S2T249 |
| 腿和尾巴一样长，比头和脖子长。 | 1 | 比头和脖子长 | other_unparsed | S3T294 |
| 腿和尾巴一样长，长于脖子和头。 | 1 | 长于脖子和头 | other_unparsed | S3T225 |
| 腿和尾巴长，脖子和头中。 | 1 | 脖子和头中 | other_unparsed | S1T256 |
| 腿和尾巴，头中等，脖子较短。 | 1 | 腿和尾巴 | other_unparsed | S2T182 |
| 腿显著的短于脖子、头和尾巴。 | 1 | 腿显著的短于脖子、头和尾巴 | other_unparsed | S3T82 |
| 腿最短，脖子、尾巴、头都比它长。 | 1 | 脖子、尾巴、头都比它长 | other_unparsed | S3T317 |
| 腿最长，脖子和尾巴中等，比腿稍微短一点点，头短。 | 1 | 比腿稍微短一点点 | other_unparsed | S1T227 |
| 腿最长，长于尾巴。 | 1 | 长于尾巴 | other_unparsed | S3T231 |
| 腿比较长，头、尾巴和脖子都比腿短很多。 | 1 | 头、尾巴和脖子都比腿短很多 | other_unparsed | S2T242 |

#### S221

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子和尾巴相差比较大，腿和头相差比较小。 | 1 | 脖子和尾巴相差比较大; 腿和头相差比较小 | vague_size | S1T125 |
| 脖子，最长。 | 1 | 脖子; 最长 | other_unparsed | S1T311 |
| 腿不是最长，头不是最短。 | 1 | 腿不是最长; 头不是最短 | other_unparsed | S1T158 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S2T85 |

#### S222

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头很大。 | 4 | 头很大 | other_unparsed | S2T125, S2T155, S2T162, S2T243 |
| 体型中等，尾巴和脖子比腿长。 | 3 | 体型中等 | vague_size | S2T60, S2T80, S2T241 |
| 体型中小。 | 2 | 体型中小 | vague_size | S2T317, S2T319 |
| 体型中等，四个部位差不多长。 | 2 | 体型中等 | vague_size | S2T151, S2T170 |
| 体型中等，腿长，尾巴短。 | 2 | 体型中等 | vague_size | S2T191, S2T204 |
| 体型大，四个部位都差不多。 | 2 | 体型大 | vague_size | S2T188, S2T198 |
| 体型小，头很长。 | 2 | 体型小 | vague_size | S2T254, S2T255 |
| 四个部位差不多长，体型偏大。 | 2 | 体型偏大 | vague_size | S2T161, S2T172 |
| 头和脖子比腿长，体型比较小。 | 2 | 体型比较小 | vague_size | S2T8, S2T13 |
| 尾巴长，体型小。 | 2 | 体型小 | vague_size | S2T208, S2T306 |
| 三个部位短，一个部位长。 | 1 | 三个部位短; 一个部位长 | count_abstract | S1T74 |
| 两个部位很长，一个部位比较长，一个部位比较短。 | 1 | 两个部位很长; 一个部位比较长; 一个部位比较短 | count_abstract | S1T23 |
| 两个部位很长，两个部位很短。 | 1 | 两个部位很长; 两个部位很短 | count_abstract | S1T39 |
| 两个部位最短，一个部位比较长，一个部位比较短。 | 1 | 两个部位最短; 一个部位比较长; 一个部位比较短 | count_abstract | S1T18 |
| 两个部位等长，一个，其他两个部位一长一短。 | 1 | 两个部位等长; 一个; 其他两个部位一长一短 | count_abstract, other_reference | S1T31 |
| 体型中小，头长。 | 1 | 体型中小 | vague_size | S2T244 |
| 体型中小，尾巴和腿差不多。 | 1 | 体型中小 | vague_size | S2T120 |
| 体型中等，四个部位都差不多。 | 1 | 体型中等 | vague_size | S2T192 |
| 体型中等，四个部位长度都差不多。 | 1 | 体型中等 | vague_size | S2T41 |
| 体型中等，头很长。 | 1 | 体型中等 | vague_size | S2T107 |
| 体型中等，头最长。 | 1 | 体型中等 | vague_size | S2T315 |
| 体型中等，小头比腿长。 | 1 | 体型中等 | vague_size | S2T133 |
| 体型中等，尾巴、头、腿差不多长。 | 1 | 体型中等 | vague_size | S2T123 |
| 体型中等，尾巴和脖子比头和腿长。 | 1 | 体型中等 | vague_size | S2T186 |
| 体型中等，尾巴和脖子比腿。 | 1 | 体型中等; 尾巴和脖子比腿 | vague_size | S2T119 |
| 体型中等，尾巴和腿差不多，脖子很长。 | 1 | 体型中等 | vague_size | S2T178 |
| 体型中等，尾巴和腿长。 | 1 | 体型中等 | vague_size | S2T231 |
| 体型中等，尾巴最长。 | 1 | 体型中等 | vague_size | S2T105 |
| 体型中等，尾巴长。 | 1 | 体型中等 | vague_size | S2T263 |
| 体型中等，尾巴长，腿短。 | 1 | 体型中等 | vague_size | S2T154 |
| 体型中等，腿最短，其他差不多。 | 1 | 体型中等 | vague_size | S2T229 |
| 体型中等，腿长。 | 1 | 体型中等 | vague_size | S2T101 |
| 体型中等，腿长，其他差不多。 | 1 | 体型中等 | vague_size | S2T134 |
| 体型偏中大。 | 1 | 体型偏中大 | vague_size | S2T114 |
| 体型偏中小，腿和尾巴差不多一样长。 | 1 | 体型偏中小 | vague_size | S2T56 |
| 体型偏中等，头、腿长，尾巴短。 | 1 | 体型偏中等 | vague_size | S2T62 |
| 体型偏大，头和腿比较长，尾巴比腿短。 | 1 | 体型偏大 | vague_size | S2T124 |
| 体型偏大，尾巴和腿差不多一样。 | 1 | 体型偏大 | vague_size | S2T177 |
| 体型偏大，尾巴和腿差不多一样长。 | 1 | 体型偏大 | vague_size | S2T57 |
| 体型偏小，尾巴长。 | 1 | 体型偏小 | vague_size | S2T59 |
| 体型偏小，脖子和头相对较长。 | 1 | 体型偏小 | vague_size | S2T179 |
| 体型偏小，脖子和尾巴差不多，腿最短。 | 1 | 体型偏小 | vague_size | S2T99 |
| 体型偏小，脖子比尾巴长。 | 1 | 体型偏小 | vague_size | S2T98 |
| 体型大，头很大。 | 1 | 体型大; 头很大 | vague_size | S2T196 |
| 体型大，腿长，尾巴。 | 1 | 体型大; 尾巴 | vague_size | S2T294 |
| 体型小。 | 1 | 体型小 | vague_size | S2T92 |
| 体型小，尾巴和脖子比头和腿长。 | 1 | 体型小 | vague_size | S2T185 |
| 体型很大，尾巴最长。 | 1 | 体型很大 | vague_size | S2T104 |
| 体型比较大，尾巴比较短。 | 1 | 体型比较大 | vague_size | S1T68 |
| 体型比较小。 | 1 | 体型比较小 | vague_size | S2T14 |
| 哪个部位长，哪个部位比较长，哪个部位比较短。 | 1 | 哪个部位长; 哪个部位比较长; 哪个部位比较短 | other_unparsed | S1T19 |
| 四个部位很都不等长。 | 1 | 四个部位很都不等长 | other_unparsed | S1T33 |
| 四个部位都差不多，体型中等。 | 1 | 体型中等 | vague_size | S2T234 |
| 四个部位都很小。 | 1 | 四个部位都很小 | vague_size | S1T95 |
| 头和尾巴差不多，体型偏小。 | 1 | 体型偏小 | vague_size | S2T168 |
| 头和腿，尾巴和腿差不多一样长。 | 1 | 头和腿 | other_unparsed | S2T47 |
| 头和腿，差不多长。 | 1 | 头和腿; 差不多长 | global_balance | S2T90 |
| 头最长，体型小。 | 1 | 体型小 | vague_size | S2T307 |
| 头比脖子长，腿最长，尾巴也差不多。 | 1 | 尾巴也差不多 | global_balance | S1T140 |
| 头比脖子长，较长。 | 1 | 较长 | other_unparsed | S1T146 |
| 头长，体型小。 | 1 | 体型小 | vague_size | S2T165 |
| 尾巴和头长，头和尾巴，腿和脖子比较短。 | 1 | 头和尾巴 | other_unparsed | S1T132 |
| 尾巴和脖子比头和腿。 | 1 | 尾巴和脖子比头和腿 | other_unparsed | S2T237 |
| 尾巴和脖子比腿长，体型偏小。 | 1 | 体型偏小 | vague_size | S2T108 |
| 尾巴很长，体型小。 | 1 | 体型小 | vague_size | S2T257 |
| 尾巴最长，体型偏小。 | 1 | 体型偏小 | vague_size | S2T259 |
| 尾巴最长，体型小。 | 1 | 体型小 | vague_size | S2T301 |
| 尾巴短，腿长，体型中等。 | 1 | 体型中等 | vague_size | S2T29 |
| 有三个部位长，面条左边脖子比较短。 | 1 | 有三个部位长 | count_abstract | S1T10 |
| 脖子和尾巴长，体型偏小。 | 1 | 体型偏小 | vague_size | S2T64 |
| 脖子和腿比，脖子和尾巴比腿长，头很长。 | 1 | 脖子和腿比 | other_unparsed | S2T7 |
| 脖子长，体型偏小。 | 1 | 体型偏小 | vague_size | S2T167 |
| 腿和头，中等长，脖子最短，尾巴最长。 | 1 | 腿和头; 中等长 | other_unparsed | S1T36 |
| 腿最短，其他比。 | 1 | 其他比 | other_reference | S2T82 |
| 腿短，头短，肩部为长。 | 1 | 肩部为长 | other_unparsed | S1T78 |
| 腿长，尾巴短，体型偏大。 | 1 | 体型偏大 | vague_size | S2T70 |
| 腿长，尾巴短，体型比较小。 | 1 | 体型比较小 | vague_size | S2T4 |
| 腿长，尾巴，头和脖子差不多。 | 1 | 尾巴 | other_unparsed | S1T103 |
| 腿长，脖子长，头，它尾巴比较短。 | 1 | 头 | other_unparsed | S1T86 |
| 该体型比较小，脖子和尾巴等长，头短一点。 | 1 | 该体型比较小 | vague_size | S1T67 |
| 较短。 | 1 | 较短 | other_unparsed | S1T55 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S2T173 |
| 面朝右边，四个部位差不多长。 | 1 | 面朝右边 | other_unparsed | S1T38 |
| 面朝右边，头和尾巴比差不多一样长，比腿和脖子短。 | 1 | 面朝右边; 比腿和脖子短 | other_unparsed | S1T4 |
| 面朝右边，头和腿，差等长，尾巴长，脖子短。 | 1 | 面朝右边; 头和腿; 差等长 | other_unparsed | S1T11 |
| 面朝右边，脖子比较短，其他三个部位比较长。 | 1 | 面朝右边 | other_unparsed | S1T15 |
| 面朝右边，腿和尾巴比较长，头和脖子一长一短。 | 1 | 面朝右边 | other_unparsed | S1T30 |
| 面朝右边，腿比较短，其他三个部位比较长。 | 1 | 面朝右边 | other_unparsed | S1T13 |
| 面朝左边，两个部位比较长，一个部位更长，一个部位更短。 | 1 | 面朝左边; 两个部位比较长; 一个部位更长; 一个部位更短 | count_abstract | S1T17 |
| 面朝左边，四个部位都挺长。 | 1 | 面朝左边 | other_unparsed | S1T34 |
| 面朝左边，头、腿、尾巴比脖子长。 | 1 | 面朝左边 | other_unparsed | S1T12 |
| 面朝左边，尾巴比较短，其他三个部位比较长。 | 1 | 面朝左边 | other_unparsed | S1T14 |
| 面朝左边，脖子和尾巴比其他两个部位长。 | 1 | 面朝左边 | other_unparsed | S1T1 |
| 面朝左边，脖子和腿一样长，头比较长，尾巴就比较短。 | 1 | 面朝左边 | other_unparsed | S1T28 |
| 面朝左边，脖子和腿一样长，头比较长，尾巴比较短。 | 1 | 面朝左边 | other_unparsed | S1T29 |
| 面朝左边，腿是最长，脖子比其他两个部位长。 | 1 | 面朝左边 | other_unparsed | S1T3 |
| 面朝左边，腿特别长，其他部位差不多一样长。 | 1 | 面朝左边 | other_unparsed | S1T2 |

#### S223

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头适中，脖子短，腿适中，尾巴短，头比脖子长，头也比腿短。 | 1 | 头也比腿短 | other_unparsed | S2T38 |
| 头适中，脖子短，腿长，尾巴短，脖子比腿短，头也比腿短。 | 1 | 头也比腿短 | other_unparsed | S2T28 |

#### S224

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 三个部位均较短，腿和脖子较长。 | 1 | 三个部位均较短 | count_abstract | S2T196 |
| 三个部位均较长，脖子最短。 | 1 | 三个部位均较长 | count_abstract | S2T123 |
| 三个部位长度适中，尾巴稍微短一些。 | 1 | 三个部位长度适中 | count_abstract | S3T1 |
| 只有头较短，脖子相当长。 | 1 | 脖子相当长 | global_balance | S2T13 |
| 只有尾巴较短，差不多。 | 1 | 差不多 | global_balance | S2T82 |
| 只有腿，腿最长。 | 1 | 只有腿 | other_unparsed | S3T309 |
| 四个部位都差不多长，腿最长，长度都适中。 | 1 | 长度都适中 | other_unparsed | S1T259 |
| 头、脖子和尾巴较短，腿居中。 | 1 | 腿居中 | other_unparsed | S2T314 |
| 头和脖子明显比尾巴和腿长。 | 1 | 头和脖子明显比尾巴和腿长 | other_unparsed | S3T293 |
| 头和脖子明显比腿、尾巴长。 | 1 | 头和脖子明显比腿、尾巴长 | other_unparsed | S3T275 |
| 头和脖子明显比腿、尾巴长，头最长。 | 1 | 头和脖子明显比腿、尾巴长 | other_unparsed | S3T250 |
| 头和脖子明显比腿和尾巴长。 | 1 | 头和脖子明显比腿和尾巴长 | other_unparsed | S3T138 |
| 头和脖子相对。 | 1 | 头和脖子相对 | other_unparsed | S1T112 |
| 头和脖子较短，其余居中。 | 1 | 其余居中 | other_reference | S3T125 |
| 头和脖子较短，比例较为协调。 | 1 | 比例较为协调 | proportion_or_ratio | S2T242 |
| 头和脖子非常长，比腿和尾巴长得多。 | 1 | 比腿和尾巴长得多 | other_unparsed | S2T3 |
| 头最长，尾巴，腿很短。 | 1 | 尾巴 | other_unparsed | S1T87 |
| 头比尾巴长，比脖子长。 | 1 | 比脖子长 | other_unparsed | S1T82 |
| 头比脖子长，比尾巴长。 | 1 | 比尾巴长 | other_unparsed | S1T155 |
| 脖子比其他三个部位来说较短，也适中长。 | 1 | 也适中长 | other_unparsed | S3T93 |
| 脖子长，其他都比较短，腿还可以。 | 1 | 腿还可以 | other_unparsed | S1T302 |
| 腿明显短于其他三个部位。 | 1 | 腿明显短于其他三个部位 | count_abstract, other_reference | S2T50 |
| 腿最长，其余长度协调。 | 1 | 其余长度协调 | other_reference, proportion_or_ratio | S1T307 |
| 腿极短，比头短，尾巴短。 | 1 | 比头短 | other_unparsed | S3T50 |
| 腿比脖子短，比尾巴短。 | 1 | 比尾巴短 | other_unparsed | S2T35 |
| 腿较，尾巴和脖子较短，头较短。 | 1 | 腿较 | other_unparsed | S2T95 |
| 腿，腿极短。 | 1 | 腿 | other_unparsed | S4T45 |
| 长度均适中，头和腿较短一些。 | 1 | 长度均适中 | other_unparsed | S2T112 |

#### S225

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头远比脖子长。 | 1 | 头远比脖子长 | other_unparsed | S1T240 |
| 脖子远比头长。 | 1 | 脖子远比头长 | other_unparsed | S1T239 |

#### S226

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 两长两短。 | 12 | 两长两短 | count_abstract | S1T46, S1T48, S1T49, S1T52, S1T56, S1T58, S1T61, S1T65 |
| 三长一短。 | 11 | 三长一短 | count_abstract | S1T26, S1T32, S1T45, S1T50, S1T53, S1T57, S1T66, S1T74 |
| 三个差不多长。 | 4 | 三个差不多长 | global_balance | S1T67, S1T72, S1T78, S1T151 |
| 三短一长。 | 3 | 三短一长 | other_unparsed | S1T44, S1T59, S1T262 |
| 两短两长。 | 3 | 两短两长 | other_unparsed | S1T25, S1T28, S1T258 |
| 三个都短，只有腿长。 | 2 | 三个都短 | other_unparsed | S1T113, S1T115 |
| 三长一短，尾巴短。 | 2 | 三长一短 | count_abstract | S1T27, S1T29 |
| 四个差不多，像马。 | 2 | 像马 | other_unparsed | S1T91, S1T132 |
| 四个差不多，都挺长。 | 2 | 都挺长 | other_unparsed | S1T268, S1T269 |
| 头比脖子长，比尾巴长。 | 2 | 比尾巴长 | other_unparsed | S2T27, S2T29 |
| 尾巴短，两个长。 | 2 | 两个长 | other_unparsed | S1T275, S1T277 |
| 有一个很短。 | 2 | 有一个很短 | other_unparsed | S1T54, S1T55 |
| 一个特别长。 | 1 | 一个特别长 | other_unparsed | S1T60 |
| 一个长，腿短。 | 1 | 一个长 | other_unparsed | S1T109 |
| 一长三短。 | 1 | 一长三短 | other_unparsed | S1T51 |
| 三个差不多。 | 1 | 三个差不多 | global_balance | S1T82 |
| 三个比较长，头短。 | 1 | 三个比较长 | other_unparsed | S1T181 |
| 三个比较长，腿短。 | 1 | 三个比较长 | other_unparsed | S1T182 |
| 三个短，只有头长。 | 1 | 三个短 | other_unparsed | S1T187 |
| 三个长，只有头短。 | 1 | 三个长 | other_unparsed | S1T185 |
| 三个长，尾巴短。 | 1 | 三个长 | other_unparsed | S1T86 |
| 三个长，有一个短。 | 1 | 三个长; 有一个短 | other_unparsed | S1T129 |
| 三短一长，脖子长。 | 1 | 三短一长 | other_unparsed | S1T34 |
| 三短一长，腿短。 | 1 | 三短一长 | other_unparsed | S1T33 |
| 三短一长，腿短，尾巴短。 | 1 | 三短一长 | other_unparsed | S1T75 |
| 三长一短，头短。 | 1 | 三长一短 | count_abstract | S1T35 |
| 不知道什么规律，随便选的。 | 1 | 不知道什么规律; 随便选的 | meta_or_uncertain | S1T127 |
| 两个差不多长。 | 1 | 两个差不多长 | global_balance | S1T69 |
| 两个很长。 | 1 | 两个很长 | other_unparsed | S1T85 |
| 两短一长。 | 1 | 两短一长 | other_unparsed | S1T24 |
| 两长两大。 | 1 | 两长两大 | other_unparsed | S1T264 |
| 两长两短，尾巴短。 | 1 | 两长两短 | count_abstract | S1T76 |
| 像马，四个都差不多。 | 1 | 像马 | other_unparsed | S1T90 |
| 像马，腿长。 | 1 | 像马 | other_unparsed | S1T112 |
| 四个依次变化。 | 1 | 四个依次变化 | other_unparsed | S1T42 |
| 四个差不多长，像马。 | 1 | 像马 | other_unparsed | S1T107 |
| 四个差不多长，都挺长。 | 1 | 都挺长 | other_unparsed | S1T188 |
| 四个差不多，加起来都挺长。 | 1 | 加起来都挺长 | proportion_or_ratio | S1T245 |
| 四个差不多，都长。 | 1 | 都长 | other_unparsed | S1T270 |
| 小马。 | 1 | 小马 | other_unparsed | S1T110 |
| 尾巴短，一个长。 | 1 | 一个长 | other_unparsed | S1T288 |
| 尾巴短，加起来短。 | 1 | 加起来短 | proportion_or_ratio | S1T254 |
| 尾巴长，一个短。 | 1 | 一个短 | other_unparsed | S1T291 |
| 尾巴长，两个长。 | 1 | 两个长 | other_unparsed | S1T292 |
| 尾巴长，有一个短。 | 1 | 有一个短 | other_unparsed | S1T287 |
| 差不多都挺长。 | 1 | 差不多都挺长 | other_unparsed | S1T234 |
| 差不多长。 | 1 | 差不多长 | global_balance | S1T77 |
| 差不多，头有点短。 | 1 | 差不多 | global_balance | S1T276 |
| 有点短。 | 1 | 有点短 | other_unparsed | S2T133 |
| 点错了。 | 1 | 点错了 | other_unparsed | S1T96 |
| 真的一长。 | 1 | 真的一长 | other_unparsed | S1T267 |
| 腿、尾巴短，两个长。 | 1 | 两个长 | other_unparsed | S1T281 |
| 腿很短，比较像狗。 | 1 | 比较像狗 | other_unparsed | S1T92 |
| 腿挺长，像马。 | 1 | 像马 | other_unparsed | S1T128 |
| 腿短三个长，加起来还挺长。 | 1 | 加起来还挺长 | proportion_or_ratio | S1T247 |
| 腿短，一个长。 | 1 | 一个长 | other_unparsed | S1T201 |
| 腿短，但有三个长。 | 1 | 但有三个长 | other_unparsed | S1T104 |
| 腿短，和尾巴差不多，另外两个很长。 | 1 | 和尾巴差不多 | global_balance | S1T137 |
| 腿短，脖子短，加起来一般。 | 1 | 加起来一般 | proportion_or_ratio | S1T253 |
| 腿长，三个长一个短。 | 1 | 三个长一个短 | other_unparsed | S1T138 |
| 腿长，两个短。 | 1 | 两个短 | other_unparsed | S1T200 |
| 腿长，像马。 | 1 | 像马 | other_unparsed | S1T150 |
| 腿长，其他两个短，一个长。 | 1 | 一个长 | other_unparsed | S1T206 |
| 腿长，另外三个差不多，像马。 | 1 | 像马 | other_unparsed | S1T126 |
| 腿长，尾巴长，两个短。 | 1 | 两个短 | other_unparsed | S1T285 |
| 这个差不多。 | 1 | 这个差不多 | global_balance | S1T81 |
| 这个差不多长，像马。 | 1 | 这个差不多长; 像马 | global_balance | S1T106 |
| 都挺长，但头比较短。 | 1 | 都挺长 | other_unparsed | S1T246 |
| 都挺长，但头短。 | 1 | 都挺长 | other_unparsed | S1T249 |
| 都挺长，加起来挺长。 | 1 | 都挺长; 加起来挺长 | proportion_or_ratio | S1T248 |
| 都比身子短。 | 1 | 都比身体短 | body_geometry | S2T201 |
| 都短，只有一个脖子长。 | 1 | 都短 | other_unparsed | S1T174 |
| 都跟躯干差不多长。 | 1 | 都跟躯干差不多长 | body_geometry, global_balance | S2T210 |

#### S227

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度各不相同。 | 7 | 四个部位长度各不相同 | disjoint_inequality | S1T97, S1T98, S1T100, S1T101, S1T103, S1T132, S1T133 |
| 四个部位长度不一。 | 2 | 四个部位长度不一 | other_unparsed | S1T50, S1T148 |
| 四个部位长度不同。 | 2 | 四个部位长度不同 | disjoint_inequality | S1T114, S1T115 |
| 尾巴和腿长度近似。 | 2 | 尾巴和腿长度近似 | other_unparsed | S1T96, S1T171 |
| 脖子和腿长度近似。 | 2 | 脖子和腿长度近似 | other_unparsed | S1T92, S1T173 |
| 头和脖子长度近似。 | 1 | 头和脖子长度近似 | other_unparsed | S1T95 |
| 尾巴长，头，腿和脖子比较短。 | 1 | 头 | other_unparsed | S1T22 |
| 有两个部位的长度超过了躯干。 | 1 | 有两个部位的长度超过了躯干 | count_abstract, body_geometry | S1T107 |
| 脖子短，其余三个部位较长，且差不多长。 | 1 | 且差不多长 | global_balance | S1T53 |
| 脖子长度和躯干近似。 | 1 | 脖子长度和躯干近似 | body_geometry | S1T93 |
| 腿长，四个部位长度各不相同。 | 1 | 四个部位长度各不相同 | disjoint_inequality | S1T94 |

#### S228

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 比较均匀。 | 10 | 比较均匀 | global_balance | S1T146, S2T79, S2T82, S2T88, S2T96, S2T98, S2T102, S2T125 |
| 均匀。 | 5 | 均匀 | global_balance | S2T104, S2T107, S2T110, S2T112, S2T116 |
| 三个都在躯干上面。 | 1 | 三个都在躯干上面 | body_geometry | S1T270 |
| 三长一短。 | 1 | 三长一短 | count_abstract | S1T77 |
| 整体都比较短。 | 1 | 整体都比较短 | other_unparsed | S1T179 |
| 整体都比较长。 | 1 | 整体都比较长 | other_unparsed | S1T172 |
| 脖子不是最短。 | 1 | 脖子不是最短 | other_unparsed | S1T87 |

#### S231

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 各个部分差不多长。 | 2 | 各个部分差不多长 | global_balance | S1T302, S3T125 |
| 肘比腿长，脖子比尾巴长。 | 2 | 肘比腿长 | other_unparsed | S3T209, S3T302 |
| 各个部分差不多长，脖子、尾巴一样长。 | 1 | 各个部分差不多长 | global_balance | S1T286 |
| 各个部分看上去差不多，腿和尾巴略短。 | 1 | 各个部分看上去差不多 | global_balance | S1T160 |
| 各个部分都是中等长度。 | 1 | 各个部分都是中等长度 | other_unparsed | S2T125 |
| 头和脖子短，腿和尾巴长，两长，两短。 | 1 | 两长; 两短 | other_unparsed | S1T40 |
| 头最短，其他部位也短，比头长一些。 | 1 | 比头长一些 | other_unparsed | S2T100 |
| 头最短，脖子和尾巴差不多长，都长。 | 1 | 都长 | other_unparsed | S1T73 |
| 头最长，其他三个部分差不多，比较短。 | 1 | 比较短 | other_unparsed | S1T64 |
| 尾巴和脖子。 | 1 | 尾巴和脖子 | other_unparsed | S1T140 |
| 尾巴最长，其他部位稍短，并且长度差不多。 | 1 | 并且长度差不多 | global_balance | S1T30 |
| 尾巴最长，然后是脖子，其他两个部位差不多。 | 1 | 然后是脖子 | ordinal_or_secondary | S1T22 |
| 尾巴比较短，腿长，头和脖子也比较长，相差不大。 | 1 | 相差不大 | other_unparsed | S2T213 |
| 差不多长，腿略短一点。 | 1 | 差不多长 | global_balance | S2T143 |
| 手比腿长，脖子比尾巴长。 | 1 | 手比腿长 | other_unparsed | S3T314 |
| 脖子最短，腿和尾巴最长，并且差不多。 | 1 | 并且差不多 | global_balance | S1T31 |
| 脖子最长，腿可能和腿差不多。 | 1 | 腿可能和腿差不多 | global_balance | S1T60 |
| 脖子较短，各个部分都比较长。 | 1 | 各个部分都比较长 | other_unparsed | S1T214 |
| 腿最长，头和脖子都短，并且差不多。 | 1 | 并且差不多 | global_balance | S1T35 |
| 都差不多，头略短一些。 | 1 | 都差不多 | global_balance | S1T156 |
| 都比较短。 | 1 | 都比较短 | other_unparsed | S4T66 |

#### S301

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子和头都很长，长度相近。 | 3 | 长度相近 | other_unparsed | S1T309, S2T10, S2T11 |
| 头和脖子很长，腿也很长，三者长度相近。 | 1 | 三者长度相近 | other_unparsed | S1T110 |
| 头和脖子极长，比尾巴和腿都长。 | 1 | 比尾巴和腿都长 | other_unparsed | S1T188 |
| 头和脖子相对较长，腿和尾巴中等，稍短一些。 | 1 | 稍短一些 | other_unparsed | S1T160 |
| 头和脖子较长，均比腿长，尾巴较短。 | 1 | 均比腿长 | other_unparsed | S1T97 |
| 头和脖子非常长，均比腿长。 | 1 | 均比腿长 | other_unparsed | S1T78 |
| 头和脖子非常长，比尾巴长，腿较短。 | 1 | 比尾巴长 | other_unparsed | S1T217 |
| 头很长、比脖子长一些，尾巴和腿相对中等。 | 1 | 头很长、比脖子长一些 | other_unparsed | S1T133 |
| 头很长，比脖子长很多，腿也比较长。 | 1 | 比脖子长很多 | other_unparsed | S1T169 |
| 头很长，比脖子长，脖子比腿要短。 | 1 | 比脖子长 | other_unparsed | S1T252 |
| 头极长，比尾巴长。 | 1 | 比尾巴长 | other_unparsed | S1T192 |
| 头长，尾巴，脖子短，尾巴长。 | 1 | 尾巴 | other_unparsed | S1T317 |
| 头长，脖子短，相对来说，头和脖子都很长。 | 1 | 相对来说 | other_unparsed | S1T304 |
| 头长，脖子短，相对较小。 | 1 | 相对较小 | other_unparsed | S1T293 |
| 头非常长，比脖子长很多。 | 1 | 比脖子长很多 | other_unparsed | S1T241 |
| 尾巴很长，头很长，两者长度相近，腿很短。 | 1 | 两者长度相近 | other_unparsed | S1T138 |
| 相对来说，脖子比较长，腿非常长。 | 1 | 相对来说 | other_unparsed | S1T229 |
| 脖子和头都较短，长度相近，尾巴较长。 | 1 | 长度相近 | other_unparsed | S2T20 |
| 脖子和头长度相近，都偏短。 | 1 | 都偏短 | other_unparsed | S2T13 |
| 脖子很长、长于头，腿较长。 | 1 | 脖子很长、长于头 | other_unparsed | S1T112 |
| 脖子很长，比头和尾巴都长。 | 1 | 比头和尾巴都长 | other_unparsed | S1T281 |
| 脖子很长，相对来说，比头和尾巴长。 | 1 | 相对来说; 比头和尾巴长 | other_unparsed | S1T213 |
| 脖子比腿长，比头长。 | 1 | 比头长 | other_unparsed | S1T245 |
| 脖子非常长、长于头，腿较长，尾巴较短。 | 1 | 脖子非常长、长于头 | other_unparsed | S1T105 |
| 脖子非常长、长于尾巴，头极短，腿极短。 | 1 | 脖子非常长、长于尾巴 | other_unparsed | S1T100 |
| 脖子非常长，其余三个部位较长，长度相近。 | 1 | 长度相近 | other_unparsed | S1T114 |
| 腿很长，其余三个部位较长，长度相近。 | 1 | 长度相近 | other_unparsed | S1T94 |
| 腿很长，头和脖子相近，也都比较长。 | 1 | 也都比较长 | other_unparsed | S1T72 |
| 腿极短，头、脖子、尾巴都较长，长度相近。 | 1 | 长度相近 | other_unparsed | S1T42 |
| 腿极长，头、脖子和尾巴相近，稍短一些。 | 1 | 稍短一些 | other_unparsed | S1T36 |
| 腿比较长，脖子比较长，比头长。 | 1 | 比头长 | other_unparsed | S1T242 |
| 腿较长，脖子很长，比头长。 | 1 | 比头长 | other_unparsed | S1T139 |
| 腿较长，脖子相对较长、比头长。 | 1 | 脖子相对较长、比头长 | other_unparsed | S1T102 |
| 长度比较均衡，相对来说脖子长一些。 | 1 | 长度比较均衡 | global_balance | S1T270 |

#### S302

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿长，其他部位都不是很长，比较匀称。 | 2 | 比较匀称 | global_balance | S1T109, S1T136 |
| 上半身很长，腿很短。 | 1 | 上半身很长 | other_unparsed | S1T85 |
| 上半身比较长，腿相对短。 | 1 | 上半身比较长 | other_unparsed | S1T89 |
| 头比较短，其他部位比较匀称。 | 1 | 其他部位比较匀称 | other_reference, global_balance | S1T103 |
| 头相对长，其他部位不是很长，比较匀称。 | 1 | 比较匀称 | global_balance | S1T106 |
| 尾巴长，脖子、头、腿都不是很长，整体比较匀称。 | 1 | 整体比较匀称 | global_balance | S1T93 |
| 脖子和头都比较长，整体，腿和尾巴也相对长。 | 1 | 整体 | other_unparsed | S1T129 |
| 脖子比，头比脖子长。 | 1 | 脖子比 | other_unparsed | S1T156 |
| 脖子长，尾巴长，头短，腿中等，整体比较匀称。 | 1 | 整体比较匀称 | global_balance | S1T81 |
| 脖子，脖子长，尾巴短。 | 1 | 脖子 | other_unparsed | S1T42 |
| 腿很长，上半身都比较短。 | 1 | 上半身都比较短 | other_unparsed | S1T90 |
| 都比较中等，都不是很长。 | 1 | 都比较中等; 都不是很长 | other_unparsed | S1T146 |

#### S303

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头，脖子、腿都短，尾巴也短、比其他部位长一点。 | 1 | 头 | other_unparsed | S1T90 |

#### S304

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和腿长度相同。 | 2 | 头和腿长度相同 | other_unparsed | S1T23, S1T25 |
| 头和躯干长度相同，其他部位长度是躯干的0.7倍。 | 1 | 头和躯干长度相同; 其他部位长度是躯干的0.7倍 | other_reference, body_geometry | S1T6 |

#### S305

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 挺高大。 | 8 | 挺高大 | body_geometry, vague_size | S1T50, S1T51, S1T52, S1T54, S1T55, S1T56, S1T59, S1T61 |
| 身材高大。 | 7 | 身材高大 | body_geometry, vague_size | S1T70, S1T80, S1T82, S1T89, S1T92, S1T94, S1T96 |
| 很高大。 | 6 | 很高大 | body_geometry, vague_size | S1T1, S1T12, S1T23, S1T24, S1T31, S1T46 |
| 挺高大，脖子短。 | 5 | 挺高大 | body_geometry, vague_size | S1T37, S1T40, S1T42, S1T45, S1T47 |
| 中等身材，头很短。 | 1 | 中等身材 | body_geometry, vague_size | S1T4 |
| 很高。 | 1 | 很高 | other_unparsed | S1T3 |
| 比较适中，头和尾巴长，脖子短。 | 1 | 比较适中 | other_unparsed | S1T32 |
| 比较高大。 | 1 | 比较高大 | body_geometry, vague_size | S1T34 |
| 脖子高，脖子短。 | 1 | 脖子高 | other_unparsed | S1T35 |
| 腿适中，其实还挺短。 | 1 | 其实还挺短 | other_unparsed | S1T29 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S1T163 |

#### S306

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子比头长，和腿差不多，比腿稍长，尾巴最短。 | 1 | 和腿差不多; 比腿稍长 | other_unparsed | S1T44 |
| 腿和头差不多一样长，长度较长，尾巴和脖子相对来说较短。 | 1 | 长度较长 | other_unparsed | S1T42 |
| 腿和尾巴一样长，较长，脖子非常长，头较短。 | 1 | 较长 | other_unparsed | S1T11 |
| 腿很长，脖子，头较短，尾巴适中。 | 1 | 脖子 | other_unparsed | S2T83 |
| 腿明显很短，头比脖子长，比尾巴长很多。 | 1 | 比尾巴长很多 | other_unparsed | S1T214 |
| 腿明显比较长，头比脖子长，两者都很长，尾巴短。 | 1 | 两者都很长 | other_unparsed | S1T218 |
| 腿短，尾巴很长，脖子和头，较短。 | 1 | 脖子和头; 较短 | other_unparsed | S1T147 |
| 腿短，尾巴长，头和脖子，也都很长。 | 1 | 头和脖子; 也都很长 | other_unparsed | S2T66 |
| 腿短，脖子和头接近，长度较长，尾巴更长一些。 | 1 | 长度较长 | other_unparsed | S2T17 |
| 腿短，脖子和尾巴一样长，比头短。 | 1 | 比头短 | other_unparsed | S1T155 |
| 腿短，脖子比头稍长，比尾巴长很多。 | 1 | 比尾巴长很多 | other_unparsed | S1T88 |
| 腿短，脖子比尾巴稍长，也比头长一些。 | 1 | 也比头长一些 | other_unparsed | S1T73 |
| 腿短，脖子，脖子稍长。 | 1 | 脖子 | other_unparsed | S2T259 |
| 腿较短，其他三者较长，且长度接近。 | 1 | 且长度接近 | other_unparsed | S2T36 |
| 腿较短，头较长、略长于尾巴，脖子较短。 | 1 | 头较长、略长于尾巴 | other_unparsed | S2T43 |
| 腿较短，尾巴短，头，和脖子较长，脖子比头长一些。 | 1 | 头 | other_unparsed | S2T22 |
| 腿较短，脖子、头较长，尾巴略比前两者略短一些。 | 1 | 尾巴略比前两者略短一些 | other_unparsed | S1T247 |
| 腿较短，脖子比头子，脖子比头长，尾巴和脖子差不多一样长。 | 1 | 脖子比头子 | other_unparsed | S1T220 |
| 腿较短，脖子较短，头和尾巴长度接近，较长。 | 1 | 较长 | other_unparsed | S2T37 |
| 腿较短，脖子较长，头和尾巴差不多一样长，比脖子短。 | 1 | 比脖子短 | other_unparsed | S1T174 |
| 腿较长，头较长、略长于脖子，脖子长度略短于尾巴。 | 1 | 头较长、略长于脖子; 脖子长度略短于尾巴 | other_unparsed | S2T42 |
| 腿较长，尾巴较短，头和脖子长度接近，较长。 | 1 | 较长 | other_unparsed | S2T18 |
| 腿较长，尾巴，尾巴长，脖子比头短。 | 1 | 尾巴 | other_unparsed | S1T222 |
| 腿长，头，较长。 | 1 | 头; 较长 | other_unparsed | S2T95 |
| 腿长，脖子和尾巴一样长，比头长。 | 1 | 比头长 | other_unparsed | S1T72 |
| 腿长，脖子和尾巴差不多长，比头长一些。 | 1 | 比头长一些 | other_unparsed | S2T14 |
| 较躯干来说，腿短，尾巴长。 | 1 | 较躯干来说 | body_geometry | S1T65 |
| 较躯干来说，腿短，尾巴长，头比脖子稍长。 | 1 | 较躯干来说 | body_geometry | S1T67 |
| 较躯干来说，腿较短，其他部位均较长，脖子比头长。 | 1 | 较躯干来说 | body_geometry | S1T57 |
| 较躯干来说，腿较长，尾巴短，脖子和头都较长。 | 1 | 较躯干来说 | body_geometry | S1T63 |
| 较躯干来说，腿较长，脖子比头长，是头比脖子长，尾巴短。 | 1 | 较躯干来说 | body_geometry | S1T64 |
| 较躯干来说，腿适中，其他部位较长一些。 | 1 | 较躯干来说 | body_geometry | S1T58 |
| 较躯干而言，腿的适中，脖子较短，头较长，尾巴短。 | 1 | 较躯干而言 | body_geometry | S1T60 |
| 较躯干而言，腿短，头比脖子稍长一些，四个部位都短，尾巴最长。 | 1 | 较躯干而言 | body_geometry | S1T62 |
| 较躯干而言，腿适中，脖子比头长。 | 1 | 较躯干而言 | body_geometry | S1T61 |
| 较躯干而言，腿适中，脖子较短，其他部位适中。 | 1 | 较躯干而言 | body_geometry | S1T59 |
| 较躯干而言，腿长，尾巴短。 | 1 | 较躯干而言 | body_geometry | S1T66 |

#### S307

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和脖子差不多长，它们都是中等，腿较长，尾巴较长。 | 1 | 它们都是中等 | other_unparsed | S1T86 |
| 头和脖子差不多长，都是较长，尾巴中等，腿较长。 | 1 | 都是较长 | other_unparsed | S1T66 |
| 头和脖子，腿较短，尾巴很短。 | 1 | 头和脖子 | other_unparsed | S1T159 |
| 头较长，脖，脖子，中等，尾巴中等，腿较短。 | 1 | 脖子; 中等 | other_unparsed | S1T174 |
| 脖子是头的两倍。 | 1 | 脖子是头的两倍 | other_unparsed | S1T141 |
| 脖子比头长一点，两个都是较长，尾巴较长，腿很短。 | 1 | 两个都是较长 | other_unparsed | S2T4 |
| 腿、脖子和尾巴，四个部位都短，头较长。 | 1 | 腿、脖子和尾巴 | other_unparsed | S1T14 |
| 腿和尾巴很长，脖子很长，头很小。 | 1 | 头很小 | vague_size | S2T62 |
| 腿很长，脖子，脖子、尾巴较长，头很短。 | 1 | 脖子 | other_unparsed | S2T193 |

#### S308

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子不是最长。 | 12 | 脖子不是最长 | other_unparsed | S2T235, S2T242, S2T243, S2T244, S2T257, S2T258, S2T273, S2T280 |
| 哪个最长，头和脖子长度相似，腿最短。 | 1 | 哪个最长 | other_unparsed | S1T160 |
| 头与尾巴长度相似，比脖子长。 | 1 | 比脖子长 | other_unparsed | S1T92 |
| 头最短，腿与尾巴等长，且更长。 | 1 | 且更长 | other_unparsed | S1T112 |
| 尾巴最长，其，头、脖子、腿长度相似。 | 1 | 其 | other_unparsed | S1T89 |
| 尾巴比腿，腿长。 | 1 | 尾巴比腿 | other_unparsed | S1T296 |
| 脖子不是最长，尾巴长。 | 1 | 脖子不是最长 | other_unparsed | S2T241 |
| 脖子不是最长，腿长。 | 1 | 脖子不是最长 | other_unparsed | S2T239 |
| 脖子与尾巴长度相似，且最长。 | 1 | 且最长 | other_unparsed | S1T63 |
| 脖子与腿长度相似，且长度长于头。 | 1 | 且长度长于头 | other_unparsed | S1T55 |
| 脖子比头长，脖子与尾巴长度无法分辨。 | 1 | 脖子与尾巴长度无法分辨 | other_unparsed | S2T14 |
| 脖子比头长，腿与尾巴长度相似，且比且比脖子更长。 | 1 | 且比且比脖子更长 | other_unparsed | S1T210 |
| 脖子比头长，腿和尾巴长度相似，且比脖子更长。 | 1 | 且比脖子更长 | other_unparsed | S1T194 |
| 脖子比尾巴长，比腿长。 | 1 | 比腿长 | other_unparsed | S2T35 |
| 腿与尾巴，最短。 | 1 | 腿与尾巴; 最短 | other_unparsed | S1T53 |
| 腿最长，尾巴最短，脖子比头长，短于腿。 | 1 | 短于腿 | other_unparsed | S1T135 |

#### S309

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 整体都较为均匀。 | 2 | 整体都较为均匀 | global_balance | S1T26, S1T27 |
| 比较均衡。 | 2 | 比较均衡 | global_balance | S1T258, S1T264 |
| 基本都很均衡。 | 1 | 基本都很均衡 | global_balance | S1T317 |
| 尾巴短，小小。 | 1 | 小小 | other_unparsed | S1T259 |
| 尾巴非常短，其他部位正常。 | 1 | 其他部位正常 | other_reference | S1T23 |
| 腿、头、尾巴都比脖子长。 | 1 | 腿、头、尾巴都比脖子长 | other_unparsed | S1T53 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S3T9 |
| 非常均衡。 | 1 | 非常均衡 | global_balance | S1T295 |

#### S310

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短，躯干长。 | 2 | 躯干长 | body_geometry | S3T300, S3T319 |
| 头、脖子和腿长度差不多，尾巴要稍短一些，和躯干的长度相当，全身比较均衡。 | 1 | 全身比较均衡 | global_balance | S2T37 |
| 头和尾巴长，脖子和腿，短。 | 1 | 脖子和腿; 短 | other_unparsed | S2T227 |
| 头和尾巴，相当躯干，差不多，脖子最长，腿最短。 | 1 | 头和尾巴; 相当躯干; 差不多 | body_geometry, global_balance | S3T69 |
| 头和腿长，脖子和腿，和尾巴短。 | 1 | 脖子和腿 | other_unparsed | S1T300 |
| 头很长，脖子和腿，长度差不多。 | 1 | 脖子和腿; 长度差不多 | global_balance | S1T160 |
| 头最长、比躯干长，脖子、腿、尾巴长度相当，且与躯干长度差不多。 | 1 | 且与躯干长度差不多 | body_geometry, global_balance | S3T49 |
| 头最长，也比较短，脖子、腿、尾巴都很短。 | 1 | 也比较短 | other_unparsed | S2T109 |
| 头最长，尾巴稍微比脖子和腿短一点。 | 1 | 尾巴稍微比脖子和腿短一点 | other_unparsed | S2T67 |
| 头最长，脖子、腿，差不多长，尾巴很短。 | 1 | 脖子、腿; 差不多长 | global_balance | S2T82 |
| 头最长，腿最短，相差较大。 | 1 | 相差较大 | other_unparsed | S1T41 |
| 头比躯干短，其余各部位长度比头长，且长度相当。 | 1 | 其余各部位长度比头长; 且长度相当 | other_reference, global_balance | S4T17 |
| 头比躯干短，尾巴短，躯干最长，且明显长于其他部位。 | 1 | 躯干最长 | body_geometry | S4T16 |
| 头比躯干短，腿比其他部位都长，且最长。 | 1 | 且最长 | other_unparsed | S4T13 |
| 头比躯干短，躯干较长，其余各部分较短。 | 1 | 躯干较长 | body_geometry | S4T18 |
| 头短，其余各肢平衡。 | 1 | 其余各肢平衡 | other_reference | S3T285 |
| 头短，尾巴、躯干较长，腿，脖子较短。 | 1 | 腿 | other_unparsed | S3T262 |
| 头短，脖子长，尾巴长，躯干长，腿最短。 | 1 | 躯干长 | body_geometry | S4T36 |
| 头短，躯干最长，其余各部位都较短且差不多长。 | 1 | 躯干最长 | body_geometry | S4T40 |
| 头稍短，脖子、躯干、尾巴、腿长度差不多，比头长。 | 1 | 比头长 | other_unparsed | S3T42 |
| 头稍长，躯干最长，脖子、尾巴、腿较短。 | 1 | 躯干最长 | body_geometry | S3T92 |
| 头，尾巴稍短，脖子最长，腿稍长。 | 1 | 头 | other_unparsed | S3T28 |
| 头，脖子和腿较长，尾巴稍短。 | 1 | 头 | other_unparsed | S2T103 |
| 头，腿、尾巴很长，脖子很短。 | 1 | 头 | other_unparsed | S3T77 |
| 尾巴和腿，长度比躯干更长，头和脖子很短。 | 1 | 尾巴和腿; 长度比躯干更长 | body_geometry | S3T75 |
| 尾巴长，腿第二，头第三，脖子，次。 | 1 | 头第三; 脖子; 次 | ordinal_or_secondary | S2T76 |
| 差不多长。 | 1 | 差不多长 | global_balance | S2T117 |
| 脖子很长，其他，差不多。 | 1 | 其他; 差不多 | other_reference, global_balance | S2T155 |
| 脖子最长，头，尾巴和腿稍短。 | 1 | 头 | other_unparsed | S2T224 |
| 脖子最，头最长，脖子最短，腿和尾巴中间。 | 1 | 脖子最 | other_unparsed | S2T226 |
| 脖子长，头和尾巴，中间腿最短。 | 1 | 头和尾巴 | other_unparsed | S1T235 |
| 腿很长，尾巴稍短，头和脖子要比尾巴更短一些。 | 1 | 头和脖子要比尾巴更短一些 | other_unparsed | S2T27 |
| 腿最短，躯干最长，头、脖子、尾巴长度中间且差不多。 | 1 | 躯干最长 | body_geometry | S3T71 |
| 腿最长，其他部位都相对较短，且长度差不多。 | 1 | 且长度差不多 | global_balance | S1T251 |
| 腿最长，头、尾巴、脖子较短，且长度差不多。 | 1 | 且长度差不多 | global_balance | S1T236 |
| 腿最长，头和躯干差不多，脖子最短，尾巴稍短。 | 1 | 头和躯干差不多 | body_geometry, global_balance | S3T78 |

#### S311

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和脖子最长，腿和尾巴。 | 2 | 腿和尾巴 | other_unparsed | S1T172, S1T174 |
| 头最长，尾巴、脖子和腿。 | 2 | 尾巴、脖子和腿 | other_unparsed | S1T92, S1T179 |
| 尾巴最长，腿、头和脖子。 | 2 | 腿、头和脖子 | other_unparsed | S1T110, S1T250 |
| 脖子最长，尾巴、头和腿。 | 2 | 尾巴、头和腿 | other_unparsed | S1T99, S1T294 |
| 脖子最长，腿、头和尾巴。 | 2 | 腿、头和尾巴 | other_unparsed | S1T93, S1T216 |
| 腿最长，头、尾巴和脖子。 | 2 | 头、尾巴和脖子 | other_unparsed | S1T61, S1T106 |
| 腿最长，尾巴、头和脖子。 | 2 | 尾巴、头和脖子 | other_unparsed | S1T96, S1T202 |
| 头、脖子、尾巴、腿略长，且都比较接近。 | 1 | 且都比较接近 | other_unparsed | S1T1 |
| 头和尾巴很长，脖子，腿最短。 | 1 | 脖子 | other_unparsed | S1T143 |
| 头和尾巴最长，脖子和腿。 | 1 | 脖子和腿 | other_unparsed | S1T242 |
| 头和尾巴略长，腿和脖子。 | 1 | 腿和脖子 | other_unparsed | S1T142 |
| 头和脖子最长，尾巴和腿。 | 1 | 尾巴和腿 | other_unparsed | S1T136 |
| 头和腿最长，尾巴和脖子。 | 1 | 尾巴和脖子 | other_unparsed | S1T286 |
| 头最长，尾巴、脖子和腿，很短。 | 1 | 尾巴、脖子和腿; 很短 | other_unparsed | S1T137 |
| 头最长，尾巴和腿。 | 1 | 尾巴和腿 | other_unparsed | S1T192 |
| 头最长，尾巴，脖子和腿略短。 | 1 | 尾巴 | other_unparsed | S1T25 |
| 头最长，脖子、腿和尾巴。 | 1 | 脖子、腿和尾巴 | other_unparsed | S1T251 |
| 头最长，腿、尾巴和脖子。 | 1 | 腿、尾巴和脖子 | other_unparsed | S1T98 |
| 头最长，腿，脖子和尾巴很短。 | 1 | 腿 | other_unparsed | S1T102 |
| 尾巴和头。 | 1 | 尾巴和头 | other_unparsed | S2T8 |
| 尾巴和脖子最长，腿和头。 | 1 | 腿和头 | other_unparsed | S1T163 |
| 尾巴和脖子比较长，头，腿最短。 | 1 | 头 | other_unparsed | S1T161 |
| 尾巴和腿也比较长，脖子，头最短。 | 1 | 脖子 | other_unparsed | S1T123 |
| 尾巴最长，脖子、头和腿，很短。 | 1 | 脖子、头和腿; 很短 | other_unparsed | S1T78 |
| 尾巴最长，脖子和腿，还有头。 | 1 | 脖子和腿; 还有头 | other_unparsed | S1T317 |
| 脖子、腿、尾巴比较长，且比较接近，头最短。 | 1 | 且比较接近 | other_unparsed | S1T6 |
| 脖子最长，其次是尾巴、头，和腿。 | 1 | 和腿 | other_unparsed | S1T115 |
| 脖子最长，头、腿和尾巴。 | 1 | 头、腿和尾巴 | other_unparsed | S1T135 |
| 脖子最长，头，腿和尾巴。 | 1 | 头; 腿和尾巴 | other_unparsed | S1T149 |
| 脖子最长，尾巴和头，还有腿。 | 1 | 尾巴和头; 还有腿 | other_unparsed | S1T240 |
| 脖子略长，尾巴，再是腿和头。 | 1 | 尾巴; 再是腿和头 | ordinal_or_secondary | S1T270 |
| 腿和尾巴比较长，头，脖子最短。 | 1 | 头 | other_unparsed | S1T160 |
| 腿最长，其次是头和脖子，腿。 | 1 | 腿 | other_unparsed | S1T222 |
| 腿最长，头、脖子和尾巴。 | 1 | 头、脖子和尾巴 | other_unparsed | S1T121 |
| 腿最长，头和尾巴，还有脖子。 | 1 | 头和尾巴; 还有脖子 | other_unparsed | S1T156 |
| 腿最长，尾巴、脖子和头。 | 1 | 尾巴、脖子和头 | other_unparsed | S1T148 |
| 腿最长，脖子、头和尾巴。 | 1 | 脖子、头和尾巴 | other_unparsed | S1T112 |

#### S312

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 一样长。 | 1 | 一样长 | other_unparsed | S1T197 |
| 中等长度，头和尾巴略短。 | 1 | 中等长度 | other_unparsed | S3T137 |
| 中等长度，腿略长，脖子很短。 | 1 | 中等长度 | other_unparsed | S3T138 |
| 头、腿、尾巴，很短。 | 1 | 头、腿、尾巴; 很短 | other_unparsed | S3T54 |
| 差不多长。 | 1 | 差不多长 | global_balance | S2T254 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S2T140 |
| 都略短，头比尾巴长。 | 1 | 都略短 | other_unparsed | S3T18 |
| 都较短，头略长。 | 1 | 都较短 | other_unparsed | S3T253 |
| 都较短，尾巴和腿更短一点。 | 1 | 都较短 | other_unparsed | S3T315 |

#### S313

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、尾巴、腿均小于或等于，躯干。 | 1 | 头、脖子、尾巴、腿均小于或等于; 躯干 | body_geometry | S3T53 |
| 头和腿，脖子和尾巴比躯干短。 | 1 | 头和腿 | other_unparsed | S3T37 |
| 头很长，尾巴、腿、脖子差不多长，都比头稍微短一点。 | 1 | 都比头稍微短一点 | other_unparsed | S2T70 |
| 尾巴短，腿短，脖子比头短，都比尾巴和腿长。 | 1 | 都比尾巴和腿长 | other_unparsed | S2T281 |
| 尾巴，巴长，腿短，头短。 | 1 | 尾巴; 巴长 | other_unparsed | S1T165 |
| 腿短，尾巴长，头较短，和脖子相比。 | 1 | 和脖子相比 | other_unparsed | S2T176 |
| 腿长，尾巴长，头和脖子差不多长，头和脖子都比尾巴和腿短。 | 1 | 头和脖子都比尾巴和腿短 | other_unparsed | S2T79 |

#### S314

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 躯干长于脖子，且短于腿。 | 6 | 且短于腿 | other_unparsed | S3T137, S3T139, S3T141, S3T142, S3T145, S3T146 |
| 朝向向左。 | 4 | 朝向向左 | other_unparsed | S1T37, S1T39, S1T40, S1T42 |
| 有一个部位比躯干长。 | 3 | 有一个部位比躯干长 | count_abstract, body_geometry | S1T43, S1T44, S1T45 |
| 躯干短于脖子，且长于尾巴。 | 2 | 且长于尾巴 | other_unparsed | S3T136, S3T140 |
| 去判断于脖子和尾巴。 | 1 | 去判断于脖子和尾巴 | other_unparsed | S3T190 |
| 去干短于脖子，且去干短于尾巴。 | 1 | 去干短于脖子; 且去干短于尾巴 | other_unparsed | S3T84 |
| 去干长于脖子，且去干长于腿。 | 1 | 去干长于脖子; 且去干长于腿 | other_unparsed | S2T290 |
| 朝向向右。 | 1 | 朝向向右 | other_unparsed | S1T41 |
| 脖子长于头、长于躯干，长于尾巴、长于腿。 | 1 | 长于尾巴、长于腿 | other_unparsed | S1T67 |
| 腿长于头、长于躯干，长于脖子。 | 1 | 长于脖子 | other_unparsed | S1T68 |
| 躯干短于脖子，且短于尾巴。 | 1 | 且短于尾巴 | other_unparsed | S3T138 |
| 躯干短于，脖子和尾巴。 | 1 | 躯干短于; 脖子和尾巴 | body_geometry | S3T179 |

#### S315

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖。 | 1 | 脖子 | other_unparsed | S1T95 |

#### S316

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短，且是最短，头比脖子长。 | 4 | 且是最短 | other_unparsed | S1T249, S1T250, S1T253, S1T269 |
| 腿短，且是最短，头比脖子短。 | 3 | 且是最短 | other_unparsed | S1T254, S1T265, S1T270 |
| 腿短，且脖子不是头、脖子、尾巴里最短。 | 2 | 且脖子不是头、脖子、尾巴里最短 | other_unparsed | S1T137, S1T139 |
| 腿短，尾巴不比脖子长。 | 2 | 尾巴不比脖子长 | other_unparsed | S1T178, S1T179 |
| 腿长，尾巴明显比腿短。 | 2 | 尾巴明显比腿短 | other_unparsed | S1T62, S1T63 |
| 尾巴和脖子很长，腿很长，这题。 | 1 | 这题 | other_unparsed | S1T47 |
| 脖子不是最短，脖子比尾巴长，尾巴是头、脖子、尾巴里最短。 | 1 | 脖子不是最短 | other_unparsed | S1T153 |
| 脖子和尾巴不比头长，腿短，头最长。 | 1 | 脖子和尾巴不比头长 | other_unparsed | S1T213 |
| 脖子和尾巴都比头长。 | 1 | 脖子和尾巴都比头长 | other_unparsed | S1T155 |
| 脖子和腿的长度在2/3以上，头和尾巴几乎是最小长度。 | 1 | 脖子和腿的长度在2/3以上 | other_unparsed | S1T2 |
| 脖子比头长，尾巴也比头长，脖子比尾巴长。 | 1 | 尾巴也比头长 | other_unparsed | S1T96 |
| 腿不是很短，脖子和尾巴比较长，比腿略长一些。 | 1 | 比腿略长一些 | other_unparsed | S1T69 |
| 腿中等，腿比尾巴长，脖子是尾巴的两倍以上，脖子也是头的两倍以上。 | 1 | 脖子是尾巴的两倍以上; 脖子也是头的两倍以上 | other_unparsed | S1T170 |
| 腿比较短，脖子不是最短。 | 1 | 脖子不是最短 | other_unparsed | S1T109 |
| 腿没有明显长于脖子和尾巴，且头和脖子长度相似。 | 1 | 腿没有明显长于脖子和尾巴 | other_unparsed | S1T51 |
| 腿没有那么长，腿的长度已经很长了，还是。 | 1 | 还是 | other_unparsed | S1T61 |
| 腿短不是最短，脖子和尾巴比脖子长一点，头是最短。 | 1 | 腿短不是最短 | other_unparsed | S1T258 |
| 腿短，且不是最短，脖子不是最长。 | 1 | 且不是最短; 脖子不是最长 | other_unparsed | S1T257 |
| 腿短，且不是最短，脖子最长。 | 1 | 且不是最短 | other_unparsed | S1T271 |
| 腿短，且最短，头比脖子长。 | 1 | 且最短 | other_unparsed | S1T243 |
| 腿短，且脖子不是另外三个里最短。 | 1 | 且脖子不是另外三个里最短 | other_reference | S1T129 |
| 腿短，且脖子不是最短。 | 1 | 且脖子不是最短 | other_unparsed | S1T136 |
| 腿短，尾巴和脖子都比腿短，头比腿长。 | 1 | 尾巴和脖子都比腿短 | other_unparsed | S1T231 |
| 腿短，尾巴比脖子长，头也比脖子长。 | 1 | 头也比脖子长 | other_unparsed | S1T181 |
| 腿短，尾巴比脖子长，有，类了。 | 1 | 有; 类了 | other_unparsed | S1T206 |
| 腿短，是最短，尾巴比腿短，脖子是最长。 | 1 | 是最短 | other_unparsed | S1T251 |
| 腿短，是最短，脖子不是最长。 | 1 | 是最短; 脖子不是最长 | other_unparsed | S1T268 |
| 腿短，是最短，脖子是最长。 | 1 | 是最短 | other_unparsed | S1T248 |
| 腿短，脖子不明显，短于尾巴和头。 | 1 | 脖子不明显; 短于尾巴和头 | other_unparsed | S1T119 |
| 腿短，脖子不是另外三个里最短。 | 1 | 脖子不是另外三个里最短 | other_reference | S1T117 |
| 腿短，脖子不是头、脖子、尾巴里最短。 | 1 | 脖子不是头、脖子、尾巴里最短 | other_unparsed | S1T112 |
| 腿短，脖子不是头、脖子和尾巴里最短。 | 1 | 脖子不是头、脖子和尾巴里最短 | other_unparsed | S1T141 |
| 腿短，脖子与头相近，脖子肯定不是最短。 | 1 | 脖子肯定不是最短 | other_unparsed | S1T152 |
| 腿短，脖子和尾巴几乎一样长，远长于头和腿。 | 1 | 远长于头和腿 | other_unparsed | S1T66 |
| 腿短，脖子和尾巴都不比腿短。 | 1 | 脖子和尾巴都不比腿短 | other_unparsed | S1T237 |
| 腿短，脖子是尾巴的两倍以上，脖子比头长很多。 | 1 | 脖子是尾巴的两倍以上 | other_unparsed | S1T168 |
| 腿短，脖子比头长，比尾巴短。 | 1 | 比尾巴短 | other_unparsed | S1T149 |
| 腿短，脖子比腿短，且，尾巴和头非常长。 | 1 | 且 | other_unparsed | S1T189 |
| 腿短，脖子短，头远长于脖子，尾巴中等长度。 | 1 | 头远长于脖子 | other_unparsed | S1T74 |
| 腿短，腿不是最短。 | 1 | 腿不是最短 | other_unparsed | S1T236 |
| 腿短，腿不是最短，头比腿短。 | 1 | 腿不是最短 | other_unparsed | S1T244 |
| 腿短，腿是最短，头、脖子、尾巴都比腿长，头是最长。 | 1 | 头、脖子、尾巴都比腿长 | other_unparsed | S1T234 |
| 腿长，尾巴最短，脖子和头分不清楚。 | 1 | 脖子和头分不清楚 | other_unparsed | S1T145 |
| 腿长，尾巴比脖子长，头也比脖子长。 | 1 | 头也比脖子长 | other_unparsed | S1T188 |

#### S317

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 有部位达到最长或最短长度。 | 6 | 有部位达到最长或最短长度 | count_abstract, extreme_endpoint | S2T109, S2T110, S2T113, S2T316, S2T317, S2T319 |
| 脖子不是最长的部位。 | 5 | 脖子不是最长的部位 | other_unparsed | S2T176, S2T177, S2T178, S2T179, S2T181 |
| 腿不是最短的部位。 | 5 | 腿不是最短的部位 | other_unparsed | S2T185, S2T186, S2T187, S2T189, S2T190 |
| 没有部位达到最长或最短长度。 | 4 | 没有部位达到最长或最短长度 | count_abstract, extreme_endpoint | S2T111, S2T112, S2T315, S2T318 |
| 腿没有达到最大长度。 | 4 | 腿没有达到最大长度 | extreme_endpoint | S2T71, S2T154, S2T155, S2T156 |
| 有比腿更短的部位。 | 3 | 有比腿更短的部位 | other_unparsed | S2T172, S2T173, S2T174 |
| 两个以上部位比躯干的一半长。 | 2 | 两个以上部位比躯干的一半长 | body_geometry | S2T62, S2T63 |
| 头不比腿短。 | 2 | 头不比腿短 | other_unparsed | S2T124, S2T125 |
| 小于两个部位一样长。 | 2 | 小于两个部位一样长 | count_abstract | S2T313, S2T314 |
| 尾巴不短于脖子。 | 2 | 尾巴不短于脖子 | other_unparsed | S2T67, S2T68 |
| 有两个部位达到最长长度。 | 2 | 有两个部位达到最长长度 | count_abstract, extreme_endpoint | S2T286, S2T288 |
| 没有部位达到最长长度。 | 2 | 没有部位达到最长长度 | count_abstract, extreme_endpoint | S2T83, S2T94 |
| 腿没有达到最长或最短。 | 2 | 腿没有达到最长或最短 | extreme_endpoint | S2T76, S2T77 |
| 一个部位达到最长长度。 | 1 | 一个部位达到最长长度 | count_abstract, extreme_endpoint | S2T89 |
| 出现两个最长长度。 | 1 | 出现两个最长长度 | extreme_endpoint | S2T93 |
| 出现最长长度，且出现了三个最长长度。 | 1 | 出现最长长度; 且出现了三个最长长度 | extreme_endpoint | S2T92 |
| 大于两个部位一样长。 | 1 | 大于两个部位一样长 | count_abstract | S2T312 |
| 头不是最短的部位。 | 1 | 头不是最短的部位 | other_unparsed | S2T192 |
| 头长度不变。 | 1 | 头长度不变 | other_unparsed | S3T105 |
| 尾巴大约3/4躯干，脖子大约3/4躯干。 | 1 | 尾巴大约3/4躯干; 脖子大约3/4躯干 | body_geometry | S3T294 |
| 有三个部位和躯干一样长。 | 1 | 有三个部位和躯干一样长 | count_abstract, body_geometry | S1T15 |
| 有两个部位比躯干长。 | 1 | 有两个部位比躯干长 | count_abstract, body_geometry | S1T18 |
| 有部位达到最长或最短。 | 1 | 有部位达到最长或最短 | count_abstract, extreme_endpoint | S2T79 |
| 没有两个部位达到最长长度。 | 1 | 没有两个部位达到最长长度 | count_abstract, extreme_endpoint | S2T287 |
| 没有出现最小长度，尾巴小于脖子。 | 1 | 没有出现最小长度 | extreme_endpoint | S2T91 |
| 没有出现最长长度。 | 1 | 没有出现最长长度 | extreme_endpoint | S2T90 |
| 没有达到最长长度。 | 1 | 没有达到最长长度 | extreme_endpoint | S2T95 |
| 没有部位是最大或最小长度。 | 1 | 没有部位是最大或最小长度 | count_abstract, extreme_endpoint | S2T80 |
| 没有部位是最长或者最短。 | 1 | 没有部位是最长或者最短 | count_abstract | S2T78 |
| 没有部位是最长的。 | 1 | 没有部位是最长的 | count_abstract | S2T108 |
| 没有部位达到最长或最短的长度。 | 1 | 没有部位达到最长或最短的长度 | count_abstract, extreme_endpoint | S2T114 |
| 腿和头长度变化。 | 1 | 腿和头长度变化 | other_unparsed | S3T107 |
| 腿长度变化。 | 1 | 腿长度变化 | other_unparsed | S3T106 |
| 达到最长或最短长度的部位数小于等于一。 | 1 | 达到最长或最短长度的部位数小于等于一 | extreme_endpoint | S2T320 |

#### S318

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 判断不出哪一部分更长，尾巴不短。 | 1 | 判断不出哪一部分更长 | other_unparsed | S1T248 |
| 头不比尾巴长，腿比较长。 | 1 | 头不比尾巴长 | other_unparsed | S1T241 |
| 头和尾巴差不多长，腿不短，脖子比较长，算太长。 | 1 | 算太长 | other_unparsed | S1T263 |
| 头和尾巴相比，头更长，脖子比较短。 | 1 | 头和尾巴相比 | other_unparsed | S1T260 |
| 头和尾巴相比，尾巴更长，脖子和尾巴都很长，躯干腿很短。 | 1 | 头和尾巴相比 | other_unparsed | S1T259 |
| 头很长，头和尾巴相差很大，脖子也很长。 | 1 | 头和尾巴相差很大 | other_unparsed | S1T203 |
| 头最长，躯干头和尾巴相差很大。 | 1 | 躯干头和尾巴相差很大 | body_geometry | S1T202 |
| 头没有比尾巴长很多，脖子很长。 | 1 | 头没有比尾巴长很多 | other_unparsed | S1T215 |
| 尾巴和躯干看不清。 | 1 | 尾巴和躯干看不清 | body_geometry | S2T110 |
| 尾巴比腿长，头和脖子，脖子更长。 | 1 | 头和脖子 | other_unparsed | S1T72 |
| 尾巴比腿长，差距不大，脖子和头都很长。 | 1 | 差距不大 | other_unparsed | S1T104 |
| 尾巴比腿长，躯干差距很大。 | 1 | 躯干差距很大 | body_geometry | S1T102 |
| 尾巴比躯干看不清，脖子比较短，腿比较长。 | 1 | 尾巴比躯干看不清 | body_geometry | S2T107 |
| 尾巴比躯干长，腿和躯干差不多。 | 1 | 腿和躯干差不多 | body_geometry, global_balance | S2T116 |
| 脖子最短，尾巴比腿短，比脖子长，头也比较长。 | 1 | 比脖子长 | other_unparsed | S1T46 |
| 脖子没有明显的优势，腿比较长，腿和尾巴都很长。 | 1 | 脖子没有明显的优势 | other_unparsed | S1T272 |
| 腿最短，脖子和尾巴一样长，比腿长，头最长。 | 1 | 比腿长 | other_unparsed | S1T63 |

#### S319

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和脖子的比例大于腿和尾巴。 | 26 | 头和脖子的比例大于腿和尾巴 | proportion_or_ratio | S1T88, S1T100, S1T103, S1T108, S1T109, S1T111, S1T113, S1T115 |
| 头和脖子的比例小于腿和尾巴。 | 10 | 头和脖子的比例小于腿和尾巴 | proportion_or_ratio | S1T98, S1T101, S1T102, S1T104, S1T112, S1T118, S1T122, S1T123 |
| 选错了。 | 6 | 选错了 | meta_or_uncertain | S2T201, S2T267, S2T283, S3T66, S3T102, S3T123 |
| 头和脖子都比腿长。 | 3 | 头和脖子都比腿长 | other_unparsed | S2T252, S3T52, S3T53 |
| 三个部位都比躯干长。 | 2 | 三个部位都比躯干长 | count_abstract, body_geometry | S4T50, S4T51 |
| 三个部位长于躯干。 | 2 | 三个部位长于躯干 | count_abstract, body_geometry | S2T305, S2T307 |
| 四个部位长度比较平衡。 | 2 | 四个部位长度比较平衡 | other_unparsed | S2T219, S2T225 |
| 三个部位都比较长。 | 1 | 三个部位都比较长 | count_abstract | S3T286 |
| 四个部位比较均等。 | 1 | 四个部位比较均等 | global_balance | S2T320 |
| 四个部位长度都小于躯干，且差不多长。 | 1 | 且差不多长 | global_balance | S3T78 |
| 四个部位长度都很平衡。 | 1 | 四个部位长度都很平衡 | other_unparsed | S2T279 |
| 四个部位长度都相同。 | 1 | 四个部位长度都相同 | other_unparsed | S2T76 |
| 头和脖子比例小于腿和尾巴。 | 1 | 头和脖子比例小于腿和尾巴 | proportion_or_ratio | S1T134 |
| 头和脖子的比例大于尾巴和腿。 | 1 | 头和脖子的比例大于尾巴和腿 | proportion_or_ratio | S1T151 |
| 头和脖子的比例小于腿和尾巴，腿长。 | 1 | 头和脖子的比例小于腿和尾巴 | proportion_or_ratio | S1T161 |
| 头和脖子的比例等于腿和尾巴。 | 1 | 头和脖子的比例等于腿和尾巴 | proportion_or_ratio | S1T138 |
| 头和脖子都长于腿和尾巴。 | 1 | 头和脖子都长于腿和尾巴 | other_unparsed | S1T33 |
| 头有点小。 | 1 | 头有点小 | other_unparsed | S1T255 |
| 尾巴和腿的比例大于头和脖子的比例。 | 1 | 尾巴和腿的比例大于头和脖子的比例 | proportion_or_ratio | S1T187 |
| 尾巴长，腿很小。 | 1 | 腿很小 | vague_size | S1T295 |
| 比较均匀。 | 1 | 比较均匀 | global_balance | S1T278 |
| 腿和尾巴的比例大于头和脖子。 | 1 | 腿和尾巴的比例大于头和脖子 | proportion_or_ratio | S1T147 |
| 腿和尾巴的比例大于脖子和头的比例。 | 1 | 腿和尾巴的比例大于脖子和头的比例 | proportion_or_ratio | S1T172 |
| 长度都很平均。 | 1 | 长度都很平均 | global_balance | S4T108 |
| 除了头以外，其他部位都很长，很平均。 | 1 | 很平均 | global_balance | S3T197 |
| 除了尾巴以外，其他三个部位都很长，而且很平均。 | 1 | 而且很平均 | global_balance | S3T145 |
| 除了尾巴，都小于躯干。 | 1 | 除了尾巴; 都小于躯干 | body_geometry | S4T62 |
| 除了脖子，都很平衡。 | 1 | 除了脖子; 都很平衡 | other_unparsed | S4T95 |
| 除了腿以外，其他部位都很长，且平均。 | 1 | 且平均 | global_balance | S3T198 |

#### S321

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿不是最短的，头比脖子长。 | 2 | 腿不是最短的 | other_unparsed | S2T92, S2T109 |
| 头比脖子短，尾巴也比腿短。 | 1 | 尾巴也比腿短 | other_unparsed | S2T35 |
| 头比脖子短，腿略微比尾巴短。 | 1 | 腿略微比尾巴短 | other_unparsed | S2T13 |
| 头相对，脖子较短。 | 1 | 头相对 | other_unparsed | S2T71 |
| 脖子比头短，比尾巴也短，腿相对较短。 | 1 | 比尾巴也短 | other_unparsed | S2T138 |
| 脖子比头短，比尾巴长，腿相对较短。 | 1 | 比尾巴长 | other_unparsed | S2T139 |
| 脖子比头短，腿与尾巴都比脖子长。 | 1 | 腿与尾巴都比脖子长 | other_unparsed | S2T87 |
| 腿不是最短的，头比脖子短。 | 1 | 腿不是最短的 | other_unparsed | S2T90 |
| 腿不是最短的，头短于脖子。 | 1 | 腿不是最短的 | other_unparsed | S2T93 |
| 腿不是最短，头比脖子和尾巴短。 | 1 | 腿不是最短 | other_unparsed | S2T106 |
| 腿不是最短，头比脖子长。 | 1 | 腿不是最短 | other_unparsed | S2T108 |
| 腿不是最短，头较脖子更短。 | 1 | 腿不是最短 | other_unparsed | S2T102 |
| 腿不是最短，尾巴比头和脖子都长。 | 1 | 腿不是最短 | other_unparsed | S2T103 |
| 腿不是最短，腿较长。 | 1 | 腿不是最短 | other_unparsed | S2T148 |
| 腿超级短，头、脖子和尾巴都比腿长。 | 1 | 头、脖子和尾巴都比腿长 | other_unparsed | S1T29 |
| 腿较短，头比尾巴轻。 | 1 | 头比尾巴轻 | other_unparsed | S2T150 |
| 腿较短，是最短。 | 1 | 是最短 | other_unparsed | S2T147 |
| 腿较长，头比尾巴重。 | 1 | 头比尾巴重 | other_unparsed | S2T149 |
| 腿较，尾巴和脖子都长，头也长。 | 1 | 腿较 | other_unparsed | S1T78 |

#### S322

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头不是最长。 | 21 | 头不是最长 | other_unparsed | S1T186, S1T187, S1T223, S1T226, S1T227, S1T230, S1T233, S1T236 |
| 四个部位长度各不相同，脖子长于躯干。 | 1 | 四个部位长度各不相同 | disjoint_inequality | S2T111 |
| 四个部位长度都不一样，尾巴最长，头和腿短于躯干，脖子和尾巴长于躯干。 | 1 | 四个部位长度都不一样 | disjoint_inequality | S2T106 |
| 头不是最长，头和脖子一样长，腿和尾巴一样长。 | 1 | 头不是最长 | other_unparsed | S2T7 |
| 头不是最长，尾巴最长。 | 1 | 头不是最长 | other_unparsed | S2T6 |
| 头不是最长，脖子最长， | 1 | 头不是最长 | other_unparsed | S2T4 |
| 头不是最长，腿最长， | 1 | 头不是最长 | other_unparsed | S2T3 |
| 头和尾巴长度相等，略短于脖子。 | 1 | 略短于脖子 | other_unparsed | S1T191 |
| 头和脖子一样长，与腿相近。 | 1 | 与腿相近 | other_unparsed | S1T205 |
| 头和脖子一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S2T26 |
| 头和脖子长度相等，略长于尾巴。 | 1 | 略长于尾巴 | other_unparsed | S1T207 |
| 头和腿一样长，且不是最长。 | 1 | 且不是最长 | other_unparsed | S1T302 |
| 头和腿一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S2T32 |
| 头和腿一样长，中等偏长，脖子和尾巴一样长，略长于前两者。 | 1 | 中等偏长; 略长于前两者 | other_unparsed | S1T39 |
| 头比腿长，四个部位长度不相等。 | 1 | 四个部位长度不相等 | other_unparsed | S2T158 |
| 尾巴很短，腿和头差不多长，略长于尾巴，脖子最长。 | 1 | 略长于尾巴 | other_unparsed | S1T30 |
| 尾巴略长于头，四个部位长度不相等。 | 1 | 四个部位长度不相等 | other_unparsed | S1T85 |
| 短于躯干，两个长于躯干，脖子是最长。 | 1 | 短于躯干; 两个长于躯干 | body_geometry | S2T17 |
| 短于躯干，头和躯干一样长。 | 1 | 短于躯干 | body_geometry | S2T203 |
| 短于躯干，尾巴和躯干一样长。 | 1 | 短于躯干 | body_geometry | S2T202 |
| 短于躯干，有两个长于躯干。 | 1 | 短于躯干; 有两个长于躯干 | body_geometry | S2T20 |
| 短于躯干，腿最长，脖子很长。 | 1 | 短于躯干 | body_geometry | S2T204 |
| 脖子和头差不多长，腿和尾巴差不多长，且长于前二者。 | 1 | 且长于前二者 | other_unparsed | S1T26 |
| 脖子和头非常长，腿略短于二者，尾巴最短。 | 1 | 腿略短于二者 | other_unparsed | S1T19 |
| 脖子和尾巴一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S2T27 |
| 脖子和尾巴一样长，和头一样长。 | 1 | 和头一样长 | other_unparsed | S2T174 |
| 脖子和尾巴一样长，头不是最长。 | 1 | 头不是最长 | other_unparsed | S2T18 |
| 脖子和尾巴一样长，头略短于二者，腿最短。 | 1 | 头略短于二者 | other_unparsed | S1T18 |
| 脖子和腿一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S1T320 |
| 脖子和腿差不多长，头和尾巴差不多长，且长于前二者。 | 1 | 且长于前二者 | other_unparsed | S1T22 |
| 脖子和腿长度一样，头不是最长。 | 1 | 头不是最长 | other_unparsed | S2T9 |
| 脖子和腿长度相等，头不是最长。 | 1 | 头不是最长 | other_unparsed | S1T289 |
| 脖子和腿，头比脖子和腿长，尾巴很短。 | 1 | 脖子和腿 | other_unparsed | S1T2 |
| 脖子最长，头不是最长。 | 1 | 头不是最长 | other_unparsed | S2T8 |
| 腿和尾巴一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S2T31 |
| 腿长，头、脖子、尾巴都不一样长。 | 1 | 头、脖子、尾巴都不一样长 | disjoint_inequality | S3T89 |

#### S323

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 选错了。 | 3 | 选错了 | meta_or_uncertain | S1T146, S1T156, S1T157 |

#### S324

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿和头较长，脖子长度中，脖子长度中等，尾巴较短。 | 1 | 脖子长度中 | other_unparsed | S1T47 |
| 腿和脖子较长，头和尾巴长中等，较短。 | 1 | 较短 | other_unparsed | S1T44 |
| 腿较短，前已经较长。 | 1 | 前已经较长 | other_unparsed | S1T28 |
| 腿较长，下巴为中等。 | 1 | 下巴为中等 | other_unparsed | S1T88 |

#### S325

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 每个都长。 | 3 | 每个都长 | other_unparsed | S1T133, S1T134, S1T140 |
| 每一个都很长。 | 1 | 每一个都很长 | other_unparsed | S1T194 |

#### S326

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 假设脖子无关。 | 6 | 假设脖子无关 | meta_or_uncertain | S1T243, S1T244, S1T245, S1T246, S1T247, S1T248 |
| 朝右。 | 4 | 朝右 | other_unparsed | S1T81, S1T82, S1T83, S1T84 |
| 假设尾巴无关。 | 2 | 假设尾巴无关 | meta_or_uncertain | S1T241, S1T242 |
| 腿和躯干差不多，头比脖子长。 | 2 | 腿和躯干差不多 | body_geometry, global_balance | S1T32, S1T33 |
| 腿比尾巴短，头在躯干以下。 | 2 | 头在躯干以下 | body_geometry | S1T125, S1T126 |
| 分布均匀，像猫。 | 1 | 分布均匀; 像猫 | global_balance | S1T222 |
| 头和脖子一样长，尾巴也差不多长，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T8 |
| 头和脖子很长，尾巴很短，头在躯干以上。 | 1 | 头在躯干以上 | body_geometry | S1T9 |
| 头和脖子比尾巴长，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T11 |
| 头在躯干以上，腿长，尾巴短。 | 1 | 头在躯干以上 | body_geometry | S1T40 |
| 头在躯干以下，尾巴和腿差不多。 | 1 | 头在躯干以下 | body_geometry | S1T47 |
| 头很长，脖子、尾巴很短，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T10 |
| 头最长，尾巴短，像狗。 | 1 | 像狗 | other_unparsed | S1T220 |
| 头长，尾巴短，脖子短，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T6 |
| 头长，尾巴短，脖子长，头在躯干以上。 | 1 | 头在躯干以上 | body_geometry | S1T5 |
| 头长，脖子一般，腿长，尾巴短，右朝向。 | 1 | 右朝向 | other_unparsed | S1T4 |
| 头长，脖子短，尾巴短，头在躯干以下、腿以上。 | 1 | 头在躯干以下、腿以上 | body_geometry | S1T1 |
| 头长，脖子长，尾巴短，头在躯干和腿之间，右朝向。 | 1 | 头在躯干和腿之间; 右朝向 | body_geometry | S1T3 |
| 头长，腿长，像狗。 | 1 | 像狗 | other_unparsed | S1T221 |
| 尾巴最长，依次是脖子、腿、头。 | 1 | 依次是脖子、腿、头 | other_unparsed | S1T209 |
| 尾巴长，头在躯干和腿之间。 | 1 | 头在躯干和腿之间 | body_geometry | S1T122 |
| 左朝向，头在躯干以下、腿以上，脖子和尾巴一般。 | 1 | 左朝向; 头在躯干以下、腿以上 | body_geometry | S1T2 |
| 朝右，脖子长，尾巴长。 | 1 | 朝右 | other_unparsed | S1T78 |
| 朝左。 | 1 | 朝左 | other_unparsed | S1T80 |
| 朝左，头短。 | 1 | 朝左 | other_unparsed | S1T79 |
| 朝左，脖子短，头短。 | 1 | 朝左 | other_unparsed | S1T77 |
| 狗状。 | 1 | 狗状 | other_unparsed | S1T295 |
| 脖子和头一样长，头在躯干以上。 | 1 | 头在躯干以上 | body_geometry | S1T16 |
| 脖子和尾巴一样长，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T15 |
| 脖子最长，依次是头、腿、尾巴。 | 1 | 依次是头、腿、尾巴 | other_unparsed | S1T210 |
| 腿最长，依次是脖子、头、尾巴。 | 1 | 依次是脖子、头、尾巴 | other_unparsed | S1T211 |
| 腿比较短，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T48 |
| 腿比较长，头在躯干以上，尾巴最长。 | 1 | 头在躯干以上 | body_geometry | S1T39 |
| 腿比较长，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T38 |
| 腿短，头在躯干以上，尾巴长。 | 1 | 头在躯干以上 | body_geometry | S1T46 |
| 腿长，像狗。 | 1 | 像狗 | other_unparsed | S1T224 |
| 腿长，头在躯干以下，尾巴长。 | 1 | 头在躯干以下 | body_geometry | S1T41 |
| 腿长，头在躯干以下，脖子、尾巴长。 | 1 | 头在躯干以下 | body_geometry | S1T42 |
| 腿长，脖子最短，然后是头和尾巴。 | 1 | 然后是头和尾巴 | ordinal_or_secondary | S1T21 |
| 蜥蜴状。 | 1 | 蜥蜴状 | other_unparsed | S1T294 |

#### S327

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短，尾巴短，脖子长，腿中上。 | 1 | 腿中上 | other_unparsed | S1T43 |

#### S328

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴和腿都比较长，差不多，头加脖子比较短。 | 1 | 差不多 | global_balance | S1T95 |
| 尾巴特别短，尾巴明显比腿短，头和脖子加在一起比较长。 | 1 | 尾巴明显比腿短 | other_unparsed | S1T23 |
| 尾巴短，头和脖子都比腿长。 | 1 | 头和脖子都比腿长 | other_unparsed | S2T82 |
| 尾巴短，脖子比对长。 | 1 | 脖子比对长 | other_unparsed | S2T227 |
| 尾巴长，头和脖子都很短，都比腿短。 | 1 | 都比腿短 | other_unparsed | S2T39 |
| 腿和尾巴差不多长，都是中等长度，头和脖子都比较长。 | 1 | 都是中等长度 | other_unparsed | S1T52 |

#### S329

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 按错了。 | 2 | 按错了 | other_unparsed | S1T185, S1T251 |
| 全身都长。 | 1 | 全身都长 | other_unparsed | S1T70 |
| 尾巴长，其次是腿，然后是头和脖子。 | 1 | 然后是头和脖子 | ordinal_or_secondary | S1T87 |

#### S331

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 均等。 | 1 | 均等 | global_balance | S1T154 |
| 头长，尾巴长，腿也还行。 | 1 | 腿也还行 | other_unparsed | S1T25 |
| 躯干长，头短。 | 1 | 躯干长 | body_geometry | S1T29 |

## Region 未编码分析

### 被试摘要

| iSub | unprocessed_trials | unique_texts | top_categories | top_texts |
| --- | --- | --- | --- | --- |
| 102 | 49 | 38 | other_unparsed:45; count_abstract:2; meta_or_uncertain:1; global_balance:1 | 脖子比头长，但是是最短的两个。:4; 脖子比头短，但是是最长的两个。:3; 脖子比头短，都短于腿。:3; 脖子比头短，它们两个是最长的。:3; 脖子比头短，都长于腿。:2 |
| 103 | 12 | 12 | other_unparsed:12 | 尾巴很长，长于头和脖子，腿较短。:1; 腿极长，尾巴中等长度，长于头和脖子。:1; 腿很长，脖子和头中等长度，长于较短的尾巴。:1; 腿较短，尾巴较长，长于头和脖子。:1; 腿很短，脖子很长，长于尾巴，尾巴和头中等长度。:1 |
| 104 | 2 | 2 | global_balance:2 | 头长，较均匀。:1; 比较均匀。:1 |
| 105 | 1 | 1 | other_unparsed:1 | 腿短于阈值。:1 |
| 106 | 28 | 18 | body_geometry:20; other_unparsed:7; count_abstract:1 | 躯体下方比上面高。:7; 头离地面近。:2; 下面的部分更高一些。:2; 腿不是最长的。:2; 躯体下方没有上面高。:2 |
| 107 | 5 | 5 | other_unparsed:3; proportion_or_ratio:1; global_balance:1 | 头和尾巴很长，脖子也挺长，但相对短一些，腿很短。:1; 腿、脖子和头比例比较协调，尾巴短。:1; 腿、脖子和头长度差不多，都不算很长。:1; 尾巴、腿、脖子和头长度差不多，都还挺长的。:1; 腿、脖子和头不算特别长，且长度相当。:1 |
| 108 | 21 | 10 | other_unparsed:19; meta_or_uncertain:2 | 脖子不是最长的。:8; 腿比脖子短，比尾巴长。:3; 选错了。:2; 腿比脖子和尾巴短，比头长。:2; 头和脖子显著比腿和尾巴长。:1 |
| 109 | 2 | 2 | other_unparsed:2 | 脖子，头较长，腿一般。:1; 腿较长，脖子一般，头较小。:1 |
| 110 | 1 | 1 | other_unparsed:1 | 头短，脖子腿，尾巴长。:1 |
| 111 | 1 | 1 | other_unparsed:1 | 腿短，尾巴、头、脖子长度相同。:1 |
| 112 | 6 | 6 | other_unparsed:5; disjoint_inequality:1 | 尾巴和腿之间关系。:1; 尾巴比头长一点，比腿和脖子都短。:1; 尾巴跟腿之间的关系以及跟脖子的关系。:1; 尾巴跟脖子不一样长。:1; 脖子和腿之间的关系。:1 |
| 113 | 26 | 23 | other_unparsed:23; meta_or_uncertain:2; global_balance:1 | 脖子长，比头长。:3; 头比脖子长，都很长。:2; 脖子比头长，头比腿长，腿比尾巴长，都有些长。:1; 头最长，尾巴最短，脖子和腿差不多长，都比较长。:1; 尾巴比脖子长，但是都很短，脖子和腿很长，腿特别长。:1 |
| 114 | 3 | 2 | global_balance:2; other_unparsed:1 | 都差不多长。:2; 都不长。:1 |
| 116 | 5 | 5 | other_unparsed:5 | 四个部位都比较长，其中腿最明显。:1; 腿的长度最突出。:1; 脖子的长度很突出。:1; 腿和脖子都比较长，但腿是最明显的。:1; 嗯，脖子比较长。:1 |
| 118 | 65 | 27 | count_abstract:50; other_unparsed:10; extreme_endpoint:3; global_balance:2 | 三个部位很长。:17; 只有一个部位很长。:7; 两个部位很长。:6; 一个部位很长。:5; 只有两个部位很长。:3 |
| 119 | 205 | 36 | other_unparsed:150; count_abstract:52; body_geometry:37 | 头短于中间值。:62; 头长于中间值。:53; 头不是最长。:14; 有一个部位长于躯干。:13; 有三个部位长于躯干。:7 |
| 120 | 29 | 20 | body_geometry:10; other_reference:9; count_abstract:7; other_unparsed:6; global_balance:4 | 有奇数个部位长于躯干。:4; 有偶数个部位长于躯干。:3; 腿明显短于其他部位。:2; 头明显长于其他部位。:2; 脖子和尾巴的长度不一样。:2 |
| 121 | 27 | 20 | body_geometry:12; other_unparsed:8; count_abstract:6; global_balance:1 | 腿不是所有部位里最短的。:4; 头在躯干上方，腿比较长。:3; 腿不是最短的部位。:2; 头不是所有部位里最短的。:2; 头比脖子长一点。腿和尾巴差不多，都很短。:1 |
| 123 | 7 | 7 | other_unparsed:6; count_abstract:1; body_geometry:1 | 腿长，脖子，尾巴、腿都偏短。:1; 腿短，脖，脖子和头长，尾巴短。:1; 头、脖子、腿、尾巴、躯干这些部位的长度进行描述，但不一定用的是这几个词语，且可能会涉及到这几个部位之间的比较，包括大小和长短关系。:1; 腿和头。头短，尾巴和脖子长。:1; 头和脖外，头和腿长。:1 |
| 125 | 16 | 14 | other_unparsed:15; proportion_or_ratio:1 | 腿非常短，几乎是最短。:2; 腿很长，基本是最长。:2; 头长，脖子长，腿较长，尾巴长，脖子比腿长一点，头和脖子占的比重比较大。:1; 头长，脖子长，腿长，尾巴短，头和脖子的比重比较大，脖子比腿长一点。:1; 头长，脖子较短，腿短，尾巴较短，头、脖子、尾巴均比腿长。:1 |
| 127 | 2 | 2 | other_unparsed:2 | 腿和头比较长，脖子后尾巴稍微短，一点。:1; 差不多，都一样长。:1 |
| 129 | 62 | 19 | body_geometry:22; count_abstract:21; disjoint_inequality:15; other_unparsed:10 | 一个部位很长。:10; 头在躯干之下。:9; 尾巴和脖子不一样长。:7; 有两个部位一样长。:6; 低头。:4 |
| 130 | 38 | 10 | other_unparsed:28; proportion_or_ratio:10 | 腿和尾巴都比脖子长。:16; 头身比例比较协调。:5; 尾巴和腿都比脖子短。:5; 头身比例不协调。:4; 腿和尾巴都比脖子短。:2 |
| 131 | 39 | 29 | other_unparsed:23; global_balance:15; vague_size:13 | 体型分布的不均匀。:4; 体型分布的很均匀。:3; 朝左边。:3; 朝右边。:2; 腿比头长，朝右边。:2 |
| 132 | 14 | 4 | other_unparsed:14 | 尾巴不是最短的。:8; 尾巴不是最长的。:3; 尾巴短于某个数值。:2; 尾巴长于某个数值。:1 |
| 202 | 1 | 1 | meta_or_uncertain:1 | 选错了。:1 |
| 203 | 11 | 11 | other_unparsed:8; body_geometry:2; global_balance:1; meta_or_uncertain:1 | 身体各个部位都很匀称。:1; 腿很长，颈部也很短。:1; 腿很长，头很短，脖子和尾巴长度对称。:1; 头和尾巴均长于脖子，腿很短。:1; 头显著短、短于脖子，尾巴很长，腿很短。:1 |
| 204 | 14 | 14 | other_unparsed:7; global_balance:4; vague_size:2; meta_or_uncertain:1; body_geometry:1 | 头、脖子、尾巴都很长，腿比它们稍微短一点。:1; 头和尾巴都比脖子长。:1; 四个部位几乎一样长，都挺长。:1; 脖子很长，头很小。:1; 头很小，尾巴很长，脖子也很长。:1 |
| 205 | 4 | 4 | other_unparsed:3; global_balance:1 | 头、脖子和腿，差不多长。:1; 腿，头最长。:1; 腿，尾巴最长。:1; 腿，脖子最长。:1 |
| 206 | 23 | 21 | other_unparsed:21; global_balance:1; vague_size:1 | 脖子比头长，腿也比尾巴长。:2; 脖子比头短，腿也比尾巴短。:2; 短长。:1; 头和脖子。:1; 脖子很短，腿、尾巴，头很长。:1 |
| 207 | 27 | 23 | vague_size:14; other_unparsed:12; count_abstract:1; global_balance:1 | 四个部位都很长，体型很大。:4; 四个部位长度差不多，都比较长。:2; 尾巴最长，其他三个部位一样长，都比较长。:1; 头和脖子一样长，都是最长的，其他两个比较短。:1; 四个部位都比较长，长度接近。:1 |
| 208 | 22 | 21 | other_unparsed:16; global_balance:4; meta_or_uncertain:1; count_abstract:1 | 四者都很长。:2; 脖子很长，尾巴稍短一些，腿和腿一样长。:1; 头和，四个部位都长。:1; 四者都很短，尤其是腿很短。:1; 腿偏短，脖子、尾巴和头都偏长，一样长。:1 |
| 209 | 28 | 25 | other_unparsed:26; global_balance:2 | 头和尾巴长，脖子和腿短，头明显比脖子长。:2; 头最长，脖子短，腿和尾巴中等，头明显比脖子长。:2; 头最长，腿中等，脖子和尾巴短，头明显比脖子长。:2; 所有部位都是中等偏长，而且长度差不多。:1; 脖子很长，头，中等长度，尾巴和腿比较短。:1 |
| 210 | 121 | 59 | count_abstract:85; other_unparsed:26; body_geometry:7; global_balance:4; ordinal_or_secondary:1 | 有三个部位几乎一样长。:13; 三个部位几乎一样长。:11; 有两个部位几乎一样长。:9; 有三个部位长度一样。:7; 有三个部位一样长。:6 |
| 211 | 6 | 6 | other_unparsed:6 | 头最长，比脖子和尾巴都长。:1; 头，和腿基本一样长。:1; 头和脖子，尾巴和腿中有三个是一样长。:1; 头和尾巴都比较长，一样长，腿也比较长。:1; 四个部位长度差不多，比较长。:1 |
| 212 | 136 | 47 | other_unparsed:103; global_balance:32; vague_size:1 | 四个部位较均等。:17; 头比脖子短，比尾巴短。:10; 尾巴比脖子长，比头长。:7; 头比脖子长，比尾巴短。:7; 头比脖子长，比尾巴长。:7 |
| 213 | 31 | 20 | other_unparsed:18; count_abstract:12; global_balance:1 | 有两个部位比较长。:7; 有三个部位比较长。:3; 腿较短，头和尾巴均长于脖子。:2; 脖子比头长，脖子不是最长。:2; 有一个部位比较长。:2 |
| 214 | 53 | 11 | meta_or_uncertain:39; other_unparsed:8; global_balance:3; count_abstract:3 | 选错了。:39; 两长两短。:3; 差不多。:3; 头，和，腿、头和腿略短，其他中等长度。:1; 腿长，脖，脖子中等长度，尾巴略长，头略短。:1 |
| 215 | 39 | 15 | count_abstract:34; disjoint_inequality:3; other_unparsed:2 | 两个部位长，两个部位短。:7; 三个部位长，一个部位短。:6; 三个部位长。:6; 三长一短。:3; 三个部位短，一个部位长。:3 |
| 216 | 18 | 17 | other_unparsed:15; global_balance:3 | 尾巴和腿差不多长，都比脖子长。:2; 头、脖子、腿都很长，尾巴中等，微微偏短。:1; 每个部位。:1; 尾巴和腿差不多长，偏短。:1; 尾巴和腿都一样长，偏短。:1 |
| 217 | 6 | 5 | other_unparsed:3; count_abstract:2; body_geometry:1 | 尾巴长，有两个部位短。:2; 由长到短是头和腿，脖子和尾巴。:1; 由长到短是头、腿、脖子，尾巴。:1; 头比和脖子比较长。:1; 都比躯干短。:1 |
| 218 | 108 | 91 | other_unparsed:100; other_reference:8; count_abstract:2 | 脖子和尾巴都较长，明显长于腿。:4; 脖子和尾巴都较短，明显短于腿。:3; 脖子、尾巴都长于腿。:3; 头、尾巴长度明显长于脖子、腿。:2; 尾巴明显长于其余三部位。:2 |
| 219 | 51 | 26 | global_balance:30; other_unparsed:18; meta_or_uncertain:2; vague_size:1 | 比较均衡。:19; 头和尾巴。:3; 稍微短。:3; 均衡。:3; 选错了。:2 |
| 220 | 82 | 76 | other_unparsed:61; disjoint_inequality:8; other_reference:7; global_balance:4; count_abstract:4 | 脖子和尾巴长于腿，长于头。:3; 头和尾巴不一样长，且腿长。:3; 头和尾巴一样长，且脖子和腿不一样长。:2; 头、脖子和腿都比尾巴长。:2; 头、脖子、尾巴和腿都比尾巴短。:1 |
| 221 | 4 | 4 | other_unparsed:2; vague_size:1; meta_or_uncertain:1 | 脖子和尾巴相差比较大，腿和头相差比较小。:1; 腿不是最长，头不是最短。:1; 脖子，最长。:1; 选错了。:1 |
| 222 | 110 | 97 | vague_size:69; other_unparsed:30; count_abstract:7; other_reference:2; global_balance:2 | 头很大。:4; 体型中等，尾巴和脖子比腿长。:3; 四个部位差不多长，体型偏大。:2; 体型中等，四个部位差不多长。:2; 体型大，四个部位都差不多。:2 |
| 223 | 2 | 2 | other_unparsed:2 | 头适中，脖子短，腿长，尾巴短，脖子比腿短，头也比腿短。:1; 头适中，脖子短，腿适中，尾巴短，头比脖子长，头也比腿短。:1 |
| 224 | 28 | 28 | other_unparsed:19; count_abstract:4; other_reference:3; proportion_or_ratio:2; global_balance:2 | 头比尾巴长，比脖子长。:1; 头最长，尾巴，腿很短。:1; 头和脖子相对。:1; 头比脖子长，比尾巴长。:1; 四个部位都差不多长，腿最长，长度都适中。:1 |
| 225 | 2 | 2 | other_unparsed:2 | 脖子远比头长。:1; 头远比脖子长。:1 |
| 226 | 108 | 73 | other_unparsed:62; count_abstract:27; global_balance:12; proportion_or_ratio:5; body_geometry:2 | 两长两短。:12; 三长一短。:11; 三个差不多长。:4; 两短两长。:3; 三短一长。:3 |
| 227 | 21 | 11 | disjoint_inequality:10; other_unparsed:8; body_geometry:2; global_balance:1; count_abstract:1 | 四个部位长度各不相同。:7; 脖子和腿长度近似。:2; 四个部位长度不一。:2; 尾巴和腿长度近似。:2; 四个部位长度不同。:2 |
| 228 | 20 | 7 | global_balance:15; other_unparsed:3; count_abstract:1; body_geometry:1 | 比较均匀。:10; 均匀。:5; 三长一短。:1; 脖子不是最短。:1; 整体都比较长。:1 |
| 231 | 23 | 21 | other_unparsed:12; global_balance:10; ordinal_or_secondary:1 | 肘比腿长，脖子比尾巴长。:2; 各个部分差不多长。:2; 脖子最短，腿和尾巴最长，并且差不多。:1; 尾巴最长，其他部位稍短，并且长度差不多。:1; 尾巴最长，然后是脖子，其他两个部位差不多。:1 |
| 301 | 36 | 34 | other_unparsed:35; global_balance:1 | 脖子和头都很长，长度相近。:3; 腿极长，头、脖子和尾巴相近，稍短一些。:1; 腿极短，头、脖子、尾巴都较长，长度相近。:1; 腿很长，头和脖子相近，也都比较长。:1; 腿很长，其余三个部位较长，长度相近。:1 |
| 302 | 13 | 12 | other_unparsed:7; global_balance:6; other_reference:1 | 腿长，其他部位都不是很长，比较匀称。:2; 脖子，脖子长，尾巴短。:1; 上半身很长，腿很短。:1; 脖子长，尾巴长，头短，腿中等，整体比较匀称。:1; 上半身比较长，腿相对短。:1 |
| 303 | 1 | 1 | other_unparsed:1 | 头，脖子、腿都短，尾巴也短、比其他部位长一点。:1 |
| 304 | 3 | 2 | other_unparsed:2; other_reference:1; body_geometry:1 | 头和腿长度相同。:2; 头和躯干长度相同，其他部位长度是躯干的0.7倍。:1 |
| 305 | 33 | 11 | body_geometry:28; vague_size:28; other_unparsed:4; meta_or_uncertain:1 | 挺高大。:8; 身材高大。:7; 很高大。:6; 挺高大，脖子短。:5; 很高。:1 |
| 306 | 37 | 37 | other_unparsed:26; body_geometry:11 | 腿和尾巴一样长，较长，脖子非常长，头较短。:1; 腿和头差不多一样长，长度较长，尾巴和脖子相对来说较短。:1; 脖子比头长，和腿差不多，比腿稍长，尾巴最短。:1; 较躯干来说，腿较短，其他部位均较长，脖子比头长。:1; 较躯干来说，腿适中，其他部位较长一些。:1 |
| 307 | 9 | 9 | other_unparsed:8; vague_size:1 | 腿、脖子和尾巴，四个部位都短，头较长。:1; 头和脖子差不多长，都是较长，尾巴中等，腿较长。:1; 头和脖子差不多长，它们都是中等，腿较长，尾巴较长。:1; 脖子是头的两倍。:1; 头和脖子，腿较短，尾巴很短。:1 |
| 308 | 27 | 16 | other_unparsed:27 | 脖子不是最长。:12; 腿与尾巴，最短。:1; 脖子与尾巴长度相似，且最长。:1; 脖子与腿长度相似，且长度长于头。:1; 头与尾巴长度相似，比脖子长。:1 |
| 309 | 10 | 8 | global_balance:6; other_unparsed:2; other_reference:1; meta_or_uncertain:1 | 整体都较为均匀。:2; 比较均衡。:2; 尾巴非常短，其他部位正常。:1; 腿、头、尾巴都比脖子长。:1; 尾巴短，小小。:1 |
| 310 | 37 | 36 | other_unparsed:15; body_geometry:12; global_balance:11; other_reference:3; ordinal_or_secondary:1 | 头短，躯干长。:2; 头最长，腿最短，相差较大。:1; 脖子长，头和尾巴，中间腿最短。:1; 头很长，脖子和腿，长度差不多。:1; 腿最长，头、尾巴、脖子较短，且长度差不多。:1 |
| 311 | 44 | 37 | other_unparsed:43; ordinal_or_secondary:1 | 腿最长，头、尾巴和脖子。:2; 脖子最长，尾巴、头和腿。:2; 腿最长，尾巴、头和脖子。:2; 脖子最长，腿、头和尾巴。:2; 头最长，尾巴、脖子和腿。:2 |
| 312 | 9 | 9 | other_unparsed:7; meta_or_uncertain:1; global_balance:1 | 一样长。:1; 选错了。:1; 差不多长。:1; 都略短，头比尾巴长。:1; 头、腿、尾巴，很短。:1 |
| 313 | 7 | 7 | other_unparsed:6; body_geometry:1 | 尾巴，巴长，腿短，头短。:1; 头很长，尾巴、腿、脖子差不多长，都比头稍微短一点。:1; 腿长，尾巴长，头和脖子差不多长，头和脖子都比尾巴和腿短。:1; 腿短，尾巴长，头较短，和脖子相比。:1; 尾巴短，腿短，脖子比头短，都比尾巴和腿长。:1 |
| 314 | 23 | 12 | other_unparsed:19; body_geometry:4; count_abstract:3 | 躯干长于脖子，且短于腿。:6; 朝向向左。:4; 有一个部位比躯干长。:3; 躯干短于脖子，且长于尾巴。:2; 脖子长于头、长于躯干，长于尾巴、长于腿。:1 |
| 315 | 1 | 1 | other_unparsed:1 | 脖。:1 |
| 316 | 52 | 44 | other_unparsed:50; other_reference:2 | 腿短，且是最短，头比脖子长。:4; 腿短，且是最短，头比脖子短。:3; 腿长，尾巴明显比腿短。:2; 腿短，且脖子不是头、脖子、尾巴里最短。:2; 腿短，尾巴不比脖子长。:2 |
| 317 | 62 | 34 | extreme_endpoint:31; count_abstract:26; other_unparsed:21; body_geometry:5 | 有部位达到最长或最短长度。:6; 腿不是最短的部位。:5; 脖子不是最长的部位。:5; 腿没有达到最大长度。:4; 没有部位达到最长或最短长度。:4 |
| 318 | 17 | 17 | other_unparsed:12; body_geometry:5; global_balance:1 | 脖子最短，尾巴比腿短，比脖子长，头也比较长。:1; 腿最短，脖子和尾巴一样长，比腿长，头最长。:1; 尾巴比腿长，头和脖子，脖子更长。:1; 尾巴比腿长，躯干差距很大。:1; 尾巴比腿长，差距不大，脖子和头都很长。:1 |
| 319 | 73 | 29 | proportion_or_ratio:43; other_unparsed:10; global_balance:7; meta_or_uncertain:6; count_abstract:5 | 头和脖子的比例大于腿和尾巴。:26; 头和脖子的比例小于腿和尾巴。:10; 选错了。:6; 头和脖子都比腿长。:3; 四个部位长度比较平衡。:2 |
| 321 | 20 | 19 | other_unparsed:20 | 腿不是最短的，头比脖子长。:2; 腿较，尾巴和脖子都长，头也长。:1; 腿超级短，头、脖子和尾巴都比腿长。:1; 头比脖子短，腿略微比尾巴短。:1; 头比脖子短，尾巴也比腿短。:1 |
| 322 | 56 | 36 | other_unparsed:48; body_geometry:5; disjoint_inequality:3 | 头不是最长。:21; 脖子和腿，头比脖子和腿长，尾巴很短。:1; 脖子和头非常长，腿略短于二者，尾巴最短。:1; 脖子和尾巴一样长，头略短于二者，腿最短。:1; 脖子和腿差不多长，头和尾巴差不多长，且长于前二者。:1 |
| 323 | 3 | 1 | meta_or_uncertain:3 | 选错了。:3 |
| 324 | 4 | 4 | other_unparsed:4 | 腿较短，前已经较长。:1; 腿和脖子较长，头和尾巴长中等，较短。:1; 腿和头较长，脖子长度中，脖子长度中等，尾巴较短。:1; 腿较长，下巴为中等。:1 |
| 325 | 4 | 2 | other_unparsed:4 | 每个都长。:3; 每一个都很长。:1 |
| 326 | 51 | 40 | body_geometry:24; other_unparsed:17; meta_or_uncertain:8; global_balance:3; ordinal_or_secondary:1 | 假设脖子无关。:6; 朝右。:4; 假设尾巴无关。:2; 腿和躯干差不多，头比脖子长。:2; 腿比尾巴短，头在躯干以下。:2 |
| 327 | 1 | 1 | other_unparsed:1 | 头短，尾巴短，脖子长，腿中上。:1 |
| 328 | 6 | 6 | other_unparsed:5; global_balance:1 | 尾巴特别短，尾巴明显比腿短，头和脖子加在一起比较长。:1; 腿和尾巴差不多长，都是中等长度，头和脖子都比较长。:1; 尾巴和腿都比较长，差不多，头加脖子比较短。:1; 尾巴长，头和脖子都很短，都比腿短。:1; 尾巴短，头和脖子都比腿长。:1 |
| 329 | 4 | 3 | other_unparsed:3; ordinal_or_secondary:1 | 按错了。:2; 全身都长。:1; 尾巴长，其次是腿，然后是头和脖子。:1 |
| 331 | 3 | 3 | other_unparsed:1; body_geometry:1; global_balance:1 | 头长，尾巴长，腿也还行。:1; 躯干长，头短。:1; 均等。:1 |

### 高频未编码文本 Top 80

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短于中间值。 | 62 | 头短于中间值 | other_unparsed | S3T14, S3T17, S3T20, S3T21, S3T22, S3T23, S3T26, S3T27 |
| 选错了。 | 61 | 选错了 | meta_or_uncertain | S1T225, S2T69, S1T308, S1T209, S2T112, S5T316, S1T97, S1T131 |
| 头长于中间值。 | 53 | 头长于中间值 | other_unparsed | S3T15, S3T16, S3T18, S3T19, S3T24, S3T25, S3T33, S3T36 |
| 头不是最长。 | 35 | 头不是最长 | other_unparsed | S2T46, S2T47, S2T48, S2T164, S2T165, S2T166, S2T168, S2T169 |
| 头和脖子的比例大于腿和尾巴。 | 26 | 头和脖子的比例大于腿和尾巴 | proportion_or_ratio | S1T88, S1T100, S1T103, S1T108, S1T109, S1T111, S1T113, S1T115 |
| 比较均衡。 | 21 | 比较均衡 | global_balance | S1T63, S1T69, S1T75, S1T76, S1T78, S1T160, S1T189, S1T256 |
| 两长两短。 | 19 | 两长两短 | count_abstract | S1T275, S1T297, S3T220, S3T221, S3T222, S2T173, S2T174, S1T46 |
| 三个部位很长。 | 17 | 三个部位很长 | count_abstract | S1T123, S1T124, S1T125, S1T126, S1T134, S1T135, S1T136, S1T137 |
| 四个部位较均等。 | 17 | 四个部位较均等 | global_balance | S1T61, S1T65, S1T68, S1T72, S1T93, S1T119, S1T124, S1T126 |
| 腿和尾巴都比脖子长。 | 16 | 腿和尾巴都比脖子长 | other_unparsed | S1T180, S1T244, S1T245, S1T248, S1T250, S1T252, S1T253, S1T256 |
| 一个部位很长。 | 15 | 一个部位很长 | count_abstract | S1T147, S1T148, S1T150, S1T153, S1T155, S1T32, S1T33, S1T36 |
| 三长一短。 | 15 | 三长一短 | count_abstract | S2T175, S2T238, S2T240, S1T26, S1T32, S1T45, S1T50, S1T53 |
| 有一个部位长于躯干。 | 13 | 有一个部位长于躯干 | count_abstract, body_geometry | S2T281, S2T284, S2T293, S2T295, S2T296, S2T298, S2T299, S2T301 |
| 有三个部位几乎一样长。 | 13 | 有三个部位几乎一样长 | count_abstract | S1T29, S1T106, S1T108, S1T142, S1T193, S1T257, S1T260, S1T262 |
| 比较均匀。 | 12 | 比较均匀 | global_balance | S1T139, S1T146, S2T79, S2T82, S2T88, S2T96, S2T98, S2T102 |
| 脖子不是最长。 | 12 | 脖子不是最长 | other_unparsed | S2T235, S2T242, S2T243, S2T244, S2T257, S2T258, S2T273, S2T280 |
| 三个部位几乎一样长。 | 11 | 三个部位几乎一样长 | count_abstract | S1T170, S1T190, S1T223, S1T224, S1T241, S1T290, S1T310, S1T314 |
| 两个部位长，两个部位短。 | 10 | 两个部位长; 两个部位短 | count_abstract | S1T109, S1T110, S1T112, S2T52, S2T62, S2T63, S2T87, S2T225 |
| 头和脖子的比例小于腿和尾巴。 | 10 | 头和脖子的比例小于腿和尾巴 | proportion_or_ratio | S1T98, S1T101, S1T102, S1T104, S1T112, S1T118, S1T122, S1T123 |
| 头比脖子短，比尾巴短。 | 10 | 比尾巴短 | other_unparsed | S4T138, S4T139, S4T145, S4T146, S4T151, S4T152, S4T154, S4T155 |
| 头比脖子长，比尾巴长。 | 10 | 比尾巴长 | other_unparsed | S4T141, S4T144, S4T149, S4T150, S4T153, S4T168, S4T195, S1T155 |
| 头在躯干之下。 | 9 | 头在躯干之下 | body_geometry | S1T108, S1T111, S1T114, S1T116, S1T117, S1T118, S1T119, S1T121 |
| 有两个部位一样长。 | 9 | 有两个部位一样长 | count_abstract | S1T15, S1T59, S1T60, S1T61, S1T62, S1T64, S1T66, S1T40 |
| 有两个部位几乎一样长。 | 9 | 有两个部位几乎一样长 | count_abstract | S1T30, S1T38, S1T135, S1T138, S1T141, S1T144, S1T167, S1T259 |
| 三个部位长。 | 8 | 三个部位长 | count_abstract | S1T121, S1T122, S2T35, S2T50, S2T184, S2T188, S2T189, S2T194 |
| 尾巴不是最短的。 | 8 | 尾巴不是最短的 | other_unparsed | S1T118, S1T144, S1T145, S1T146, S1T147, S1T149, S1T151, S1T154 |
| 挺高大。 | 8 | 挺高大 | body_geometry, vague_size | S1T50, S1T51, S1T52, S1T54, S1T55, S1T56, S1T59, S1T61 |
| 脖子不是最长的。 | 8 | 脖子不是最长的 | other_unparsed | S2T34, S2T35, S2T36, S2T37, S2T38, S2T39, S2T43, S2T44 |
| 三个部位长，一个部位短。 | 7 | 三个部位长; 一个部位短 | count_abstract | S1T111, S2T51, S2T57, S2T58, S2T64, S2T231, S2T236 |
| 只有一个部位很长。 | 7 | 只有一个部位很长 | count_abstract | S1T118, S1T129, S1T132, S1T133, S1T140, S1T141, S1T142 |
| 四个部位长度各不相同。 | 7 | 四个部位长度各不相同 | disjoint_inequality | S1T97, S1T98, S1T100, S1T101, S1T103, S1T132, S1T133 |
| 头比脖子长，比尾巴短。 | 7 | 比尾巴短 | other_unparsed | S4T93, S4T140, S4T142, S4T147, S4T148, S4T163, S4T166 |
| 尾巴和脖子不一样长。 | 7 | 尾巴和脖子不一样长 | disjoint_inequality | S1T26, S1T27, S1T54, S1T55, S1T57, S1T58, S1T155 |
| 尾巴比脖子长，比头长。 | 7 | 比头长 | other_unparsed | S3T258, S3T261, S3T271, S3T272, S4T88, S4T91, S4T92 |
| 有三个部位长于躯干。 | 7 | 有三个部位长于躯干 | count_abstract, body_geometry | S2T282, S2T283, S2T286, S2T288, S2T292, S2T300, S2T306 |
| 有三个部位长度一样。 | 7 | 有三个部位长度一样 | count_abstract | S3T189, S3T190, S3T207, S3T210, S3T216, S3T236, S4T216 |
| 有两个部位比较长。 | 7 | 有两个部位比较长 | count_abstract | S2T306, S2T309, S2T310, S2T312, S2T314, S2T317, S2T318 |
| 有两个部位长于躯干。 | 7 | 有两个部位长于躯干 | count_abstract, body_geometry | S2T280, S2T285, S2T290, S2T291, S2T303, S2T307, S2T310 |
| 腿不是最短的部位。 | 7 | 腿不是最短的部位 | other_unparsed | S2T56, S2T57, S2T185, S2T186, S2T187, S2T189, S2T190 |
| 身材高大。 | 7 | 身材高大 | body_geometry, vague_size | S1T70, S1T80, S1T82, S1T89, S1T92, S1T94, S1T96 |
| 躯体下方比上面高。 | 7 | 躯体下方比上面高 | body_geometry | S1T179, S1T181, S1T183, S1T184, S1T185, S1T186, S1T187 |
| 两个部位很长。 | 6 | 两个部位很长 | count_abstract | S1T119, S1T120, S1T127, S1T128, S1T145, S1T146 |
| 假设脖子无关。 | 6 | 假设脖子无关 | meta_or_uncertain | S1T243, S1T244, S1T245, S1T246, S1T247, S1T248 |
| 四个部位较匀称。 | 6 | 四个部位较匀称 | global_balance | S1T182, S1T242, S1T246, S1T255, S1T259, S2T24 |
| 尾巴比脖子短，比头短。 | 6 | 比头短 | other_unparsed | S3T257, S3T262, S3T265, S4T87, S4T89, S4T90 |
| 很高大。 | 6 | 很高大 | body_geometry, vague_size | S1T1, S1T12, S1T23, S1T24, S1T31, S1T46 |
| 有三个部位一样长。 | 6 | 有三个部位一样长 | count_abstract | S1T64, S1T210, S2T199, S3T23, S3T51, S3T108 |
| 有部位达到最长或最短长度。 | 6 | 有部位达到最长或最短长度 | count_abstract, extreme_endpoint | S2T109, S2T110, S2T113, S2T316, S2T317, S2T319 |
| 脖子比头短，比尾巴短。 | 6 | 比尾巴短 | other_unparsed | S4T96, S4T97, S4T98, S4T101, S4T105, S4T110 |
| 躯干长于脖子，且短于腿。 | 6 | 且短于腿 | other_unparsed | S3T137, S3T139, S3T141, S3T142, S3T145, S3T146 |
| 两个部位一样长。 | 5 | 两个部位一样长 | count_abstract | S1T86, S1T90, S1T93, S1T179, S1T180 |
| 两个部位几乎一样长。 | 5 | 两个部位几乎一样长 | count_abstract | S1T164, S2T9, S2T76, S2T112, S2T116 |
| 均匀。 | 5 | 均匀 | global_balance | S2T104, S2T107, S2T110, S2T112, S2T116 |
| 头身比例比较协调。 | 5 | 头身比例比较协调 | proportion_or_ratio | S1T30, S1T31, S1T32, S1T33, S1T36 |
| 尾巴和腿都比脖子短。 | 5 | 尾巴和腿都比脖子短 | other_unparsed | S1T280, S1T281, S1T282, S1T284, S1T285 |
| 挺高大，脖子短。 | 5 | 挺高大 | body_geometry, vague_size | S1T37, S1T40, S1T42, S1T45, S1T47 |
| 有一个部位比躯干长。 | 5 | 有一个部位比躯干长 | count_abstract, body_geometry | S1T130, S1T131, S1T43, S1T44, S1T45 |
| 脖子不是最长的部位。 | 5 | 脖子不是最长的部位 | other_unparsed | S2T176, S2T177, S2T178, S2T179, S2T181 |
| 脖子比头长，比尾巴长。 | 5 | 比尾巴长 | other_unparsed | S4T94, S4T99, S4T103, S4T106, S4T109 |
| 腿不是最短。 | 5 | 腿不是最短 | other_unparsed | S2T157, S2T158, S2T159, S2T161, S2T162 |
| 腿比脖子短，比尾巴短。 | 5 | 比尾巴短 | other_unparsed | S4T170, S4T173, S4T174, S4T176, S2T35 |
| 都差不多长。 | 5 | 都差不多长 | global_balance | S1T237, S2T73, S2T242, S2T312, S2T214 |
| 三个差不多长。 | 4 | 三个差不多长 | global_balance | S1T67, S1T72, S1T78, S1T151 |
| 三个部位短，一个部位长。 | 4 | 三个部位短; 一个部位长 | count_abstract | S2T59, S2T86, S2T237, S1T74 |
| 三个部位长度一样。 | 4 | 三个部位长度一样 | count_abstract | S3T82, S3T83, S3T120, S3T122 |
| 三短一长。 | 4 | 三短一长 | other_unparsed | S2T239, S1T44, S1T59, S1T262 |
| 两个部位比中间值长。 | 4 | 两个部位比中间值长 | count_abstract | S2T149, S2T150, S2T151, S2T152 |
| 低头。 | 4 | 低头 | other_unparsed | S1T94, S1T95, S1T97, S1T98 |
| 体型分布的不均匀。 | 4 | 体型分布的不均匀 | global_balance, vague_size | S1T70, S1T71, S1T79, S1T81 |
| 四个部位都很长，体型很大。 | 4 | 体型很大 | vague_size | S1T157, S1T184, S1T195, S1T210 |
| 头在躯干之上。 | 4 | 头在躯干之上 | body_geometry | S1T110, S1T112, S1T113, S1T126 |
| 头很大。 | 4 | 头很大 | other_unparsed | S2T125, S2T155, S2T162, S2T243 |
| 头比脖子短，比尾巴长。 | 4 | 比尾巴长 | other_unparsed | S4T137, S4T161, S4T167, S4T194 |
| 头身比例不协调。 | 4 | 头身比例不协调 | proportion_or_ratio | S1T29, S1T34, S1T35, S1T37 |
| 尾巴长，头的位置高。 | 4 | 头的位置高 | other_unparsed | S6T75, S6T76, S6T77, S6T81 |
| 差不多长。 | 4 | 差不多长 | global_balance | S3T237, S1T77, S2T117, S2T254 |
| 有三个部位比较长。 | 4 | 有三个部位比较长 | count_abstract | S2T96, S2T305, S2T311, S2T316 |
| 有奇数个部位长于躯干。 | 4 | 有奇数个部位长于躯干 | count_abstract, body_geometry | S1T153, S1T154, S1T155, S1T159 |
| 朝右。 | 4 | 朝右 | other_unparsed | S1T81, S1T82, S1T83, S1T84 |
| 朝向向左。 | 4 | 朝向向左 | other_unparsed | S1T37, S1T39, S1T40, S1T42 |

### 逐被试未编码文本

#### S102

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子比头长，但是是最短的两个。 | 4 | 但是是最短的两个 | other_unparsed | S1T149, S1T150, S1T160, S1T172 |
| 脖子比头短，但是是最长的两个。 | 3 | 但是是最长的两个 | other_unparsed | S1T178, S1T179, S1T197 |
| 脖子比头短，它们两个是最长的。 | 3 | 它们两个是最长的 | other_unparsed | S1T204, S1T205, S1T206 |
| 脖子比头短，都短于腿。 | 3 | 都短于腿 | other_unparsed | S1T310, S1T311, S1T313 |
| 脖子比头短，都长于腿。 | 2 | 都长于腿 | other_unparsed | S1T307, S1T308 |
| 脖子短于头，但它们是最小的。 | 2 | 但它们是最小的 | other_unparsed | S1T231, S1T244 |
| 三长一中。 | 1 | 三长一中 | count_abstract | S1T286 |
| 两长两中。 | 1 | 两长两中 | other_unparsed | S1T279 |
| 两长两短。 | 1 | 两长两短 | count_abstract | S1T275 |
| 脖子小于头，但它们是最大的。 | 1 | 但它们是最大的 | other_unparsed | S1T251 |
| 脖子微短于头，腿适中，尾巴短。 | 1 | 脖子微短于头 | other_unparsed | S1T109 |
| 脖子比头短，但是是最短的两个。 | 1 | 但是是最短的两个 | other_unparsed | S1T176 |
| 脖子比头短，但是都小于腿。 | 1 | 但是都小于腿 | other_unparsed | S1T301 |
| 脖子比头短，但是都长于腿。 | 1 | 但是都长于腿 | other_unparsed | S1T299 |
| 脖子比头短，但都比尾巴短。 | 1 | 但都比尾巴短 | other_unparsed | S1T182 |
| 脖子比头短，而且不是最长的两个。 | 1 | 而且不是最长的两个 | other_unparsed | S1T184 |
| 脖子比头短，而且是最长的两个。 | 1 | 而且是最长的两个 | other_unparsed | S1T186 |
| 脖子比头短，腿和尾巴差不多，都适中。 | 1 | 都适中 | other_unparsed | S1T138 |
| 脖子比头短，都大于腿。 | 1 | 都大于腿 | other_unparsed | S1T303 |
| 脖子比头长，不确定是不是最短的两个。 | 1 | 不确定是不是最短的两个 | meta_or_uncertain | S1T157 |
| 脖子比头长，但是它们两个是最长的。 | 1 | 但是它们两个是最长的 | other_unparsed | S1T201 |
| 脖子比头长，但是是最长的两个。 | 1 | 但是是最长的两个 | other_unparsed | S1T174 |
| 脖子比头长，但是都长于腿。 | 1 | 但是都长于腿 | other_unparsed | S1T298 |
| 脖子比头长，但都比较短。 | 1 | 但都比较短 | other_unparsed | S1T199 |
| 脖子比头长，都大于腿。 | 1 | 都大于腿 | other_unparsed | S1T309 |
| 脖子比头长，都短于腿。 | 1 | 都短于腿 | other_unparsed | S1T305 |
| 脖子比头长，都长于腿。 | 1 | 都长于腿 | other_unparsed | S1T306 |
| 脖子略微短于头，腿适中，尾巴适中。 | 1 | 脖子略微短于头 | other_unparsed | S1T84 |
| 脖子短于头，但它们是最小的两个。 | 1 | 但它们是最小的两个 | other_unparsed | S1T222 |
| 脖子短于头，但它们是最长的两个。 | 1 | 但它们是最长的两个 | other_unparsed | S1T214 |
| 脖子短于头，但是两个都很长。 | 1 | 但是两个都很长 | other_unparsed | S1T237 |
| 脖子短于头，但是它们的平均长度长于尾巴。 | 1 | 但是它们的平均长度长于尾巴 | global_balance | S1T209 |
| 脖子短于头，腿和尾巴都比它们短。 | 1 | 腿和尾巴都比它们短 | other_unparsed | S1T91 |
| 脖子长于头，但它们是最大的两个。 | 1 | 但它们是最大的两个 | other_unparsed | S1T219 |
| 脖子长于头，但它们是最小的两个。 | 1 | 但它们是最小的两个 | other_unparsed | S1T218 |
| 脖子长于头，但它们是最短的两个。 | 1 | 但它们是最短的两个 | other_unparsed | S1T211 |
| 腿长，尾巴适中，脖子略微长于头。 | 1 | 脖子略微长于头 | other_unparsed | S1T78 |
| 腿长，尾巴长，脖子略微长于头。 | 1 | 脖子略微长于头 | other_unparsed | S1T79 |

#### S103

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、尾巴都中等，略长一点，腿较短。 | 1 | 略长一点 | other_unparsed | S1T95 |
| 尾巴很长，长于头和脖子，且腿中等偏短。 | 1 | 长于头和脖子 | other_unparsed | S1T143 |
| 尾巴很长，长于头和脖子，头、脖子、腿都很短。 | 1 | 长于头和脖子 | other_unparsed | S1T35 |
| 尾巴很长，长于头和脖子，腿较短。 | 1 | 长于头和脖子 | other_unparsed | S1T4 |
| 尾巴较长，长于脖子和头。 | 1 | 长于脖子和头 | other_unparsed | S1T185 |
| 脖子和尾巴较长，长于头和腿。 | 1 | 长于头和腿 | other_unparsed | S1T202 |
| 腿中等，头较长，尾巴较长，明显长于脖子。 | 1 | 明显长于脖子 | other_unparsed | S1T44 |
| 腿很短，脖子很长，长于尾巴，尾巴和头中等长度。 | 1 | 长于尾巴 | other_unparsed | S1T33 |
| 腿很长，脖子和头中等长度，长于较短的尾巴。 | 1 | 长于较短的尾巴 | other_unparsed | S1T13 |
| 腿极长，尾巴中等长度，长于头和脖子。 | 1 | 长于头和脖子 | other_unparsed | S1T5 |
| 腿较短，尾巴较长，长于头和脖子。 | 1 | 长于头和脖子 | other_unparsed | S1T21 |
| 腿较长，脖子中等较长，长于尾巴。 | 1 | 长于尾巴 | other_unparsed | S1T52 |

#### S104

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头长，较均匀。 | 1 | 较均匀 | global_balance | S1T109 |
| 比较均匀。 | 1 | 比较均匀 | global_balance | S1T139 |

#### S105

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短于阈值。 | 1 | 腿短于阈值 | other_unparsed | S1T17 |

#### S106

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 躯体下方比上面高。 | 7 | 躯体下方比上面高 | body_geometry | S1T179, S1T181, S1T183, S1T184, S1T185, S1T186, S1T187 |
| 下面的部分更高一些。 | 2 | 下面的部分更高一些 | body_geometry | S1T174, S1T175 |
| 头离地面近。 | 2 | 头离地面近 | other_unparsed | S1T32, S1T33 |
| 腿不是最长的。 | 2 | 腿不是最长的 | other_unparsed | S1T55, S1T58 |
| 躯体下方没有上面高。 | 2 | 躯体下方没有上面高 | body_geometry | S1T180, S1T182 |
| 上面的高度大于下面。 | 1 | 上面的高度大于下面 | body_geometry | S1T173 |
| 上面的高度小于下面。 | 1 | 上面的高度小于下面 | body_geometry | S1T172 |
| 下半身腿比上面的高。 | 1 | 下半身腿比上面的高 | body_geometry | S1T170 |
| 下面的部分高于上面。 | 1 | 下面的部分高于上面 | body_geometry | S1T176 |
| 下面的高度比上面高。 | 1 | 下面的高度比上面高 | body_geometry | S1T178 |
| 下面的高度高于上面。 | 1 | 下面的高度高于上面 | body_geometry | S1T177 |
| 头离地面远。 | 1 | 头离地面远 | other_unparsed | S1T34 |
| 头距离地面比较近。 | 1 | 头距离地面比较近 | other_unparsed | S1T35 |
| 有两个部位一样长。 | 1 | 有两个部位一样长 | count_abstract | S1T15 |
| 点错了。 | 1 | 点错了 | other_unparsed | S1T105 |
| 腿离地面距离不如躯体最高点的距离高。 | 1 | 腿离地面距离不如躯体最高点的距离高 | body_geometry | S1T171 |
| 躯体下方比上面高一些。 | 1 | 躯体下方比上面高一些 | body_geometry | S1T190 |
| 躯体下方没有上面高，腿太短。 | 1 | 躯体下方没有上面高 | body_geometry | S1T188 |

#### S107

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴很长，脖子也挺长，但相对短一些，腿很短。 | 1 | 但相对短一些 | other_unparsed | S1T45 |
| 尾巴、腿、脖子和头长度差不多，都还挺长的。 | 1 | 都还挺长的 | other_unparsed | S1T164 |
| 腿、脖子和头不算特别长，且长度相当。 | 1 | 且长度相当 | global_balance | S1T191 |
| 腿、脖子和头比例比较协调，尾巴短。 | 1 | 腿、脖子和头比例比较协调 | proportion_or_ratio | S1T95 |
| 腿、脖子和头长度差不多，都不算很长。 | 1 | 都不算很长 | other_unparsed | S1T111 |

#### S108

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子不是最长的。 | 8 | 脖子不是最长的 | other_unparsed | S2T34, S2T35, S2T36, S2T37, S2T38, S2T39, S2T43, S2T44 |
| 腿比脖子短，比尾巴长。 | 3 | 比尾巴长 | other_unparsed | S1T26, S1T42, S1T164 |
| 腿比脖子和尾巴短，比头长。 | 2 | 比头长 | other_unparsed | S1T53, S1T54 |
| 选错了。 | 2 | 选错了 | meta_or_uncertain | S1T225, S2T69 |
| 头和脖子显著比腿和尾巴长。 | 1 | 头和脖子显著比腿和尾巴长 | other_unparsed | S1T14 |
| 脖子不是最长的，腿是最长的。 | 1 | 脖子不是最长的 | other_unparsed | S2T48 |
| 脖子和尾巴显著比腿长。 | 1 | 脖子和尾巴显著比腿长 | other_unparsed | S1T8 |
| 腿比尾巴，腿比脖子短。 | 1 | 腿比尾巴 | other_unparsed | S1T166 |
| 腿比脖子短，比头和尾巴长。 | 1 | 比头和尾巴长 | other_unparsed | S1T29 |
| 腿比脖子长，比尾巴短。 | 1 | 比尾巴短 | other_unparsed | S1T31 |

#### S109

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子，头较长，腿一般。 | 1 | 脖子 | other_unparsed | S1T211 |
| 腿较长，脖子一般，头较小。 | 1 | 头较小 | other_unparsed | S1T213 |

#### S110

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短，脖子腿，尾巴长。 | 1 | 脖子腿 | other_unparsed | S1T169 |

#### S111

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短，尾巴、头、脖子长度相同。 | 1 | 尾巴、头、脖子长度相同 | other_unparsed | S1T18 |

#### S112

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴和头之间有交集。 | 1 | 尾巴和头之间有交集 | other_unparsed | S1T136 |
| 尾巴和腿之间关系。 | 1 | 尾巴和腿之间关系 | other_unparsed | S1T29 |
| 尾巴比头长一点，比腿和脖子都短。 | 1 | 比腿和脖子都短 | other_unparsed | S1T34 |
| 尾巴跟脖子不一样长。 | 1 | 尾巴跟脖子不一样长 | disjoint_inequality | S1T39 |
| 尾巴跟腿之间的关系以及跟脖子的关系。 | 1 | 尾巴跟腿之间的关系以及跟脖子的关系 | other_unparsed | S1T37 |
| 脖子和腿之间的关系。 | 1 | 脖子和腿之间的关系 | other_unparsed | S1T47 |

#### S113

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子长，比头长。 | 3 | 比头长 | other_unparsed | S1T293, S1T294, S1T318 |
| 头比脖子长，都很长。 | 2 | 都很长 | other_unparsed | S1T135, S1T137 |
| 不知道。 | 1 | 不知道 | meta_or_uncertain | S1T218 |
| 头和尾巴很长，脖子短一些，腿第二短，但也比较长。 | 1 | 但也比较长 | other_unparsed | S1T24 |
| 头和脖子差不多长，都很长，尾巴和腿比较短。 | 1 | 都很长 | other_unparsed | S1T62 |
| 头和脖子差不多长，都比较长，腿和尾巴很短。 | 1 | 都比较长 | other_unparsed | S1T80 |
| 头和腿一样长，都比较长，脖子和尾巴比较短。 | 1 | 都比较长 | other_unparsed | S1T4 |
| 头最长，尾巴最短，脖子和腿差不多长，都比较长。 | 1 | 都比较长 | other_unparsed | S1T13 |
| 头比脖子短，都有些短。 | 1 | 都有些短 | other_unparsed | S1T136 |
| 头比脖子长，尾巴和腿适中，比较长。 | 1 | 比较长 | other_unparsed | S1T66 |
| 头比脖子长，尾巴和腿适中，短一些。 | 1 | 短一些 | other_unparsed | S1T87 |
| 尾巴和腿很短，头比适中，脖子有些长。 | 1 | 头比适中 | other_unparsed | S1T30 |
| 尾巴最长，头最短，脖子和腿适中，比较长。 | 1 | 比较长 | other_unparsed | S1T38 |
| 尾巴比头长，都比较长，脖子和腿比较短，腿比脖子短。 | 1 | 都比较长 | other_unparsed | S1T3 |
| 尾巴比脖子长，但是都很短，脖子和腿很长，腿特别长。 | 1 | 但是都很短 | other_unparsed | S1T20 |
| 点错了。 | 1 | 点错了 | other_unparsed | S2T32 |
| 脖子中等长度，比头短。 | 1 | 比头短 | other_unparsed | S1T307 |
| 脖子明显比头短。 | 1 | 脖子明显比头短 | other_unparsed | S1T252 |
| 脖子比头长，头比腿长，腿比尾巴长，都有些长。 | 1 | 都有些长 | other_unparsed | S1T1 |
| 脖子短，但是和头的长度差不多。 | 1 | 但是和头的长度差不多 | global_balance | S1T262 |
| 脖子短，比头短。 | 1 | 比头短 | other_unparsed | S1T304 |
| 脖子长，比头短。 | 1 | 比头短 | other_unparsed | S1T319 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S1T308 |

#### S114

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 都差不多长。 | 2 | 都差不多长 | global_balance | S1T237, S2T73 |
| 都不长。 | 1 | 都不长 | other_unparsed | S1T96 |

#### S116

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 嗯，脖子比较长。 | 1 | 嗯 | other_unparsed | S1T48 |
| 四个部位都比较长，其中腿最明显。 | 1 | 其中腿最明显 | other_unparsed | S1T24 |
| 脖子的长度很突出。 | 1 | 脖子的长度很突出 | other_unparsed | S1T28 |
| 腿和脖子都比较长，但腿是最明显的。 | 1 | 但腿是最明显的 | other_unparsed | S1T36 |
| 腿的长度最突出。 | 1 | 腿的长度最突出 | other_unparsed | S1T27 |

#### S118

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 三个部位很长。 | 17 | 三个部位很长 | count_abstract | S1T123, S1T124, S1T125, S1T126, S1T134, S1T135, S1T136, S1T137 |
| 只有一个部位很长。 | 7 | 只有一个部位很长 | count_abstract | S1T118, S1T129, S1T132, S1T133, S1T140, S1T141, S1T142 |
| 两个部位很长。 | 6 | 两个部位很长 | count_abstract | S1T119, S1T120, S1T127, S1T128, S1T145, S1T146 |
| 一个部位很长。 | 5 | 一个部位很长 | count_abstract | S1T147, S1T148, S1T150, S1T153, S1T155 |
| 三个部位都很长。 | 3 | 三个部位都很长 | count_abstract | S1T107, S1T117, S1T130 |
| 两个部位长，两个部位短。 | 3 | 两个部位长; 两个部位短 | count_abstract | S1T109, S1T110, S1T112 |
| 只有两个部位很长。 | 3 | 只有两个部位很长 | count_abstract | S1T131, S1T143, S1T159 |
| 三个部位长。 | 2 | 三个部位长 | count_abstract | S1T121, S1T122 |
| 三个部位都很短。 | 1 | 三个部位都很短 | count_abstract | S1T113 |
| 三个部位都很长，只有头是最短的。 | 1 | 三个部位都很长 | count_abstract | S1T108 |
| 三个部位长，一个部位短。 | 1 | 三个部位长; 一个部位短 | count_abstract | S1T111 |
| 两个部位很长，两个部位很短。 | 1 | 两个部位很长; 两个部位很短 | count_abstract | S1T114 |
| 头、脖子、尾巴、腿都很短，都差不多长度。 | 1 | 都差不多长度 | global_balance | S1T52 |
| 头和尾巴一样长，都很长，脖子稍微比头和尾巴短一点，尾巴比腿长，脖子和腿差不多长。 | 1 | 都很长; 脖子稍微比头和尾巴短一点 | other_unparsed | S1T13 |
| 头和脖子都是它们长度范围的1/2，头和脖子一样长，尾巴也非常长，腿也比较长，但没有达到最大长度。 | 1 | 但没有达到最大长度 | extreme_endpoint | S1T26 |
| 头是最长的，脖子第二长，大概在最长长度的1/2，腿也是在它最大长度的1/2，尾巴比较短。 | 1 | 大概在最长长度的1/2 | extreme_endpoint | S1T4 |
| 头短，脖子短，尾巴最长，腿很短，是中等长度以下。 | 1 | 是中等长度以下 | other_unparsed | S1T42 |
| 头短，脖子短，尾巴稍微长一点，是最长的部位，腿短。 | 1 | 是最长的部位 | other_unparsed | S1T40 |
| 头长，脖子长，腿长，尾巴长，都很长。 | 1 | 都很长 | other_unparsed | S1T87 |
| 尾巴、腿、脖子、头都挺长的，都差不多长。 | 1 | 都差不多长 | global_balance | S1T32 |
| 尾巴、腿都非常长，达到了它们的最长长度，脖子也达到了最长长度，头是最短的，但也很长，可能是它自身最长长度的1/2。 | 1 | 但也很长 | other_unparsed | S1T9 |
| 尾巴和腿都很短，尾巴在1/4到1/5之间，腿在1/4到1/5之间，但是脖子和头都比较长。 | 1 | 尾巴在1/4到1/5之间; 腿在1/4到1/5之间 | other_unparsed | S1T3 |
| 尾巴比较大。 | 1 | 尾巴比较大 | other_unparsed | S1T256 |
| 按错了。 | 1 | 按错了 | other_unparsed | S1T248 |
| 脖子最长，腿第二长，尾巴比较短，大概是它最长长度的1/3，头和尾巴差不多长。 | 1 | 大概是它最长长度的1/3 | extreme_endpoint | S1T11 |
| 腿非常短，是最短的部位，也是是它自身最短长度，尾巴是第二长的，头和脖子都比较长。 | 1 | 是最短的部位 | other_unparsed | S1T21 |
| 腿非常短，达到了最小长度，头比腿长，脖子也比腿长，尾巴也比腿长，尾巴达到了最长长度。 | 1 | 脖子也比腿长; 尾巴也比腿长 | other_unparsed | S1T8 |

#### S119

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短于中间值。 | 62 | 头短于中间值 | other_unparsed | S3T14, S3T17, S3T20, S3T21, S3T22, S3T23, S3T26, S3T27 |
| 头长于中间值。 | 53 | 头长于中间值 | other_unparsed | S3T15, S3T16, S3T18, S3T19, S3T24, S3T25, S3T33, S3T36 |
| 头不是最长。 | 14 | 头不是最长 | other_unparsed | S2T46, S2T47, S2T48, S2T164, S2T165, S2T166, S2T168, S2T169 |
| 有一个部位长于躯干。 | 13 | 有一个部位长于躯干 | count_abstract, body_geometry | S2T281, S2T284, S2T293, S2T295, S2T296, S2T298, S2T299, S2T301 |
| 有三个部位长于躯干。 | 7 | 有三个部位长于躯干 | count_abstract, body_geometry | S2T282, S2T283, S2T286, S2T288, S2T292, S2T300, S2T306 |
| 有两个部位长于躯干。 | 7 | 有两个部位长于躯干 | count_abstract, body_geometry | S2T280, S2T285, S2T290, S2T291, S2T303, S2T307, S2T310 |
| 腿不是最短。 | 5 | 腿不是最短 | other_unparsed | S2T157, S2T158, S2T159, S2T161, S2T162 |
| 两个部位比中间值长。 | 4 | 两个部位比中间值长 | count_abstract | S2T149, S2T150, S2T151, S2T152 |
| 腿长于中间值。 | 4 | 腿长于中间值 | other_unparsed | S2T156, S3T11, S3T12, S3T13 |
| 两个部位长于中间值。 | 3 | 两个部位长于中间值 | count_abstract | S2T100, S2T102, S2T103 |
| 有两个部位长于中间值。 | 3 | 有两个部位长于中间值 | count_abstract | S2T97, S2T98, S2T268 |
| 没有部位长于躯干。 | 3 | 没有部位长于躯干 | count_abstract, body_geometry | S2T287, S2T294, S2T297 |
| 一个部位比中间值长。 | 2 | 一个部位比中间值长 | count_abstract | S2T153, S2T154 |
| 一个部位长于中间值。 | 2 | 一个部位长于中间值 | count_abstract | S2T99, S2T101 |
| 脖子比中间值长。 | 2 | 脖子比中间值长 | other_unparsed | S2T145, S2T147 |
| 三个部位长于中间值。 | 1 | 三个部位长于中间值 | count_abstract | S2T215 |
| 只有一个部位长于躯干。 | 1 | 只有一个部位长于躯干 | count_abstract, body_geometry | S1T35 |
| 大部分长于躯干。 | 1 | 大部分长于躯干 | body_geometry | S2T183 |
| 头比中间值长。 | 1 | 头比中间值长 | other_unparsed | S2T146 |
| 头没有长于腿。 | 1 | 头没有长于腿 | other_unparsed | S2T67 |
| 少于两个部位长于中间值。 | 1 | 少于两个部位长于中间值 | count_abstract | S2T269 |
| 尾巴和脖子不是最短的。 | 1 | 尾巴和脖子不是最短的 | other_unparsed | S2T193 |
| 有一个部位长于中间值。 | 1 | 有一个部位长于中间值 | count_abstract | S2T267 |
| 有一个部位长长于躯干。 | 1 | 有一个部位长长于躯干 | count_abstract, body_geometry | S2T289 |
| 有三个部位比较长。 | 1 | 有三个部位比较长 | count_abstract | S2T96 |
| 脖子和腿长度差异比较大。 | 1 | 脖子和腿长度差异比较大 | other_unparsed | S1T16 |
| 脖子没有长于尾巴。 | 1 | 脖子没有长于尾巴 | other_unparsed | S2T69 |
| 脖子长于中间值。 | 1 | 脖子长于中间值 | other_unparsed | S2T272 |
| 腿不是最长。 | 1 | 腿不是最长 | other_unparsed | S2T111 |
| 腿和尾巴长于中间值。 | 1 | 腿和尾巴长于中间值 | other_unparsed | S2T273 |
| 腿比中间值长。 | 1 | 腿比中间值长 | other_unparsed | S2T148 |
| 腿短于中间值。 | 1 | 腿短于中间值 | other_unparsed | S3T10 |
| 超过两个等于躯干。 | 1 | 超过两个等于躯干 | body_geometry | S2T182 |
| 超过两个部位短于躯干。 | 1 | 超过两个部位短于躯干 | count_abstract, body_geometry | S2T180 |
| 超过两个部位长于躯干。 | 1 | 超过两个部位长于躯干 | count_abstract, body_geometry | S1T34 |
| 都短于躯干。 | 1 | 都短于躯干 | body_geometry | S2T181 |

#### S120

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 有奇数个部位长于躯干。 | 4 | 有奇数个部位长于躯干 | count_abstract, body_geometry | S1T153, S1T154, S1T155, S1T159 |
| 有偶数个部位长于躯干。 | 3 | 有偶数个部位长于躯干 | count_abstract, body_geometry | S1T156, S1T157, S1T158 |
| 头明显长于其他部位。 | 2 | 头明显长于其他部位 | other_reference | S1T28, S1T33 |
| 脖子和尾巴的长度不一样。 | 2 | 脖子和尾巴的长度不一样 | disjoint_inequality | S1T42, S1T43 |
| 脖子明显短于其他部位。 | 2 | 脖子明显短于其他部位 | other_reference | S1T109, S1T186 |
| 腿明显短于其他部位。 | 2 | 腿明显短于其他部位 | other_reference | S1T35, S1T36 |
| 五个部位的长度都差不多。 | 1 | 五个部位的长度都差不多 | global_balance | S1T25 |
| 头和尾巴加起来短于脖子和躯干，也短于脖子和腿。 | 1 | 也短于脖子和腿 | other_unparsed | S1T58 |
| 头和尾巴长度大于脖子和腿的长度。 | 1 | 头和尾巴长度大于脖子和腿的长度 | other_unparsed | S1T63 |
| 头明显长于其他四个部位。 | 1 | 头明显长于其他四个部位 | other_reference | S1T26 |
| 头比较长的，长于尾巴，脖子长于腿。 | 1 | 长于尾巴 | other_unparsed | S1T74 |
| 头比较长，和躯干差不多。 | 1 | 和躯干差不多 | body_geometry, global_balance | S1T5 |
| 尾巴和脖子长度不一样。 | 1 | 尾巴和脖子长度不一样 | disjoint_inequality | S1T41 |
| 点错了。 | 1 | 点错了 | other_unparsed | S1T310 |
| 脖子明显短于其他四个部位。 | 1 | 脖子明显短于其他四个部位 | other_reference | S1T59 |
| 脖子明显长于其他的四个部位。 | 1 | 脖子明显长于其他的四个部位 | other_reference | S1T107 |
| 脖子比较长，跟腿差不多，也跟躯干差不多。 | 1 | 跟腿差不多; 也跟躯干差不多 | body_geometry, global_balance | S1T9 |
| 脖子比较长，长于头。 | 1 | 长于头 | other_unparsed | S1T13 |
| 脖子非常长，跟躯干差不多。 | 1 | 跟躯干差不多 | body_geometry, global_balance | S1T2 |
| 腿比较长，长于头和尾巴。 | 1 | 长于头和尾巴 | other_unparsed | S1T6 |

#### S121

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿不是所有部位里最短的。 | 4 | 腿不是所有部位里最短的 | count_abstract | S2T30, S2T34, S2T35, S2T36 |
| 头在躯干上方，腿比较长。 | 3 | 头在躯干上方 | body_geometry | S1T291, S1T294, S1T296 |
| 头不是所有部位里最短的。 | 2 | 头不是所有部位里最短的 | count_abstract | S2T38, S2T39 |
| 腿不是最短的部位。 | 2 | 腿不是最短的部位 | other_unparsed | S2T56, S2T57 |
| 头和脖子差不多，都比较长，腿较短。 | 1 | 都比较长 | other_unparsed | S1T237 |
| 头在躯干上方，腿比较短。 | 1 | 头在躯干上方 | body_geometry | S1T293 |
| 头在躯干上方，腿比较长，尾巴很短。 | 1 | 头在躯干上方 | body_geometry | S1T295 |
| 头在躯干上方，腿较短。 | 1 | 头在躯干上方 | body_geometry | S1T299 |
| 头在躯干上方，腿较长。 | 1 | 头在躯干上方 | body_geometry | S1T298 |
| 头在躯干下方。 | 1 | 头在躯干下方 | body_geometry | S1T246 |
| 头在躯干下方，腿比较短。 | 1 | 头在躯干下方 | body_geometry | S1T292 |
| 头在躯干下方，腿较短。 | 1 | 头在躯干下方 | body_geometry | S1T297 |
| 头在躯干下方，腿较长。 | 1 | 头在躯干下方 | body_geometry | S1T300 |
| 头在躯干的上方。 | 1 | 头在躯干的上方 | body_geometry | S1T289 |
| 头比脖子短，腿和尾巴差不多，都很长。 | 1 | 都很长 | other_unparsed | S1T207 |
| 头比脖子短，腿比尾巴，差不多。 | 1 | 腿比尾巴; 差不多 | global_balance | S1T212 |
| 头比脖子长一点。腿和尾巴差不多，都很短。 | 1 | 都很短 | other_unparsed | S1T145 |
| 头比脖子长，腿远远长于头和脖子，尾巴较短。 | 1 | 腿远远长于头和脖子 | other_unparsed | S1T127 |
| 头脖子，尾巴，头脖子腿差不多，尾巴较短。 | 1 | 头脖子; 尾巴 | other_unparsed | S1T140 |
| 脖子和腿差别比较大。 | 1 | 脖子和腿差别比较大 | other_unparsed | S1T21 |

#### S123

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、腿、尾巴、躯干这些部位的长度进行描述，但不一定用的是这几个词语，且可能会涉及到这几个部位之间的比较，包括大小和长短关系。 | 1 | 头、脖子、腿、尾巴、躯干这些部位的长度进行描述; 但不一定用的是这几个词语; 且可能会涉及到这几个部位之间的比较; 包括大小和长短关系 | count_abstract, body_geometry | S1T73 |
| 头和脖外，头和腿长。 | 1 | 头和脖子外 | other_unparsed | S2T63 |
| 头，脖子，腿，尾巴长。 | 1 | 头; 脖子; 腿 | other_unparsed | S2T86 |
| 头，腿长。 | 1 | 头 | other_unparsed | S2T105 |
| 腿和头。头短，尾巴和脖子长。 | 1 | 腿和头 | other_unparsed | S1T91 |
| 腿短，脖，脖子和头长，尾巴短。 | 1 | 脖子 | other_unparsed | S1T60 |
| 腿长，脖子，尾巴、腿都偏短。 | 1 | 脖子 | other_unparsed | S1T25 |

#### S125

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿很长，基本是最长。 | 2 | 基本是最长 | other_unparsed | S1T93, S1T101 |
| 腿非常短，几乎是最短。 | 2 | 几乎是最短 | other_unparsed | S1T92, S1T109 |
| 头短，脖子较短，腿较短，尾巴短，整体看起来比较短小。 | 1 | 整体看起来比较短小 | other_unparsed | S1T39 |
| 头短，脖子较长，腿较长，尾巴较短，整体看起来比较修长。 | 1 | 整体看起来比较修长 | other_unparsed | S1T50 |
| 头长度适中，脖子短，腿短，尾巴较短，整体看起来比较短小。 | 1 | 整体看起来比较短小 | other_unparsed | S1T51 |
| 头长，脖子短，腿短，尾巴较短，整体看起来比较短小。 | 1 | 整体看起来比较短小 | other_unparsed | S1T43 |
| 头长，脖子较短，腿短，尾巴较短，头、脖子、尾巴均比腿长。 | 1 | 头、脖子、尾巴均比腿长 | other_unparsed | S1T8 |
| 头长，脖子长，腿较长，尾巴长，脖子比腿长一点，头和脖子占的比重比较大。 | 1 | 头和脖子占的比重比较大 | other_unparsed | S1T40 |
| 头长，脖子长，腿长，尾巴短，头和脖子的比重比较大，脖子比腿长一点。 | 1 | 头和脖子的比重比较大 | other_unparsed | S1T42 |
| 腿很短，几乎是最短。 | 1 | 几乎是最短 | other_unparsed | S1T114 |
| 腿长度适中，头和脖子比较长，尾巴长度适中，在整体比例中腿显得比较短。 | 1 | 在整体比例中腿显得比较短 | proportion_or_ratio | S1T58 |
| 腿长度适中，脖子比腿更长，整体很修长。 | 1 | 整体很修长 | other_unparsed | S1T102 |
| 腿非常短，基本是最短。 | 1 | 基本是最短 | other_unparsed | S1T95 |
| 腿非常长，基本是最长。 | 1 | 基本是最长 | other_unparsed | S1T121 |

#### S127

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 差不多，都一样长。 | 1 | 差不多; 都一样长 | other_unparsed | S1T90 |
| 腿和头比较长，脖子后尾巴稍微短，一点。 | 1 | 一点 | other_unparsed | S1T17 |

#### S129

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 一个部位很长。 | 10 | 一个部位很长 | count_abstract | S1T32, S1T33, S1T36, S1T37, S1T38, S1T39, S1T43, S1T44 |
| 头在躯干之下。 | 9 | 头在躯干之下 | body_geometry | S1T108, S1T111, S1T114, S1T116, S1T117, S1T118, S1T119, S1T121 |
| 尾巴和脖子不一样长。 | 7 | 尾巴和脖子不一样长 | disjoint_inequality | S1T26, S1T27, S1T54, S1T55, S1T57, S1T58, S1T155 |
| 有两个部位一样长。 | 6 | 有两个部位一样长 | count_abstract | S1T59, S1T60, S1T61, S1T62, S1T64, S1T66 |
| 低头。 | 4 | 低头 | other_unparsed | S1T94, S1T95, S1T97, S1T98 |
| 头在躯干之上。 | 4 | 头在躯干之上 | body_geometry | S1T110, S1T112, S1T113, S1T126 |
| 四个部位和躯干都不一样长。 | 3 | 四个部位和躯干都不一样长 | body_geometry, disjoint_inequality | S1T158, S1T159, S1T160 |
| 头在腿之上。 | 3 | 头在腿之上 | body_geometry | S1T89, S1T91, S1T101 |
| 头和尾巴不一样长。 | 2 | 头和尾巴不一样长 | disjoint_inequality | S1T23, S1T24 |
| 头在腿上。 | 2 | 头在腿上 | other_unparsed | S1T92, S1T93 |
| 头朝左。 | 2 | 头朝左 | other_unparsed | S1T6, S1T20 |
| 有一个部位比躯干长。 | 2 | 有一个部位比躯干长 | count_abstract, body_geometry | S1T130, S1T131 |
| 没有两个部位一样长。 | 2 | 没有两个部位一样长 | count_abstract | S1T63, S1T67 |
| 一个部位很短。 | 1 | 一个部位很短 | count_abstract | S1T34 |
| 抬头。 | 1 | 抬头 | other_unparsed | S1T96 |
| 脖子和尾巴不一样长。 | 1 | 脖子和尾巴不一样长 | disjoint_inequality | S1T165 |
| 脖子尾巴不一样长。 | 1 | 脖子尾巴不一样长 | disjoint_inequality | S1T166 |
| 腿和躯干不一样长。 | 1 | 腿和躯干不一样长 | body_geometry, disjoint_inequality | S1T171 |
| 腿短，低头。 | 1 | 低头 | other_unparsed | S1T100 |

#### S130

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿和尾巴都比脖子长。 | 16 | 腿和尾巴都比脖子长 | other_unparsed | S1T180, S1T244, S1T245, S1T248, S1T250, S1T252, S1T253, S1T256 |
| 头身比例比较协调。 | 5 | 头身比例比较协调 | proportion_or_ratio | S1T30, S1T31, S1T32, S1T33, S1T36 |
| 尾巴和腿都比脖子短。 | 5 | 尾巴和腿都比脖子短 | other_unparsed | S1T280, S1T281, S1T282, S1T284, S1T285 |
| 头身比例不协调。 | 4 | 头身比例不协调 | proportion_or_ratio | S1T29, S1T34, S1T35, S1T37 |
| 像爬行类的动物。 | 2 | 像爬行类的动物 | other_unparsed | S1T113, S1T114 |
| 腿和尾巴都比脖子短。 | 2 | 腿和尾巴都比脖子短 | other_unparsed | S1T276, S1T288 |
| 像直立行走的动物。 | 1 | 像直立行走的动物 | other_unparsed | S1T122 |
| 头和尾巴都比脖子长。 | 1 | 头和尾巴都比脖子长 | other_unparsed | S1T184 |
| 头和脖子都比腿短。 | 1 | 头和脖子都比腿短 | other_unparsed | S1T8 |
| 比例看起来不是很协调。 | 1 | 比例看起来不是很协调 | proportion_or_ratio | S1T20 |

#### S131

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 体型分布的不均匀。 | 4 | 体型分布的不均匀 | global_balance, vague_size | S1T70, S1T71, S1T79, S1T81 |
| 体型分布的很均匀。 | 3 | 体型分布的很均匀 | global_balance, vague_size | S1T74, S1T75, S1T77 |
| 朝左边。 | 3 | 朝左边 | other_unparsed | S1T27, S1T51, S1T53 |
| 它像一个小型的动物。 | 2 | 它像一个小型的动物 | other_unparsed | S1T65, S1T67 |
| 朝右边。 | 2 | 朝右边 | other_unparsed | S1T52, S1T54 |
| 腿比头长，朝右边。 | 2 | 朝右边 | other_unparsed | S1T42, S1T47 |
| 体型分布得不均匀。 | 1 | 体型分布得不均匀 | global_balance, vague_size | S1T80 |
| 体型分布得很均匀。 | 1 | 体型分布得很均匀 | global_balance, vague_size | S1T76 |
| 体型分布的不均匀，且方向是朝左。 | 1 | 体型分布的不均匀; 且方向是朝左 | global_balance, vague_size | S1T82 |
| 体型分布的还算均匀，头很长。 | 1 | 体型分布的还算均匀 | global_balance, vague_size | S1T72 |
| 体型比较大且分布均匀。 | 1 | 体型比较大且分布均匀 | global_balance, vague_size | S1T69 |
| 体型看上去很大，各个部位都很长。 | 1 | 体型看上去很大 | vague_size | S1T15 |
| 各部位分布均匀，都很长。 | 1 | 都很长 | other_unparsed | S1T64 |
| 头长，腿很长，分布均匀。 | 1 | 分布均匀 | global_balance | S1T68 |
| 它像一个大型的动物。 | 1 | 它像一个大型的动物 | other_unparsed | S1T66 |
| 尾巴很长，朝右侧。 | 1 | 朝右侧 | other_unparsed | S1T25 |
| 尾巴长，腿短，是个大型动物。 | 1 | 是个大型动物 | other_unparsed | S1T130 |
| 是个大型动物。 | 1 | 是个大型动物 | other_unparsed | S1T131 |
| 朝右边，腿长。 | 1 | 朝右边 | other_unparsed | S1T55 |
| 朝左侧，尾巴很短。 | 1 | 朝左侧 | other_unparsed | S1T24 |
| 朝左边，头长。 | 1 | 朝左边 | other_unparsed | S1T56 |
| 朝左边，腿长，头长。 | 1 | 朝左边 | other_unparsed | S1T57 |
| 腿很长，个子很高。 | 1 | 个子很高 | other_unparsed | S1T83 |
| 腿很长，分布得很均匀。 | 1 | 分布得很均匀 | global_balance | S1T73 |
| 腿很长，尾巴很短，像一条狗。 | 1 | 像一条狗 | other_unparsed | S1T84 |
| 腿比头短，朝右边。 | 1 | 朝右边 | other_unparsed | S1T48 |
| 腿比头短，朝左边。 | 1 | 朝左边 | other_unparsed | S1T43 |
| 腿比头长，朝左边。 | 1 | 朝左边 | other_unparsed | S1T41 |
| 腿短，不均匀。 | 1 | 不均匀 | global_balance | S1T63 |

#### S132

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴不是最短的。 | 8 | 尾巴不是最短的 | other_unparsed | S1T118, S1T144, S1T145, S1T146, S1T147, S1T149, S1T151, S1T154 |
| 尾巴不是最长的。 | 3 | 尾巴不是最长的 | other_unparsed | S1T93, S1T152, S1T153 |
| 尾巴短于某个数值。 | 2 | 尾巴短于某个数值 | other_unparsed | S2T14, S2T15 |
| 尾巴长于某个数值。 | 1 | 尾巴长于某个数值 | other_unparsed | S2T16 |

#### S202

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S1T209 |

#### S203

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴均长于脖子，腿很短。 | 1 | 头和尾巴均长于脖子 | other_unparsed | S1T97 |
| 头和尾巴很长，长于脖子，长于腿。 | 1 | 长于脖子; 长于腿 | other_unparsed | S1T137 |
| 头显著短、短于脖子，尾巴很长，腿很短。 | 1 | 头显著短、短于脖子 | other_unparsed | S1T107 |
| 尾巴很长，头最短，躯干中等。 | 1 | 躯干中等 | body_geometry | S1T155 |
| 所有部位都中等偏长，中等偏短。 | 1 | 中等偏短 | other_unparsed | S2T28 |
| 脖子长于头和尾巴，但都它们都很长，腿很短。 | 1 | 但都它们都很长 | other_unparsed | S1T109 |
| 腿很长，头很短，脖子和尾巴长度对称。 | 1 | 脖子和尾巴长度对称 | other_unparsed | S1T37 |
| 腿很长，尾巴很短，体很长，体中等。 | 1 | 体很长; 体中等 | other_unparsed | S1T303 |
| 腿很长，颈部也很短。 | 1 | 颈部也很短 | other_unparsed | S1T30 |
| 身体各个部位都很匀称。 | 1 | 身体各个部位都很匀称 | body_geometry, global_balance | S1T15 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S2T112 |

#### S204

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位几乎一样长，都挺长。 | 1 | 都挺长 | other_unparsed | S1T130 |
| 四个部位都比较长，长度都差不多。 | 1 | 长度都差不多 | global_balance | S1T203 |
| 头、脖子、尾巴都很长，腿比它们稍微短一点。 | 1 | 腿比它们稍微短一点 | other_unparsed | S1T1 |
| 头和尾巴比较短，脖子和腿长，选错了。 | 1 | 选错了 | meta_or_uncertain | S1T186 |
| 头和尾巴都比脖子长。 | 1 | 头和尾巴都比脖子长 | other_unparsed | S1T113 |
| 头很小，尾巴很长，脖子也很长。 | 1 | 头很小 | vague_size | S1T179 |
| 头很长，腿差不多正好。 | 1 | 腿差不多正好 | other_unparsed | S1T241 |
| 头比尾巴长，脖子也比尾巴长。 | 1 | 脖子也比尾巴长 | other_unparsed | S1T223 |
| 头短，脖子差不多刚好。 | 1 | 脖子差不多刚好 | other_unparsed | S2T91 |
| 头短，脖子长，脖子和躯干差不多。 | 1 | 脖子和躯干差不多 | body_geometry, global_balance | S1T276 |
| 头长，腿长，很均衡。 | 1 | 很均衡 | global_balance | S1T311 |
| 尾巴不比脖子长。 | 1 | 尾巴不比脖子长 | other_unparsed | S1T259 |
| 尾巴相当短，腿相当长。 | 1 | 腿相当长 | global_balance | S1T187 |
| 脖子很长，头很小。 | 1 | 头很小 | vague_size | S1T159 |

#### S205

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子和腿，差不多长。 | 1 | 头、脖子和腿; 差不多长 | global_balance | S1T126 |
| 腿，头最长。 | 1 | 腿 | other_unparsed | S2T224 |
| 腿，尾巴最长。 | 1 | 腿 | other_unparsed | S2T235 |
| 腿，脖子最长。 | 1 | 腿 | other_unparsed | S3T7 |

#### S206

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子比头短，腿也比尾巴短。 | 2 | 腿也比尾巴短 | other_unparsed | S3T82, S3T85 |
| 脖子比头长，腿也比尾巴长。 | 2 | 腿也比尾巴长 | other_unparsed | S3T35, S3T37 |
| 像小狗。 | 1 | 像小狗 | other_unparsed | S2T144 |
| 像小狗一样。 | 1 | 像小狗一样 | other_unparsed | S2T86 |
| 像食蚁兽。 | 1 | 像食蚁兽 | other_unparsed | S2T92 |
| 像食蚁兽，脖子很长。 | 1 | 像食蚁兽 | other_unparsed | S2T64 |
| 头和尾巴比较长，尾巴，腿比较短，脖子也很长。 | 1 | 尾巴 | other_unparsed | S2T7 |
| 头和脖子。 | 1 | 头和脖子 | other_unparsed | S1T27 |
| 头和脖子都比腿短。 | 1 | 头和脖子都比腿短 | other_unparsed | S2T51 |
| 头和脖子都比较小，腿很长。 | 1 | 头和脖子都比较小 | vague_size | S4T74 |
| 头明显比脖子长，腿和尾巴都很长。 | 1 | 头明显比脖子长 | other_unparsed | S4T32 |
| 短长。 | 1 | 短长 | other_unparsed | S1T92 |
| 脖子和头比，脖子很短，腿比尾巴要长。 | 1 | 脖子和头比 | other_unparsed | S3T281 |
| 脖子很短，腿、尾巴，头很长。 | 1 | 腿、尾巴 | other_unparsed | S1T8 |
| 脖子很长，比头长，腿比尾巴长。 | 1 | 比头长 | other_unparsed | S3T184 |
| 脖子比头略长，腿比尾巴略长，它们都很长。 | 1 | 它们都很长 | other_unparsed | S3T305 |
| 腿和脖子，腿比较短，脖子比较长，头比较短。 | 1 | 腿和脖子 | other_unparsed | S3T127 |
| 腿很短，头和脖子，较为长。 | 1 | 头和脖子; 较为长 | other_unparsed | S5T119 |
| 腿比较短，像小狗。 | 1 | 像小狗 | other_unparsed | S2T65 |
| 较为均衡。 | 1 | 较为均衡 | global_balance | S1T158 |
| 较为等长。 | 1 | 较为等长 | other_unparsed | S1T101 |

#### S207

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位都很长，体型很大。 | 4 | 体型很大 | vague_size | S1T157, S1T184, S1T195, S1T210 |
| 四个部位长度差不多，都比较长。 | 2 | 都比较长 | other_unparsed | S1T63, S1T114 |
| 四个部位都很长，体型非常大。 | 1 | 体型非常大 | vague_size | S1T198 |
| 四个部位都比较长，体型比较大。 | 1 | 体型比较大 | vague_size | S1T145 |
| 四个部位都比较长，长度接近。 | 1 | 长度接近 | other_unparsed | S1T101 |
| 四个部位长度相仿。 | 1 | 四个部位长度相仿 | other_unparsed | S1T8 |
| 头、尾巴和腿都非常长，脖子明显比这三个短。 | 1 | 脖子明显比这三个短 | other_unparsed | S1T152 |
| 头、脖子和腿都很长，体型相对比较大。 | 1 | 体型相对比较大 | vague_size | S1T160 |
| 头和脖子一样长，都是最长的，其他两个比较短。 | 1 | 都是最长的 | other_unparsed | S1T40 |
| 头和脖子都非常长，体型比较大。 | 1 | 体型比较大 | vague_size | S1T151 |
| 头和腿很长，比脖子长。 | 1 | 比脖子长 | other_unparsed | S1T247 |
| 头最长，腿和尾巴也很长，和头比较接近，脖子很短。 | 1 | 和头比较接近 | other_unparsed | S1T136 |
| 尾巴很长，和前三个部位长度差不多。 | 1 | 和前三个部位长度差不多 | count_abstract, global_balance | S1T166 |
| 尾巴最长，其他三个部位一样长，都比较长。 | 1 | 都比较长 | other_unparsed | S1T34 |
| 脖子和腿都很长，体型很大。 | 1 | 体型很大 | vague_size | S1T199 |
| 脖子明显比头长很多。 | 1 | 脖子明显比头长很多 | other_unparsed | S2T4 |
| 脖子最长，体型很大。 | 1 | 体型很大 | vague_size | S1T206 |
| 脖子最长，体型比较大。 | 1 | 体型比较大 | vague_size | S1T187 |
| 脖子最长，体型相对比较小。 | 1 | 体型相对比较小 | vague_size | S1T177 |
| 脖子最长，比头长很多。 | 1 | 比头长很多 | other_unparsed | S2T6 |
| 脖子长比头要长一点。 | 1 | 脖子长比头要长一点 | other_unparsed | S3T106 |
| 脖子非常长，四个部位都比较长，体型很大。 | 1 | 体型很大 | vague_size | S1T134 |
| 除了头，都非常长，体型很大。 | 1 | 体型很大 | vague_size | S1T188 |

#### S208

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四者都很长。 | 2 | 四者都很长 | other_unparsed | S1T113, S1T177 |
| 三个长，一个短。 | 1 | 三个长; 一个短 | other_unparsed | S1T172 |
| 两长两短。 | 1 | 两长两短 | count_abstract | S1T297 |
| 四者均匀的很长。 | 1 | 四者均匀的很长 | global_balance | S1T195 |
| 四者有三个很长。 | 1 | 四者有三个很长 | other_unparsed | S1T114 |
| 四者都差不多长。 | 1 | 四者都差不多长 | global_balance | S1T182 |
| 四者都很短，尤其是腿很短。 | 1 | 四者都很短 | other_unparsed | S1T71 |
| 四者都比较长。 | 1 | 四者都比较长 | other_unparsed | S1T111 |
| 四者都较短。 | 1 | 四者都较短 | other_unparsed | S1T120 |
| 均匀的长。 | 1 | 均匀的长 | global_balance | S1T283 |
| 头和尾巴，头和脖子较短，腿和尾巴长。 | 1 | 头和尾巴 | other_unparsed | S1T85 |
| 头和脖子，较短。 | 1 | 头和脖子; 较短 | other_unparsed | S1T284 |
| 头和，四个部位都长。 | 1 | 头和 | other_unparsed | S1T38 |
| 头短，脖，脖子、腿和尾巴都长。 | 1 | 脖子 | other_unparsed | S1T92 |
| 就都挺短。 | 1 | 就都挺短 | other_unparsed | S1T210 |
| 就都挺长。 | 1 | 就都挺长 | other_unparsed | S1T209 |
| 我真的不知道，没什么区别。 | 1 | 我真的不知道; 没什么区别 | meta_or_uncertain | S1T160 |
| 脖子很长，尾巴稍短一些，腿和腿一样长。 | 1 | 腿和腿一样长 | other_unparsed | S1T32 |
| 腿偏短，脖子、尾巴和头都偏长，一样长。 | 1 | 一样长 | other_unparsed | S1T9 |
| 都非常短。 | 1 | 都非常短 | other_unparsed | S1T289 |
| 长度比较均匀。 | 1 | 长度比较均匀 | global_balance | S1T244 |

#### S209

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴长，脖子和腿短，头明显比脖子长。 | 2 | 头明显比脖子长 | other_unparsed | S1T88, S1T160 |
| 头最长，脖子短，腿和尾巴中等，头明显比脖子长。 | 2 | 头明显比脖子长 | other_unparsed | S1T152, S1T198 |
| 头最长，腿中等，脖子和尾巴短，头明显比脖子长。 | 2 | 头明显比脖子长 | other_unparsed | S1T157, S1T201 |
| 头、脖子、腿都长尾巴，中等长度。 | 1 | 中等长度 | other_unparsed | S1T40 |
| 头、脖子和腿，尾巴长，短。 | 1 | 头、脖子和腿; 短 | other_unparsed | S1T58 |
| 头和尾巴中等偏长，脖子和腿短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T139 |
| 头和脖子，所有部位都差不多长，其中头和脖子长度接近，并且是中等偏长。 | 1 | 头和脖子; 并且是中等偏长 | other_unparsed | S1T107 |
| 头最长，尾巴中等，脖子中等偏短，腿短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T186 |
| 头最长，脖子、腿中等，尾巴短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T163 |
| 头最长，脖子中等，尾巴和腿短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T141 |
| 头最长，腿和尾巴中等偏长，脖子短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T153 |
| 头长，尾巴长，脖子和腿中等偏短，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T90 |
| 头长，脖子中等，偏长。 | 1 | 偏长 | other_unparsed | S2T124 |
| 头长，脖子短，头、腿、尾巴中等长度，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T169 |
| 头长，脖子短，腿和尾巴中等，头明显比脖子长。 | 1 | 头明显比脖子长 | other_unparsed | S1T144 |
| 所有部位都是中等偏长，而且长度差不多。 | 1 | 而且长度差不多 | global_balance | S1T18 |
| 所有部位都是中等长度，而且都差不多长。 | 1 | 而且都差不多长 | global_balance | S1T78 |
| 脖子、腿、尾巴长，头中等偏短，头明显比脖子短。 | 1 | 头明显比脖子短 | other_unparsed | S1T135 |
| 脖子很长，头，中等长度，尾巴和腿比较短。 | 1 | 头; 中等长度 | other_unparsed | S1T21 |
| 脖子长，头中等，总体来说比较长。 | 1 | 总体来说比较长 | other_unparsed | S1T320 |
| 脖子长，腿长，头中等，尾巴短，腿明显比尾巴长。 | 1 | 腿明显比尾巴长 | other_unparsed | S1T191 |
| 腿长，头短，脖子和尾巴中等，并且接近。 | 1 | 并且接近 | other_unparsed | S1T117 |
| 腿长，头短，脖子和尾巴中等，腿明显比尾巴长。 | 1 | 腿明显比尾巴长 | other_unparsed | S1T156 |
| 腿长，尾巴，中等偏短。 | 1 | 尾巴; 中等偏短 | other_unparsed | S1T272 |
| 腿长，脖子长，头和尾巴短，腿明显比尾巴长。 | 1 | 腿明显比尾巴长 | other_unparsed | S1T190 |

#### S210

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 有三个部位几乎一样长。 | 13 | 有三个部位几乎一样长 | count_abstract | S1T29, S1T106, S1T108, S1T142, S1T193, S1T257, S1T260, S1T262 |
| 三个部位几乎一样长。 | 11 | 三个部位几乎一样长 | count_abstract | S1T170, S1T190, S1T223, S1T224, S1T241, S1T290, S1T310, S1T314 |
| 有两个部位几乎一样长。 | 9 | 有两个部位几乎一样长 | count_abstract | S1T30, S1T38, S1T135, S1T138, S1T141, S1T144, S1T167, S1T259 |
| 有三个部位长度一样。 | 7 | 有三个部位长度一样 | count_abstract | S3T189, S3T190, S3T207, S3T210, S3T216, S3T236, S4T216 |
| 有三个部位一样长。 | 6 | 有三个部位一样长 | count_abstract | S1T64, S1T210, S2T199, S3T23, S3T51, S3T108 |
| 两个部位一样长。 | 5 | 两个部位一样长 | count_abstract | S1T86, S1T90, S1T93, S1T179, S1T180 |
| 两个部位几乎一样长。 | 5 | 两个部位几乎一样长 | count_abstract | S1T164, S2T9, S2T76, S2T112, S2T116 |
| 三个部位长度一样。 | 4 | 三个部位长度一样 | count_abstract | S3T82, S3T83, S3T120, S3T122 |
| 躯干最长。 | 3 | 躯干最长 | body_geometry | S1T75, S1T78, S1T81 |
| 三个部位一样长。 | 2 | 三个部位一样长 | count_abstract | S2T272, S2T278 |
| 三个部位长度相似。 | 2 | 三个部位长度相似 | count_abstract | S1T278, S4T51 |
| 两两长度一样。 | 2 | 两两长度一样 | other_unparsed | S2T214, S2T217 |
| 头。 | 2 | 头 | other_unparsed | S4T294, S5T134 |
| 有两个部位一样长。 | 2 | 有两个部位一样长 | count_abstract | S1T40, S1T255 |
| 躯干是最长。 | 2 | 躯干是最长 | body_geometry | S1T39, S1T73 |
| 都差不多长。 | 2 | 都差不多长 | global_balance | S2T242, S2T312 |
| 长度两两相似。 | 2 | 长度两两相似 | other_unparsed | S3T10, S3T118 |
| 三个部位一样长，头最小。 | 1 | 三个部位一样长 | count_abstract | S4T300 |
| 三个部位明显长。 | 1 | 三个部位明显长 | count_abstract | S1T161 |
| 三个部位相似，脖子最长。 | 1 | 三个部位相似 | count_abstract | S4T84 |
| 三个部位相似，腿最短。 | 1 | 三个部位相似 | count_abstract | S4T83 |
| 三个部位，长最长，尾巴最短。 | 1 | 三个部位; 长最长 | count_abstract | S4T82 |
| 两两一样长。 | 1 | 两两一样长 | other_unparsed | S2T244 |
| 两两长度相似。 | 1 | 两两长度相似 | other_unparsed | S3T162 |
| 两部位一样长。 | 1 | 两部位一样长 | other_unparsed | S1T127 |
| 两部位长相似。 | 1 | 两部位长相似 | other_unparsed | S1T303 |
| 几个部位几乎一样长。 | 1 | 几个部位几乎一样长 | count_abstract | S1T235 |
| 几个部位的长度差不多。 | 1 | 几个部位的长度差不多 | count_abstract, global_balance | S3T196 |
| 几个部位都比躯干短。 | 1 | 几个部位都比躯干短 | count_abstract, body_geometry | S1T58 |
| 几个部位长度一样。 | 1 | 几个部位长度一样 | count_abstract | S4T140 |
| 几个部位长度差别不大。 | 1 | 几个部位长度差别不大 | count_abstract | S3T31 |
| 几个部位长度相差不大。 | 1 | 几个部位长度相差不大 | count_abstract | S3T145 |
| 又有两个部位一样长。 | 1 | 又有两个部位一样长 | count_abstract | S1T32 |
| 各部位都差不多，较短，尾巴最短。 | 1 | 较短 | other_unparsed | S5T74 |
| 四个部位都很长，且几乎一样长。 | 1 | 且几乎一样长 | other_unparsed | S5T16 |
| 头和尾巴一样长，都是最长。 | 1 | 都是最长 | other_unparsed | S3T102 |
| 头和尾巴长度一样，都最短，腿第二短，脖子很长。 | 1 | 都最短 | other_unparsed | S4T9 |
| 头和脖子最长，长于腿、长于尾巴。 | 1 | 长于腿、长于尾巴 | other_unparsed | S4T30 |
| 头和脖子，腿和尾巴比较短。 | 1 | 头和脖子 | other_unparsed | S5T14 |
| 头和腿几乎一样长，而且最短。 | 1 | 而且最短 | other_unparsed | S2T32 |
| 头比较长，脖子，尾巴最短。 | 1 | 脖子 | other_unparsed | S5T155 |
| 头，头比脖子短。 | 1 | 头 | other_unparsed | S2T158 |
| 头，尾巴最短。 | 1 | 头 | other_unparsed | S3T168 |
| 头，第三长。 | 1 | 头; 第三长 | count_abstract, ordinal_or_secondary | S2T101 |
| 尾巴非常短，其他部位都，也比较短。 | 1 | 其他部位都; 也比较短 | other_reference | S5T33 |
| 有三个部位比躯干长。 | 1 | 有三个部位比躯干长 | count_abstract, body_geometry | S1T137 |
| 有三个部位都一样长。 | 1 | 有三个部位都一样长 | count_abstract | S3T2 |
| 有三个部位长度一致。 | 1 | 有三个部位长度一致 | count_abstract | S3T172 |
| 有两个部位长度一样。 | 1 | 有两个部位长度一样 | count_abstract | S3T80 |
| 有两个部位长得相似。 | 1 | 有两个部位长得相似 | count_abstract | S1T299 |
| 脖子、尾巴、腿的长度，一样。 | 1 | 脖子、尾巴、腿的长度; 一样 | other_unparsed | S3T199 |
| 脖子。 | 1 | 脖子 | other_unparsed | S2T39 |
| 脖子和尾巴，几乎一样长。 | 1 | 脖子和尾巴; 几乎一样长 | other_unparsed | S2T36 |
| 腿和尾巴很长，另两个部位非常短。 | 1 | 另两个部位非常短 | count_abstract | S5T77 |
| 腿，头和尾巴一样长。 | 1 | 腿 | other_unparsed | S3T146 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S5T316 |
| 都是中等长度。 | 1 | 都是中等长度 | other_unparsed | S5T51 |
| 长度差不多。 | 1 | 长度差不多 | global_balance | S3T229 |
| 长度相似，脖子最短。 | 1 | 长度相似 | other_unparsed | S4T79 |

#### S211

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度差不多，比较长。 | 1 | 比较长 | other_unparsed | S3T57 |
| 头和尾巴都比较长，一样长，腿也比较长。 | 1 | 一样长 | other_unparsed | S2T175 |
| 头和脖子，尾巴和腿中有三个是一样长。 | 1 | 头和脖子 | other_unparsed | S1T57 |
| 头最长，比脖子和尾巴都长。 | 1 | 比脖子和尾巴都长 | other_unparsed | S1T1 |
| 头，和腿基本一样长。 | 1 | 头; 和腿基本一样长 | other_unparsed | S1T55 |
| 都不太长，只有腿最短。 | 1 | 都不太长 | other_unparsed | S3T58 |

#### S212

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位较均等。 | 17 | 四个部位较均等 | global_balance | S1T61, S1T65, S1T68, S1T72, S1T93, S1T119, S1T124, S1T126 |
| 头比脖子短，比尾巴短。 | 10 | 比尾巴短 | other_unparsed | S4T138, S4T139, S4T145, S4T146, S4T151, S4T152, S4T154, S4T155 |
| 头比脖子长，比尾巴短。 | 7 | 比尾巴短 | other_unparsed | S4T93, S4T140, S4T142, S4T147, S4T148, S4T163, S4T166 |
| 头比脖子长，比尾巴长。 | 7 | 比尾巴长 | other_unparsed | S4T141, S4T144, S4T149, S4T150, S4T153, S4T168, S4T195 |
| 尾巴比脖子长，比头长。 | 7 | 比头长 | other_unparsed | S3T258, S3T261, S3T271, S3T272, S4T88, S4T91, S4T92 |
| 四个部位较匀称。 | 6 | 四个部位较匀称 | global_balance | S1T182, S1T242, S1T246, S1T255, S1T259, S2T24 |
| 尾巴比脖子短，比头短。 | 6 | 比头短 | other_unparsed | S3T257, S3T262, S3T265, S4T87, S4T89, S4T90 |
| 脖子比头短，比尾巴短。 | 6 | 比尾巴短 | other_unparsed | S4T96, S4T97, S4T98, S4T101, S4T105, S4T110 |
| 脖子比头长，比尾巴长。 | 5 | 比尾巴长 | other_unparsed | S4T94, S4T99, S4T103, S4T106, S4T109 |
| 头比脖子短，比尾巴长。 | 4 | 比尾巴长 | other_unparsed | S4T137, S4T161, S4T167, S4T194 |
| 尾巴长，头的位置高。 | 4 | 头的位置高 | other_unparsed | S6T75, S6T76, S6T77, S6T81 |
| 腿比脖子短，比尾巴短。 | 4 | 比尾巴短 | other_unparsed | S4T170, S4T173, S4T174, S4T176 |
| 腿比脖子长，比尾巴长。 | 4 | 比尾巴长 | other_unparsed | S4T165, S4T172, S4T175, S4T177 |
| 四个部位较为匀称。 | 3 | 四个部位较为匀称 | global_balance | S1T236, S1T274, S1T278 |
| 四个部位长度较均等。 | 3 | 四个部位长度较均等 | global_balance | S1T8, S1T41, S1T57 |
| 尾巴比脖子短，比头长。 | 3 | 比头长 | other_unparsed | S3T259, S3T264, S4T86 |
| 尾巴长，头的位置低。 | 3 | 头的位置低 | other_unparsed | S6T74, S6T78, S6T80 |
| 脖子比头长，比尾巴短。 | 3 | 比尾巴短 | other_unparsed | S4T95, S4T100, S4T104 |
| 尾巴比脖子长，比头短。 | 2 | 比头短 | other_unparsed | S3T260, S3T263 |
| 尾巴比躯干短，比脖子短。 | 2 | 比脖子短 | other_unparsed | S3T316, S3T318 |
| 脖子和尾巴都比腿长。 | 2 | 脖子和尾巴都比腿长 | other_unparsed | S2T226, S2T228 |
| 脖子比尾巴短，比腿长。 | 2 | 比腿长 | other_unparsed | S4T190, S4T191 |
| 腿比脖子长，比尾巴短。 | 2 | 比尾巴短 | other_unparsed | S4T169, S4T171 |
| 上身长，腿短。 | 1 | 上身长 | other_unparsed | S2T294 |
| 四个部位均等。 | 1 | 四个部位均等 | global_balance | S1T9 |
| 四个部位的长度较均等。 | 1 | 四个部位的长度较均等 | global_balance | S1T6 |
| 四个部位较均等，腿较短。 | 1 | 四个部位较均等 | global_balance | S1T125 |
| 四个部位都很小。 | 1 | 四个部位都很小 | vague_size | S1T85 |
| 头比脖子长，比腿长。 | 1 | 比腿长 | other_unparsed | S4T143 |
| 尾巴比头短，比脖子短。 | 1 | 比脖子短 | other_unparsed | S3T270 |
| 尾巴比躯干短，比脖子长。 | 1 | 比脖子长 | other_unparsed | S3T319 |
| 尾巴短，头位置高。 | 1 | 头位置高 | other_unparsed | S6T73 |
| 尾巴短，头的位置低。 | 1 | 头的位置低 | other_unparsed | S6T79 |
| 尾巴长，头低。 | 1 | 头低 | other_unparsed | S6T72 |
| 脖子和尾巴一样长，比腿长。 | 1 | 比腿长 | other_unparsed | S2T243 |
| 脖子和尾巴短，比头长。 | 1 | 比头长 | other_unparsed | S3T255 |
| 脖子和尾巴都比腿短。 | 1 | 脖子和尾巴都比腿短 | other_unparsed | S2T225 |
| 脖子比头短，比尾巴长。 | 1 | 比尾巴长 | other_unparsed | S4T107 |
| 脖子比头长，比腿长，比尾巴长。 | 1 | 比腿长; 比尾巴长 | other_unparsed | S4T102 |
| 脖子比尾巴短，比头短。 | 1 | 比头短 | other_unparsed | S3T269 |
| 脖子比尾巴短，比头长。 | 1 | 比头长 | other_unparsed | S3T256 |
| 脖子比尾巴短，比腿短。 | 1 | 比腿短 | other_unparsed | S4T192 |
| 脖子比尾巴长，比头长。 | 1 | 比头长 | other_unparsed | S3T254 |
| 脖子比腿和尾巴短，比头短。 | 1 | 比头短 | other_unparsed | S4T108 |
| 腿比尾巴短，比脖子短。 | 1 | 比脖子短 | other_unparsed | S4T179 |
| 腿比脖子短，比尾巴长。 | 1 | 比尾巴长 | other_unparsed | S4T178 |
| 腿长，上身短。 | 1 | 上身短 | other_unparsed | S2T293 |

#### S213

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 有两个部位比较长。 | 7 | 有两个部位比较长 | count_abstract | S2T306, S2T309, S2T310, S2T312, S2T314, S2T317, S2T318 |
| 有三个部位比较长。 | 3 | 有三个部位比较长 | count_abstract | S2T305, S2T311, S2T316 |
| 有一个部位比较长。 | 2 | 有一个部位比较长 | count_abstract | S2T308, S2T315 |
| 脖子比头长，脖子不是最长。 | 2 | 脖子不是最长 | other_unparsed | S2T297, S2T301 |
| 腿较短，头和尾巴均长于脖子。 | 2 | 头和尾巴均长于脖子 | other_unparsed | S2T43, S2T44 |
| 四部位差不多长。 | 1 | 四部位差不多长 | global_balance | S1T56 |
| 头和尾巴比较长，长于脖子。 | 1 | 长于脖子 | other_unparsed | S2T65 |
| 头和尾巴都比脖子长。 | 1 | 头和尾巴都比脖子长 | other_unparsed | S1T99 |
| 头明显比脖子长，尾巴比较长。 | 1 | 头明显比脖子长 | other_unparsed | S3T81 |
| 头显著比脖子长。 | 1 | 头显著比脖子长 | other_unparsed | S4T18 |
| 头比尾巴短，比脖子短。 | 1 | 比脖子短 | other_unparsed | S1T131 |
| 头比脖子长，头不是最长。 | 1 | 头不是最长 | other_unparsed | S2T298 |
| 头比脖子长，脖子不是最短。 | 1 | 脖子不是最短 | other_unparsed | S2T300 |
| 头长，脖子长，尾巴长，腿和尾巴。 | 1 | 腿和尾巴 | other_unparsed | S1T91 |
| 脖子显著比头长。 | 1 | 脖子显著比头长 | other_unparsed | S4T17 |
| 腿和脖子都比尾巴长。 | 1 | 腿和脖子都比尾巴长 | other_unparsed | S3T161 |
| 腿比躯干长，头不比脖子短。 | 1 | 头不比脖子短 | other_unparsed | S2T24 |
| 腿比较长，头和脖子无明显差距。 | 1 | 头和脖子无明显差距 | other_unparsed | S3T167 |
| 腿比较长，头和脖子，四个部位都比较短。 | 1 | 头和脖子 | other_unparsed | S4T171 |
| 腿较长，头和脖子均长于尾巴。 | 1 | 头和脖子均长于尾巴 | other_unparsed | S2T42 |

#### S214

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 选错了。 | 39 | 选错了 | meta_or_uncertain | S1T97, S1T131, S1T148, S1T166, S1T249, S1T311, S2T27, S2T53 |
| 两长两短。 | 3 | 两长两短 | count_abstract | S3T220, S3T221, S3T222 |
| 差不多。 | 3 | 差不多 | global_balance | S2T35, S2T110, S4T18 |
| 三个中等，尾巴短。 | 1 | 三个中等 | other_unparsed | S3T139 |
| 四个部位都还行。 | 1 | 四个部位都还行 | other_unparsed | S5T246 |
| 头，和，腿、头和腿略短，其他中等长度。 | 1 | 头; 和 | other_unparsed | S1T60 |
| 头，脖子特别长，腿特别短。 | 1 | 头 | other_unparsed | S1T188 |
| 差距不大。 | 1 | 差距不大 | other_unparsed | S3T189 |
| 是长。 | 1 | 是长 | other_unparsed | S2T184 |
| 腿长，脖，脖子中等长度，尾巴略长，头略短。 | 1 | 脖子 | other_unparsed | S1T44 |
| 腿，最短。 | 1 | 腿; 最短 | other_unparsed | S3T180 |

#### S215

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 两个部位长，两个部位短。 | 7 | 两个部位长; 两个部位短 | count_abstract | S2T52, S2T62, S2T63, S2T87, S2T225, S2T232, S2T233 |
| 三个部位长。 | 6 | 三个部位长 | count_abstract | S2T35, S2T50, S2T184, S2T188, S2T189, S2T194 |
| 三个部位长，一个部位短。 | 6 | 三个部位长; 一个部位短 | count_abstract | S2T51, S2T57, S2T58, S2T64, S2T231, S2T236 |
| 三个部位短，一个部位长。 | 3 | 三个部位短; 一个部位长 | count_abstract | S2T59, S2T86, S2T237 |
| 三长一短。 | 3 | 三长一短 | count_abstract | S2T175, S2T238, S2T240 |
| 两个部位长。 | 3 | 两个部位长 | count_abstract | S2T185, S2T193, S2T195 |
| 两长两短。 | 2 | 两长两短 | count_abstract | S2T173, S2T174 |
| 只有一个部位长。 | 2 | 只有一个部位长 | count_abstract | S2T183, S2T187 |
| 一个部位长，三个部位短。 | 1 | 一个部位长; 三个部位短 | count_abstract | S2T229 |
| 三短一长。 | 1 | 三短一长 | other_unparsed | S2T239 |
| 三长一短，头最短。 | 1 | 三长一短 | count_abstract | S2T176 |
| 头和尾巴的长度超过腿。 | 1 | 头和尾巴的长度超过腿 | other_unparsed | S1T17 |
| 腿和尾巴不一样长，头比脖子长。 | 1 | 腿和尾巴不一样长 | disjoint_inequality | S2T282 |
| 腿和尾巴不一样，头和脖子不一样长。 | 1 | 腿和尾巴不一样; 头和脖子不一样长 | disjoint_inequality | S2T284 |
| 腿和尾巴都长，头和脖子不一样长。 | 1 | 头和脖子不一样长 | disjoint_inequality | S2T280 |

#### S216

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴和腿差不多长，都比脖子长。 | 2 | 都比脖子长 | other_unparsed | S2T121, S2T122 |
| 头、脖子、腿都很长，尾巴中等，微微偏短。 | 1 | 微微偏短 | other_unparsed | S1T24 |
| 尾巴和腿差不多长，偏短。 | 1 | 偏短 | other_unparsed | S1T319 |
| 尾巴和腿都一样长，偏短。 | 1 | 偏短 | other_unparsed | S1T294 |
| 尾巴最长，头和脖子都比尾巴短。 | 1 | 头和脖子都比尾巴短 | other_unparsed | S2T252 |
| 尾巴最长，比腿长也比脖子长。 | 1 | 比腿长也比脖子长 | other_unparsed | S2T115 |
| 尾巴比腿长，也比脖子长。 | 1 | 也比脖子长 | other_unparsed | S2T73 |
| 每个部位。 | 1 | 每个部位 | other_unparsed | S1T105 |
| 每个部分都差不多长。 | 1 | 每个部分都差不多长 | global_balance | S2T267 |
| 每个部分都挺长。 | 1 | 每个部分都挺长 | other_unparsed | S2T253 |
| 每个部分都短。 | 1 | 每个部分都短 | other_unparsed | S2T276 |
| 脖子和尾巴都挺，腿都挺长。 | 1 | 脖子和尾巴都挺 | other_unparsed | S2T206 |
| 脖子比腿，腿和尾巴长。 | 1 | 脖子比腿 | other_unparsed | S2T83 |
| 腿和尾巴一样长，偏长。 | 1 | 偏长 | other_unparsed | S2T37 |
| 腿最长，尾巴最短，脖子居中。 | 1 | 脖子居中 | other_unparsed | S2T141 |
| 都差不多长。 | 1 | 都差不多长 | global_balance | S2T214 |
| 都差不多长，中等长度。 | 1 | 都差不多长; 中等长度 | global_balance | S2T292 |

#### S217

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴长，有两个部位短。 | 2 | 有两个部位短 | count_abstract | S4T74, S4T75 |
| 头比和脖子比较长。 | 1 | 头比和脖子比较长 | other_unparsed | S2T97 |
| 由长到短是头、腿、脖子，尾巴。 | 1 | 尾巴 | other_unparsed | S1T240 |
| 由长到短是头和腿，脖子和尾巴。 | 1 | 脖子和尾巴 | other_unparsed | S1T226 |
| 都比躯干短。 | 1 | 都比躯干短 | body_geometry | S4T126 |

#### S218

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子和尾巴都较长，明显长于腿。 | 4 | 明显长于腿 | other_unparsed | S3T119, S3T150, S3T165, S3T215 |
| 脖子、尾巴都长于腿。 | 3 | 脖子、尾巴都长于腿 | other_unparsed | S4T69, S4T74, S4T76 |
| 脖子和尾巴都较短，明显短于腿。 | 3 | 明显短于腿 | other_unparsed | S3T93, S3T118, S4T167 |
| 头、尾巴长度明显长于脖子、腿。 | 2 | 头、尾巴长度明显长于脖子、腿 | other_unparsed | S3T145, S3T151 |
| 尾巴明显长于其余三部位。 | 2 | 尾巴明显长于其余三部位 | other_reference | S4T282, S4T311 |
| 尾巴长度明显长于脖子。 | 2 | 尾巴长度明显长于脖子 | other_unparsed | S4T27, S4T28 |
| 脖子、尾巴较长，且明显长于腿。 | 2 | 且明显长于腿 | other_unparsed | S3T47, S3T86 |
| 脖子和尾巴较长，明显长于腿。 | 2 | 明显长于腿 | other_unparsed | S3T207, S3T208 |
| 脖子和尾巴都明显长于腿。 | 2 | 脖子和尾巴都明显长于腿 | other_unparsed | S3T193, S4T198 |
| 脖子和尾巴都较短，明显短于头。 | 2 | 明显短于头 | other_unparsed | S3T189, S4T160 |
| 脖子明显长于腿，略长于尾巴。 | 2 | 略长于尾巴 | other_unparsed | S4T127, S4T159 |
| 脖子最长，明显长于尾巴。 | 2 | 明显长于尾巴 | other_unparsed | S3T48, S3T68 |
| 脖子长度明显长于尾巴和腿。 | 2 | 脖子长度明显长于尾巴和腿 | other_unparsed | S3T155, S3T156 |
| 四个部位都略长，长度相近。 | 1 | 长度相近 | other_unparsed | S3T101 |
| 四个部位都较短，且长度较相近。 | 1 | 且长度较相近 | other_unparsed | S3T136 |
| 四个部位都较长，且几乎最长。 | 1 | 且几乎最长 | other_unparsed | S3T164 |
| 四个部位都较长，且尾巴微长于脖子。 | 1 | 且尾巴微长于脖子 | other_unparsed | S4T77 |
| 四个部位都较长，且长度较相近。 | 1 | 且长度较相近 | other_unparsed | S3T137 |
| 四个部位都较长，脖子最长，头，尾巴稍短。 | 1 | 头 | other_unparsed | S2T113 |
| 头、脖子、腿、尾巴长度相近，且长度中等。 | 1 | 且长度中等 | other_unparsed | S1T44 |
| 头和脖子较长，腿和尾巴较短，且几乎最短。 | 1 | 且几乎最短 | other_unparsed | S2T156 |
| 头和脖子较长，腿和尾巴较短，且最短。 | 1 | 且最短 | other_unparsed | S2T135 |
| 头明显长于其他三个部位。 | 1 | 头明显长于其他三个部位 | count_abstract, other_reference | S4T2 |
| 头明显长于其余三部位。 | 1 | 头明显长于其余三部位 | other_reference | S4T283 |
| 头明显长于脖子，略长于尾巴和腿。 | 1 | 略长于尾巴和腿 | other_unparsed | S4T128 |
| 头最短，尾巴轻，尾巴较短，脖子较短。 | 1 | 尾巴轻 | other_unparsed | S2T20 |
| 头最长，脖子、腿、尾巴较短，且几乎最短。 | 1 | 且几乎最短 | other_unparsed | S2T189 |
| 头最长，脖子略长于尾巴，腿。 | 1 | 腿 | other_unparsed | S4T72 |
| 头最长，脖子，腿较短，尾巴较长。 | 1 | 脖子 | other_unparsed | S1T92 |
| 头最长，脖子，腿，稍短。 | 1 | 脖子; 腿; 稍短 | other_unparsed | S1T282 |
| 头略长于其余三部位。 | 1 | 头略长于其余三部位 | other_reference | S3T285 |
| 头达到头最长，脖子略短，腿短于尾巴，且都较短。 | 1 | 且都较短 | other_unparsed | S2T239 |
| 头达到最，头较长，脖子、尾巴较短，腿略长于脖子。 | 1 | 头达到最 | other_unparsed | S2T235 |
| 头，尾巴最短，脖子和腿较长。 | 1 | 头 | other_unparsed | S1T128 |
| 头，尾巴较长，脖子稍短，腿最短。 | 1 | 头 | other_unparsed | S2T225 |
| 头，腿较长。 | 1 | 头 | other_unparsed | S1T293 |
| 尾巴、腿都较长，且略长于脖子。 | 1 | 且略长于脖子 | other_unparsed | S4T32 |
| 尾巴和腿略长于脖子，且都较长。 | 1 | 且都较长 | other_unparsed | S4T52 |
| 尾巴明显长于其他三个部位。 | 1 | 尾巴明显长于其他三个部位 | count_abstract, other_reference | S3T116 |
| 尾巴明显长于头，且长于脖子和腿。 | 1 | 且长于脖子和腿 | other_unparsed | S4T199 |
| 尾巴明显长于脖子，且长于腿。 | 1 | 且长于腿 | other_unparsed | S4T168 |
| 尾巴明显长于脖子，腿。 | 1 | 腿 | other_unparsed | S4T116 |
| 尾巴最短，体腿最长，最长。 | 1 | 最长 | other_unparsed | S1T167 |
| 尾巴最短，其余部位较长，且几乎最长。 | 1 | 且几乎最长 | other_unparsed | S2T55 |
| 尾巴最长，其余部位较短，且几乎最短。 | 1 | 且几乎最短 | other_unparsed | S2T116 |
| 尾巴最长，腿稍短，头，脖子较短。 | 1 | 头 | other_unparsed | S2T130 |
| 尾巴略长于脖子，且都明显短于腿。 | 1 | 且都明显短于腿 | other_unparsed | S3T250 |
| 尾巴约等于腿，远长于脖子。 | 1 | 远长于脖子 | other_unparsed | S4T43 |
| 尾巴长度明显长于脖子、腿。 | 1 | 尾巴长度明显长于脖子、腿 | other_unparsed | S3T157 |
| 尾巴，头最长，腿较短。 | 1 | 尾巴 | other_unparsed | S1T295 |
| 脖子、头、尾巴达到较长，腿最短，最短。 | 1 | 最短 | other_unparsed | S2T90 |
| 脖子、尾巴都明显短于腿，脖子略长。 | 1 | 脖子、尾巴都明显短于腿 | other_unparsed | S3T194 |
| 脖子、尾巴都短于腿。 | 1 | 脖子、尾巴都短于腿 | other_unparsed | S4T73 |
| 脖子、尾巴长度明显长于腿。 | 1 | 脖子、尾巴长度明显长于腿 | other_unparsed | S3T154 |
| 脖子、腿较长，明显长于尾巴。 | 1 | 明显长于尾巴 | other_unparsed | S3T94 |
| 脖子和尾巴明显略短于腿。 | 1 | 脖子和尾巴明显略短于腿 | other_unparsed | S3T203 |
| 脖子和尾巴明显短于其余部位。 | 1 | 脖子和尾巴明显短于其余部位 | other_reference | S4T312 |
| 脖子和尾巴明显长于腿，明显短于腿。 | 1 | 明显短于腿 | other_unparsed | S4T99 |
| 脖子和尾巴最长，明显长于腿。 | 1 | 明显长于腿 | other_unparsed | S4T29 |
| 脖子和尾巴略短于腿，且都较长，长度相近。 | 1 | 且都较长; 长度相近 | other_unparsed | S4T5 |
| 脖子和尾巴较短，明显短于腿。 | 1 | 明显短于腿 | other_unparsed | S3T229 |
| 脖子和尾巴较短，明显短于腿和头。 | 1 | 明显短于腿和头 | other_unparsed | S3T98 |
| 脖子和尾巴都短于腿。 | 1 | 脖子和尾巴都短于腿 | other_unparsed | S4T75 |
| 脖子和尾巴都较长，明显长于腿，头最长。 | 1 | 明显长于腿 | other_unparsed | S3T166 |
| 脖子和尾巴长度明显长于头和腿。 | 1 | 脖子和尾巴长度明显长于头和腿 | other_unparsed | S3T138 |
| 脖子和尾巴，最长，头和腿最短。 | 1 | 脖子和尾巴; 最长 | other_unparsed | S2T143 |
| 脖子和尾巴，略长于腿，头最长。 | 1 | 脖子和尾巴; 略长于腿 | other_unparsed | S3T283 |
| 脖子和腿长度明显长于尾巴。 | 1 | 脖子和腿长度明显长于尾巴 | other_unparsed | S3T153 |
| 脖子明显长于尾巴，且长于腿。 | 1 | 且长于腿 | other_unparsed | S3T127 |
| 脖子明显长于尾巴，尾巴长度约等于腿。 | 1 | 尾巴长度约等于腿 | other_unparsed | S4T6 |
| 脖子最短，其余部位都较长，且几乎最长。 | 1 | 且几乎最长 | other_unparsed | S2T75 |
| 脖子最短，尾巴略长于脖子，且都小于头。 | 1 | 且都小于头 | other_unparsed | S3T129 |
| 脖子最长，尾巴稍短，且明显长于腿。 | 1 | 且明显长于腿 | other_unparsed | S3T78 |
| 脖子最长，明显长于尾巴、腿。 | 1 | 明显长于尾巴、腿 | other_unparsed | S3T95 |
| 脖子最长，长于尾巴，腿稍短。 | 1 | 长于尾巴 | other_unparsed | S3T26 |
| 脖子略长于尾巴，且两者明显短于腿。 | 1 | 且两者明显短于腿 | other_unparsed | S3T173 |
| 脖子略长于尾巴，两者都明显短于腿。 | 1 | 两者都明显短于腿 | other_unparsed | S3T114 |
| 脖子短于尾巴，且短于腿。 | 1 | 且短于腿 | other_unparsed | S4T235 |
| 脖子短于尾巴，两者都短于腿。 | 1 | 两者都短于腿 | other_unparsed | S3T110 |
| 脖子短，尾巴短，且都最短。 | 1 | 且都最短 | other_unparsed | S3T8 |
| 脖子较长，腿，尾巴较长，头较短。 | 1 | 腿 | other_unparsed | S1T272 |
| 脖子长于尾巴，且明显长于腿。 | 1 | 且明显长于腿 | other_unparsed | S3T109 |
| 脖子长度明显短于尾巴，脖子也短于腿。 | 1 | 脖子长度明显短于尾巴; 脖子也短于腿 | other_unparsed | S3T204 |
| 脖子长度约等于腿，且明显长于尾巴。 | 1 | 脖子长度约等于腿; 且明显长于尾巴 | other_unparsed | S4T25 |
| 脖子长约等于腿，且明显大于尾巴。 | 1 | 脖子长约等于腿; 且明显大于尾巴 | other_unparsed | S4T134 |
| 腿最短，脖子、头、尾巴都较长，且几乎最长。 | 1 | 且几乎最长 | other_unparsed | S2T65 |
| 腿最短，脖子，尾巴较长。 | 1 | 脖子 | other_unparsed | S1T109 |
| 腿最长，头略短，脖子略长于尾巴，且都较短。 | 1 | 且都较短 | other_unparsed | S2T236 |
| 腿最长，头，尾巴较短。 | 1 | 头 | other_unparsed | S1T173 |
| 腿略短于其余三部位。 | 1 | 腿略短于其余三部位 | other_reference | S4T314 |
| 长度较相近且长度中等。 | 1 | 长度较相近且长度中等 | other_unparsed | S1T231 |

#### S219

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 比较均衡。 | 19 | 比较均衡 | global_balance | S1T63, S1T69, S1T75, S1T76, S1T78, S1T160, S1T189, S1T256 |
| 均衡。 | 3 | 均衡 | global_balance | S2T49, S2T104, S2T123 |
| 头和尾巴。 | 3 | 头和尾巴 | other_unparsed | S1T93, S1T124, S1T309 |
| 稍微短。 | 3 | 稍微短 | other_unparsed | S3T234, S3T235, S3T237 |
| 选错了。 | 2 | 选错了 | meta_or_uncertain | S3T97, S3T137 |
| 四个部位都中等，比较均匀。 | 1 | 比较均匀 | global_balance | S3T23 |
| 头位呀。 | 1 | 头位呀 | other_unparsed | S3T246 |
| 头和尾巴明显比脖子和腿长。 | 1 | 头和尾巴明显比脖子和腿长 | other_unparsed | S3T3 |
| 头和脖子很小。 | 1 | 头和脖子很小 | vague_size | S3T50 |
| 头和腿很短，其他很长，特别是尾巴。 | 1 | 特别是尾巴 | other_unparsed | S3T102 |
| 头是最长，明显比腿长很多。 | 1 | 明显比腿长很多 | other_unparsed | S3T8 |
| 头比较大。 | 1 | 头比较大 | other_unparsed | S1T84 |
| 头还行，腿很短。 | 1 | 头还行 | other_unparsed | S1T107 |
| 头长，脖子短，腿还行。 | 1 | 腿还行 | other_unparsed | S1T102 |
| 比较均衡，头有点儿短。 | 1 | 比较均衡 | global_balance | S1T98 |
| 比较均衡，头有点短。 | 1 | 比较均衡 | global_balance | S3T207 |
| 比较均衡，尾巴有点长。 | 1 | 比较均衡 | global_balance | S1T221 |
| 比较均衡，脖子有点儿短。 | 1 | 比较均衡 | global_balance | S1T264 |
| 比较均衡，脖子短。 | 1 | 比较均衡 | global_balance | S1T291 |
| 脖子、尾巴都。 | 1 | 脖子、尾巴都 | other_unparsed | S2T167 |
| 脖子。 | 1 | 脖子 | other_unparsed | S2T208 |
| 脖子和尾巴都。 | 1 | 脖子和尾巴都 | other_unparsed | S2T183 |
| 脖子和腿偏短，其他偏长，没有很长。 | 1 | 没有很长 | other_unparsed | S3T100 |
| 都不长，比较均衡。 | 1 | 都不长; 比较均衡 | global_balance | S3T280 |
| 都中等。 | 1 | 都中等 | other_unparsed | S3T262 |
| 都比较均衡，脖子和尾巴短。 | 1 | 都比较均衡 | global_balance | S1T195 |

#### S220

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和尾巴不一样长，且腿长。 | 3 | 头和尾巴不一样长 | disjoint_inequality | S3T262, S3T263, S3T266 |
| 脖子和尾巴长于腿，长于头。 | 3 | 长于头 | other_unparsed | S3T68, S3T69, S3T70 |
| 头、脖子和腿都比尾巴长。 | 2 | 头、脖子和腿都比尾巴长 | other_unparsed | S3T84, S3T101 |
| 头和尾巴一样长，且脖子和腿不一样长。 | 2 | 且脖子和腿不一样长 | disjoint_inequality | S3T167, S3T169 |
| 三个部位长于腿。 | 1 | 三个部位长于腿 | count_abstract | S3T33 |
| 四部位差不多长。 | 1 | 四部位差不多长 | global_balance | S3T260 |
| 头、尾巴和腿长，头和尾巴，脖子、尾巴和腿长，头短。 | 1 | 头和尾巴 | other_unparsed | S1T68 |
| 头、脖子、尾巴一样长，长于腿。 | 1 | 长于腿 | other_unparsed | S4T33 |
| 头、脖子、尾巴和腿都比尾巴短。 | 1 | 头、脖子、尾巴和腿都比尾巴短 | other_unparsed | S3T83 |
| 头、脖子、尾巴都很长，且长度差不多。 | 1 | 且长度差不多 | global_balance | S1T49 |
| 头、脖子、尾巴长于四条腿。 | 1 | 头、脖子、尾巴长于四条腿 | other_unparsed | S4T35 |
| 头、脖子、腿，和尾巴差不多长。 | 1 | 头、脖子、腿 | other_unparsed | S2T73 |
| 头、脖子和尾巴差不多长，长于腿。 | 1 | 长于腿 | other_unparsed | S3T54 |
| 头、脖子和尾巴比，头、腿和尾巴比脖子长。 | 1 | 头、脖子和尾巴比 | other_unparsed | S3T96 |
| 头、脖子和尾巴都很长，长于腿。 | 1 | 长于腿 | other_unparsed | S3T53 |
| 头、脖子和尾巴长度相等，都比腿长。 | 1 | 都比腿长 | other_unparsed | S3T13 |
| 头、脖子和腿都会比尾巴更长。 | 1 | 头、脖子和腿都会比尾巴更长 | other_unparsed | S2T247 |
| 头、脖子和腿都是尾巴长度的至少两倍。 | 1 | 头、脖子和腿都是尾巴长度的至少两倍 | other_unparsed | S1T226 |
| 头、脖子和腿长度较长，比尾巴长很多。 | 1 | 比尾巴长很多 | other_unparsed | S3T14 |
| 头、脖子长，腿中等，较短，尾巴短。 | 1 | 较短 | other_unparsed | S1T253 |
| 头、腿和尾巴都是脖子长度的至少两倍。 | 1 | 头、腿和尾巴都是脖子长度的至少两倍 | other_unparsed | S1T225 |
| 头、腿和尾巴都比脖子长，脖子最短。 | 1 | 头、腿和尾巴都比脖子长 | other_unparsed | S1T89 |
| 头和尾巴一样短，脖子和腿最长，且一样长。 | 1 | 且一样长 | other_unparsed | S1T92 |
| 头和尾巴不一样长，且脖子和腿一样长。 | 1 | 头和尾巴不一样长 | disjoint_inequality | S3T264 |
| 头和尾巴不一样长，且腿短。 | 1 | 头和尾巴不一样长 | disjoint_inequality | S3T265 |
| 头和尾巴中等偏长一点，腿中等偏短一点，脖子。 | 1 | 脖子 | other_unparsed | S2T149 |
| 头和脖子一样长，比腿和尾巴都长。 | 1 | 比腿和尾巴都长 | other_unparsed | S1T47 |
| 头和脖子一样长，腿最短，尾巴中等，较短。 | 1 | 较短 | other_unparsed | S1T71 |
| 头和脖子中等，腿比中等短一点点，尾巴较长。 | 1 | 腿比中等短一点点 | other_unparsed | S2T108 |
| 头和脖子中等，较短，尾巴短，腿较长。 | 1 | 较短 | other_unparsed | S2T36 |
| 头和脖子较短，尾巴和腿较长，中等。 | 1 | 中等 | other_unparsed | S2T132 |
| 头和脖子长，尾巴和腿。 | 1 | 尾巴和腿 | other_unparsed | S3T109 |
| 头和腿长，脖子和尾巴。 | 1 | 脖子和尾巴 | other_unparsed | S3T259 |
| 头最长，四个都不一样长。 | 1 | 四个都不一样长 | disjoint_inequality | S1T46 |
| 头短，尾巴短，脖子和腿差不多长，都是中等长度。 | 1 | 都是中等长度 | other_unparsed | S2T6 |
| 头超长，尾巴超长，脖子较短，腿中等，躯干长度的一半。 | 1 | 躯干长度的一半 | body_geometry | S1T2 |
| 头长，脖子，尾巴和腿短。 | 1 | 脖子 | other_unparsed | S1T111 |
| 头长，腿中腿差不多，腿中等长，脖子和尾巴短。 | 1 | 腿中腿差不多 | global_balance | S1T67 |
| 尾巴和脖子中等，头和腿。 | 1 | 头和腿 | other_unparsed | S2T30 |
| 尾巴和腿最长，比头和脖子长。 | 1 | 比头和脖子长 | other_unparsed | S2T294 |
| 尾巴和腿，和头较长，脖子中等。 | 1 | 尾巴和腿 | other_unparsed | S2T112 |
| 尾巴大于腿，大于头，大于脖子。 | 1 | 大于头; 大于脖子 | other_unparsed | S3T66 |
| 尾巴明显短于其他三个。 | 1 | 尾巴明显短于其他三个 | other_reference | S2T299 |
| 尾巴长于头和腿，长于脖子。 | 1 | 长于脖子 | other_unparsed | S3T76 |
| 尾巴长于头，长于脖子，长于腿。 | 1 | 长于脖子; 长于腿 | other_unparsed | S3T75 |
| 尾巴长于脖子长于头，和腿。 | 1 | 和腿 | other_unparsed | S4T34 |
| 差不多长。 | 1 | 差不多长 | global_balance | S3T237 |
| 我说脖子明显短于其他三个。 | 1 | 我说脖子明显短于其他三个 | other_reference | S2T300 |
| 有两个部位显著的短于另外两个部位。 | 1 | 有两个部位显著的短于另外两个部位 | count_abstract, other_reference | S3T81 |
| 脖子、头，脖子和尾巴和腿差不多长。 | 1 | 脖子、头 | other_unparsed | S2T287 |
| 脖子、尾巴和头，腿长稍短。 | 1 | 脖子、尾巴和头 | other_unparsed | S1T216 |
| 脖子和尾巴一样长，长于头和腿。 | 1 | 长于头和腿 | other_unparsed | S3T230 |
| 脖子和尾巴中等，尾巴比中等长一点点，头中等，腿较短。 | 1 | 尾巴比中等长一点点 | other_unparsed | S2T114 |
| 脖子和尾巴中等，腿比中等稍长一点点，头比中等稍短一点点。 | 1 | 腿比中等稍长一点点; 头比中等稍短一点点 | other_unparsed | S2T113 |
| 脖子和尾巴长度差不多，比腿和头都长。 | 1 | 比腿和头都长 | other_unparsed | S3T12 |
| 脖子和尾巴，头和腿最短。 | 1 | 脖子和尾巴 | other_unparsed | S2T70 |
| 脖子和腿一样长，比头和尾巴长。 | 1 | 比头和尾巴长 | other_unparsed | S3T295 |
| 脖子和腿中等，头和尾巴比中等偏长一点点。 | 1 | 头和尾巴比中等偏长一点点 | other_unparsed | S2T41 |
| 脖子和腿都比头和尾巴长。 | 1 | 脖子和腿都比头和尾巴长 | other_unparsed | S3T85 |
| 脖子和腿，最长头和尾巴最短。 | 1 | 脖子和腿 | other_unparsed | S1T135 |
| 脖子和腿，最长，头和尾巴最短。 | 1 | 脖子和腿; 最长 | other_unparsed | S2T135 |
| 脖子显著的长于其他三个。 | 1 | 脖子显著的长于其他三个 | other_reference | S2T298 |
| 脖子显著短于其他三个部位。 | 1 | 脖子显著短于其他三个部位 | count_abstract, other_reference | S3T44 |
| 脖子显著长于其他三个。 | 1 | 脖子显著长于其他三个 | other_reference | S2T272 |
| 脖子长于三其他三个部位。 | 1 | 脖子长于三其他三个部位 | count_abstract, other_reference | S4T100 |
| 脖子长于头，长于腿和尾巴。 | 1 | 长于腿和尾巴 | other_unparsed | S3T74 |
| 腿会比头长很多。 | 1 | 腿会比头长很多 | other_unparsed | S2T249 |
| 腿和尾巴一样长，比头和脖子长。 | 1 | 比头和脖子长 | other_unparsed | S3T294 |
| 腿和尾巴一样长，长于脖子和头。 | 1 | 长于脖子和头 | other_unparsed | S3T225 |
| 腿和尾巴长，脖子和头中。 | 1 | 脖子和头中 | other_unparsed | S1T256 |
| 腿和尾巴，头中等，脖子较短。 | 1 | 腿和尾巴 | other_unparsed | S2T182 |
| 腿显著的短于脖子、头和尾巴。 | 1 | 腿显著的短于脖子、头和尾巴 | other_unparsed | S3T82 |
| 腿最短，脖子、尾巴、头都比它长。 | 1 | 脖子、尾巴、头都比它长 | other_unparsed | S3T317 |
| 腿最长，脖子和尾巴中等，比腿稍微短一点点，头短。 | 1 | 比腿稍微短一点点 | other_unparsed | S1T227 |
| 腿最长，长于尾巴。 | 1 | 长于尾巴 | other_unparsed | S3T231 |
| 腿比较长，头、尾巴和脖子都比腿短很多。 | 1 | 头、尾巴和脖子都比腿短很多 | other_unparsed | S2T242 |

#### S221

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子和尾巴相差比较大，腿和头相差比较小。 | 1 | 脖子和尾巴相差比较大; 腿和头相差比较小 | vague_size | S1T125 |
| 脖子，最长。 | 1 | 脖子; 最长 | other_unparsed | S1T311 |
| 腿不是最长，头不是最短。 | 1 | 腿不是最长; 头不是最短 | other_unparsed | S1T158 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S2T85 |

#### S222

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头很大。 | 4 | 头很大 | other_unparsed | S2T125, S2T155, S2T162, S2T243 |
| 体型中等，尾巴和脖子比腿长。 | 3 | 体型中等 | vague_size | S2T60, S2T80, S2T241 |
| 体型中小。 | 2 | 体型中小 | vague_size | S2T317, S2T319 |
| 体型中等，四个部位差不多长。 | 2 | 体型中等 | vague_size | S2T151, S2T170 |
| 体型中等，腿长，尾巴短。 | 2 | 体型中等 | vague_size | S2T191, S2T204 |
| 体型大，四个部位都差不多。 | 2 | 体型大 | vague_size | S2T188, S2T198 |
| 体型小，头很长。 | 2 | 体型小 | vague_size | S2T254, S2T255 |
| 四个部位差不多长，体型偏大。 | 2 | 体型偏大 | vague_size | S2T161, S2T172 |
| 头和脖子比腿长，体型比较小。 | 2 | 体型比较小 | vague_size | S2T8, S2T13 |
| 尾巴长，体型小。 | 2 | 体型小 | vague_size | S2T208, S2T306 |
| 三个部位短，一个部位长。 | 1 | 三个部位短; 一个部位长 | count_abstract | S1T74 |
| 两个部位很长，一个部位比较长，一个部位比较短。 | 1 | 两个部位很长; 一个部位比较长; 一个部位比较短 | count_abstract | S1T23 |
| 两个部位很长，两个部位很短。 | 1 | 两个部位很长; 两个部位很短 | count_abstract | S1T39 |
| 两个部位最短，一个部位比较长，一个部位比较短。 | 1 | 两个部位最短; 一个部位比较长; 一个部位比较短 | count_abstract | S1T18 |
| 两个部位等长，一个，其他两个部位一长一短。 | 1 | 两个部位等长; 一个; 其他两个部位一长一短 | count_abstract, other_reference | S1T31 |
| 体型中小，头长。 | 1 | 体型中小 | vague_size | S2T244 |
| 体型中小，尾巴和腿差不多。 | 1 | 体型中小 | vague_size | S2T120 |
| 体型中等，四个部位都差不多。 | 1 | 体型中等 | vague_size | S2T192 |
| 体型中等，四个部位长度都差不多。 | 1 | 体型中等 | vague_size | S2T41 |
| 体型中等，头很长。 | 1 | 体型中等 | vague_size | S2T107 |
| 体型中等，头最长。 | 1 | 体型中等 | vague_size | S2T315 |
| 体型中等，小头比腿长。 | 1 | 体型中等 | vague_size | S2T133 |
| 体型中等，尾巴、头、腿差不多长。 | 1 | 体型中等 | vague_size | S2T123 |
| 体型中等，尾巴和脖子比头和腿长。 | 1 | 体型中等 | vague_size | S2T186 |
| 体型中等，尾巴和脖子比腿。 | 1 | 体型中等; 尾巴和脖子比腿 | vague_size | S2T119 |
| 体型中等，尾巴和腿差不多，脖子很长。 | 1 | 体型中等 | vague_size | S2T178 |
| 体型中等，尾巴和腿长。 | 1 | 体型中等 | vague_size | S2T231 |
| 体型中等，尾巴最长。 | 1 | 体型中等 | vague_size | S2T105 |
| 体型中等，尾巴长。 | 1 | 体型中等 | vague_size | S2T263 |
| 体型中等，尾巴长，腿短。 | 1 | 体型中等 | vague_size | S2T154 |
| 体型中等，腿最短，其他差不多。 | 1 | 体型中等 | vague_size | S2T229 |
| 体型中等，腿长。 | 1 | 体型中等 | vague_size | S2T101 |
| 体型中等，腿长，其他差不多。 | 1 | 体型中等 | vague_size | S2T134 |
| 体型偏中大。 | 1 | 体型偏中大 | vague_size | S2T114 |
| 体型偏中小，腿和尾巴差不多一样长。 | 1 | 体型偏中小 | vague_size | S2T56 |
| 体型偏中等，头、腿长，尾巴短。 | 1 | 体型偏中等 | vague_size | S2T62 |
| 体型偏大，头和腿比较长，尾巴比腿短。 | 1 | 体型偏大 | vague_size | S2T124 |
| 体型偏大，尾巴和腿差不多一样。 | 1 | 体型偏大 | vague_size | S2T177 |
| 体型偏大，尾巴和腿差不多一样长。 | 1 | 体型偏大 | vague_size | S2T57 |
| 体型偏小，尾巴长。 | 1 | 体型偏小 | vague_size | S2T59 |
| 体型偏小，脖子和头相对较长。 | 1 | 体型偏小 | vague_size | S2T179 |
| 体型偏小，脖子和尾巴差不多，腿最短。 | 1 | 体型偏小 | vague_size | S2T99 |
| 体型偏小，脖子比尾巴长。 | 1 | 体型偏小 | vague_size | S2T98 |
| 体型大，头很大。 | 1 | 体型大; 头很大 | vague_size | S2T196 |
| 体型大，腿长，尾巴。 | 1 | 体型大; 尾巴 | vague_size | S2T294 |
| 体型小。 | 1 | 体型小 | vague_size | S2T92 |
| 体型小，尾巴和脖子比头和腿长。 | 1 | 体型小 | vague_size | S2T185 |
| 体型很大，尾巴最长。 | 1 | 体型很大 | vague_size | S2T104 |
| 体型比较大，尾巴比较短。 | 1 | 体型比较大 | vague_size | S1T68 |
| 体型比较小。 | 1 | 体型比较小 | vague_size | S2T14 |
| 哪个部位长，哪个部位比较长，哪个部位比较短。 | 1 | 哪个部位长; 哪个部位比较长; 哪个部位比较短 | other_unparsed | S1T19 |
| 四个部位很都不等长。 | 1 | 四个部位很都不等长 | other_unparsed | S1T33 |
| 四个部位都差不多，体型中等。 | 1 | 体型中等 | vague_size | S2T234 |
| 四个部位都很小。 | 1 | 四个部位都很小 | vague_size | S1T95 |
| 头和尾巴差不多，体型偏小。 | 1 | 体型偏小 | vague_size | S2T168 |
| 头和腿，尾巴和腿差不多一样长。 | 1 | 头和腿 | other_unparsed | S2T47 |
| 头和腿，差不多长。 | 1 | 头和腿; 差不多长 | global_balance | S2T90 |
| 头最长，体型小。 | 1 | 体型小 | vague_size | S2T307 |
| 头比脖子长，腿最长，尾巴也差不多。 | 1 | 尾巴也差不多 | global_balance | S1T140 |
| 头比脖子长，较长。 | 1 | 较长 | other_unparsed | S1T146 |
| 头长，体型小。 | 1 | 体型小 | vague_size | S2T165 |
| 尾巴和头长，头和尾巴，腿和脖子比较短。 | 1 | 头和尾巴 | other_unparsed | S1T132 |
| 尾巴和脖子比头和腿。 | 1 | 尾巴和脖子比头和腿 | other_unparsed | S2T237 |
| 尾巴和脖子比腿长，体型偏小。 | 1 | 体型偏小 | vague_size | S2T108 |
| 尾巴很长，体型小。 | 1 | 体型小 | vague_size | S2T257 |
| 尾巴最长，体型偏小。 | 1 | 体型偏小 | vague_size | S2T259 |
| 尾巴最长，体型小。 | 1 | 体型小 | vague_size | S2T301 |
| 尾巴短，腿长，体型中等。 | 1 | 体型中等 | vague_size | S2T29 |
| 有三个部位长，面条左边脖子比较短。 | 1 | 有三个部位长 | count_abstract | S1T10 |
| 脖子和尾巴长，体型偏小。 | 1 | 体型偏小 | vague_size | S2T64 |
| 脖子和腿比，脖子和尾巴比腿长，头很长。 | 1 | 脖子和腿比 | other_unparsed | S2T7 |
| 脖子长，体型偏小。 | 1 | 体型偏小 | vague_size | S2T167 |
| 腿和头，中等长，脖子最短，尾巴最长。 | 1 | 腿和头; 中等长 | other_unparsed | S1T36 |
| 腿最短，其他比。 | 1 | 其他比 | other_reference | S2T82 |
| 腿短，头短，肩部为长。 | 1 | 肩部为长 | other_unparsed | S1T78 |
| 腿长，尾巴短，体型偏大。 | 1 | 体型偏大 | vague_size | S2T70 |
| 腿长，尾巴短，体型比较小。 | 1 | 体型比较小 | vague_size | S2T4 |
| 腿长，尾巴，头和脖子差不多。 | 1 | 尾巴 | other_unparsed | S1T103 |
| 腿长，脖子长，头，它尾巴比较短。 | 1 | 头 | other_unparsed | S1T86 |
| 该体型比较小，脖子和尾巴等长，头短一点。 | 1 | 该体型比较小 | vague_size | S1T67 |
| 较短。 | 1 | 较短 | other_unparsed | S1T55 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S2T173 |
| 面朝右边，四个部位差不多长。 | 1 | 面朝右边 | other_unparsed | S1T38 |
| 面朝右边，头和尾巴比差不多一样长，比腿和脖子短。 | 1 | 面朝右边; 比腿和脖子短 | other_unparsed | S1T4 |
| 面朝右边，头和腿，差等长，尾巴长，脖子短。 | 1 | 面朝右边; 头和腿; 差等长 | other_unparsed | S1T11 |
| 面朝右边，脖子比较短，其他三个部位比较长。 | 1 | 面朝右边 | other_unparsed | S1T15 |
| 面朝右边，腿和尾巴比较长，头和脖子一长一短。 | 1 | 面朝右边 | other_unparsed | S1T30 |
| 面朝右边，腿比较短，其他三个部位比较长。 | 1 | 面朝右边 | other_unparsed | S1T13 |
| 面朝左边，两个部位比较长，一个部位更长，一个部位更短。 | 1 | 面朝左边; 两个部位比较长; 一个部位更长; 一个部位更短 | count_abstract | S1T17 |
| 面朝左边，四个部位都挺长。 | 1 | 面朝左边 | other_unparsed | S1T34 |
| 面朝左边，头、腿、尾巴比脖子长。 | 1 | 面朝左边 | other_unparsed | S1T12 |
| 面朝左边，尾巴比较短，其他三个部位比较长。 | 1 | 面朝左边 | other_unparsed | S1T14 |
| 面朝左边，脖子和尾巴比其他两个部位长。 | 1 | 面朝左边 | other_unparsed | S1T1 |
| 面朝左边，脖子和腿一样长，头比较长，尾巴就比较短。 | 1 | 面朝左边 | other_unparsed | S1T28 |
| 面朝左边，脖子和腿一样长，头比较长，尾巴比较短。 | 1 | 面朝左边 | other_unparsed | S1T29 |
| 面朝左边，腿是最长，脖子比其他两个部位长。 | 1 | 面朝左边 | other_unparsed | S1T3 |
| 面朝左边，腿特别长，其他部位差不多一样长。 | 1 | 面朝左边 | other_unparsed | S1T2 |

#### S223

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头适中，脖子短，腿适中，尾巴短，头比脖子长，头也比腿短。 | 1 | 头也比腿短 | other_unparsed | S2T38 |
| 头适中，脖子短，腿长，尾巴短，脖子比腿短，头也比腿短。 | 1 | 头也比腿短 | other_unparsed | S2T28 |

#### S224

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 三个部位均较短，腿和脖子较长。 | 1 | 三个部位均较短 | count_abstract | S2T196 |
| 三个部位均较长，脖子最短。 | 1 | 三个部位均较长 | count_abstract | S2T123 |
| 三个部位长度适中，尾巴稍微短一些。 | 1 | 三个部位长度适中 | count_abstract | S3T1 |
| 只有头较短，脖子相当长。 | 1 | 脖子相当长 | global_balance | S2T13 |
| 只有尾巴较短，差不多。 | 1 | 差不多 | global_balance | S2T82 |
| 只有腿，腿最长。 | 1 | 只有腿 | other_unparsed | S3T309 |
| 四个部位都差不多长，腿最长，长度都适中。 | 1 | 长度都适中 | other_unparsed | S1T259 |
| 头、脖子和尾巴较短，腿居中。 | 1 | 腿居中 | other_unparsed | S2T314 |
| 头和脖子明显比尾巴和腿长。 | 1 | 头和脖子明显比尾巴和腿长 | other_unparsed | S3T293 |
| 头和脖子明显比腿、尾巴长。 | 1 | 头和脖子明显比腿、尾巴长 | other_unparsed | S3T275 |
| 头和脖子明显比腿、尾巴长，头最长。 | 1 | 头和脖子明显比腿、尾巴长 | other_unparsed | S3T250 |
| 头和脖子明显比腿和尾巴长。 | 1 | 头和脖子明显比腿和尾巴长 | other_unparsed | S3T138 |
| 头和脖子相对。 | 1 | 头和脖子相对 | other_unparsed | S1T112 |
| 头和脖子较短，其余居中。 | 1 | 其余居中 | other_reference | S3T125 |
| 头和脖子较短，比例较为协调。 | 1 | 比例较为协调 | proportion_or_ratio | S2T242 |
| 头和脖子非常长，比腿和尾巴长得多。 | 1 | 比腿和尾巴长得多 | other_unparsed | S2T3 |
| 头最长，尾巴，腿很短。 | 1 | 尾巴 | other_unparsed | S1T87 |
| 头比尾巴长，比脖子长。 | 1 | 比脖子长 | other_unparsed | S1T82 |
| 头比脖子长，比尾巴长。 | 1 | 比尾巴长 | other_unparsed | S1T155 |
| 脖子比其他三个部位来说较短，也适中长。 | 1 | 也适中长 | other_unparsed | S3T93 |
| 脖子长，其他都比较短，腿还可以。 | 1 | 腿还可以 | other_unparsed | S1T302 |
| 腿明显短于其他三个部位。 | 1 | 腿明显短于其他三个部位 | count_abstract, other_reference | S2T50 |
| 腿最长，其余长度协调。 | 1 | 其余长度协调 | other_reference, proportion_or_ratio | S1T307 |
| 腿极短，比头短，尾巴短。 | 1 | 比头短 | other_unparsed | S3T50 |
| 腿比脖子短，比尾巴短。 | 1 | 比尾巴短 | other_unparsed | S2T35 |
| 腿较，尾巴和脖子较短，头较短。 | 1 | 腿较 | other_unparsed | S2T95 |
| 腿，腿极短。 | 1 | 腿 | other_unparsed | S4T45 |
| 长度均适中，头和腿较短一些。 | 1 | 长度均适中 | other_unparsed | S2T112 |

#### S225

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头远比脖子长。 | 1 | 头远比脖子长 | other_unparsed | S1T240 |
| 脖子远比头长。 | 1 | 脖子远比头长 | other_unparsed | S1T239 |

#### S226

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 两长两短。 | 12 | 两长两短 | count_abstract | S1T46, S1T48, S1T49, S1T52, S1T56, S1T58, S1T61, S1T65 |
| 三长一短。 | 11 | 三长一短 | count_abstract | S1T26, S1T32, S1T45, S1T50, S1T53, S1T57, S1T66, S1T74 |
| 三个差不多长。 | 4 | 三个差不多长 | global_balance | S1T67, S1T72, S1T78, S1T151 |
| 三短一长。 | 3 | 三短一长 | other_unparsed | S1T44, S1T59, S1T262 |
| 两短两长。 | 3 | 两短两长 | other_unparsed | S1T25, S1T28, S1T258 |
| 三个都短，只有腿长。 | 2 | 三个都短 | other_unparsed | S1T113, S1T115 |
| 三长一短，尾巴短。 | 2 | 三长一短 | count_abstract | S1T27, S1T29 |
| 四个差不多，像马。 | 2 | 像马 | other_unparsed | S1T91, S1T132 |
| 四个差不多，都挺长。 | 2 | 都挺长 | other_unparsed | S1T268, S1T269 |
| 头比脖子长，比尾巴长。 | 2 | 比尾巴长 | other_unparsed | S2T27, S2T29 |
| 尾巴短，两个长。 | 2 | 两个长 | other_unparsed | S1T275, S1T277 |
| 有一个很短。 | 2 | 有一个很短 | other_unparsed | S1T54, S1T55 |
| 一个特别长。 | 1 | 一个特别长 | other_unparsed | S1T60 |
| 一个长，腿短。 | 1 | 一个长 | other_unparsed | S1T109 |
| 一长三短。 | 1 | 一长三短 | other_unparsed | S1T51 |
| 三个差不多。 | 1 | 三个差不多 | global_balance | S1T82 |
| 三个比较长，头短。 | 1 | 三个比较长 | other_unparsed | S1T181 |
| 三个比较长，腿短。 | 1 | 三个比较长 | other_unparsed | S1T182 |
| 三个短，只有头长。 | 1 | 三个短 | other_unparsed | S1T187 |
| 三个长，只有头短。 | 1 | 三个长 | other_unparsed | S1T185 |
| 三个长，尾巴短。 | 1 | 三个长 | other_unparsed | S1T86 |
| 三个长，有一个短。 | 1 | 三个长; 有一个短 | other_unparsed | S1T129 |
| 三短一长，脖子长。 | 1 | 三短一长 | other_unparsed | S1T34 |
| 三短一长，腿短。 | 1 | 三短一长 | other_unparsed | S1T33 |
| 三短一长，腿短，尾巴短。 | 1 | 三短一长 | other_unparsed | S1T75 |
| 三长一短，头短。 | 1 | 三长一短 | count_abstract | S1T35 |
| 不知道什么规律，随便选的。 | 1 | 不知道什么规律; 随便选的 | meta_or_uncertain | S1T127 |
| 两个差不多长。 | 1 | 两个差不多长 | global_balance | S1T69 |
| 两个很长。 | 1 | 两个很长 | other_unparsed | S1T85 |
| 两短一长。 | 1 | 两短一长 | other_unparsed | S1T24 |
| 两长两大。 | 1 | 两长两大 | other_unparsed | S1T264 |
| 两长两短，尾巴短。 | 1 | 两长两短 | count_abstract | S1T76 |
| 像马，四个都差不多。 | 1 | 像马 | other_unparsed | S1T90 |
| 像马，腿长。 | 1 | 像马 | other_unparsed | S1T112 |
| 四个依次变化。 | 1 | 四个依次变化 | other_unparsed | S1T42 |
| 四个差不多长，像马。 | 1 | 像马 | other_unparsed | S1T107 |
| 四个差不多长，都挺长。 | 1 | 都挺长 | other_unparsed | S1T188 |
| 四个差不多，加起来都挺长。 | 1 | 加起来都挺长 | proportion_or_ratio | S1T245 |
| 四个差不多，都长。 | 1 | 都长 | other_unparsed | S1T270 |
| 小马。 | 1 | 小马 | other_unparsed | S1T110 |
| 尾巴短，一个长。 | 1 | 一个长 | other_unparsed | S1T288 |
| 尾巴短，加起来短。 | 1 | 加起来短 | proportion_or_ratio | S1T254 |
| 尾巴长，一个短。 | 1 | 一个短 | other_unparsed | S1T291 |
| 尾巴长，两个长。 | 1 | 两个长 | other_unparsed | S1T292 |
| 尾巴长，有一个短。 | 1 | 有一个短 | other_unparsed | S1T287 |
| 差不多都挺长。 | 1 | 差不多都挺长 | other_unparsed | S1T234 |
| 差不多长。 | 1 | 差不多长 | global_balance | S1T77 |
| 差不多，头有点短。 | 1 | 差不多 | global_balance | S1T276 |
| 有点短。 | 1 | 有点短 | other_unparsed | S2T133 |
| 点错了。 | 1 | 点错了 | other_unparsed | S1T96 |
| 真的一长。 | 1 | 真的一长 | other_unparsed | S1T267 |
| 腿、尾巴短，两个长。 | 1 | 两个长 | other_unparsed | S1T281 |
| 腿很短，比较像狗。 | 1 | 比较像狗 | other_unparsed | S1T92 |
| 腿挺长，像马。 | 1 | 像马 | other_unparsed | S1T128 |
| 腿短三个长，加起来还挺长。 | 1 | 加起来还挺长 | proportion_or_ratio | S1T247 |
| 腿短，一个长。 | 1 | 一个长 | other_unparsed | S1T201 |
| 腿短，但有三个长。 | 1 | 但有三个长 | other_unparsed | S1T104 |
| 腿短，和尾巴差不多，另外两个很长。 | 1 | 和尾巴差不多 | global_balance | S1T137 |
| 腿短，脖子短，加起来一般。 | 1 | 加起来一般 | proportion_or_ratio | S1T253 |
| 腿长，三个长一个短。 | 1 | 三个长一个短 | other_unparsed | S1T138 |
| 腿长，两个短。 | 1 | 两个短 | other_unparsed | S1T200 |
| 腿长，像马。 | 1 | 像马 | other_unparsed | S1T150 |
| 腿长，其他两个短，一个长。 | 1 | 一个长 | other_unparsed | S1T206 |
| 腿长，另外三个差不多，像马。 | 1 | 像马 | other_unparsed | S1T126 |
| 腿长，尾巴长，两个短。 | 1 | 两个短 | other_unparsed | S1T285 |
| 这个差不多。 | 1 | 这个差不多 | global_balance | S1T81 |
| 这个差不多长，像马。 | 1 | 这个差不多长; 像马 | global_balance | S1T106 |
| 都挺长，但头比较短。 | 1 | 都挺长 | other_unparsed | S1T246 |
| 都挺长，但头短。 | 1 | 都挺长 | other_unparsed | S1T249 |
| 都挺长，加起来挺长。 | 1 | 都挺长; 加起来挺长 | proportion_or_ratio | S1T248 |
| 都比身子短。 | 1 | 都比身体短 | body_geometry | S2T201 |
| 都短，只有一个脖子长。 | 1 | 都短 | other_unparsed | S1T174 |
| 都跟躯干差不多长。 | 1 | 都跟躯干差不多长 | body_geometry, global_balance | S2T210 |

#### S227

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 四个部位长度各不相同。 | 7 | 四个部位长度各不相同 | disjoint_inequality | S1T97, S1T98, S1T100, S1T101, S1T103, S1T132, S1T133 |
| 四个部位长度不一。 | 2 | 四个部位长度不一 | other_unparsed | S1T50, S1T148 |
| 四个部位长度不同。 | 2 | 四个部位长度不同 | disjoint_inequality | S1T114, S1T115 |
| 尾巴和腿长度近似。 | 2 | 尾巴和腿长度近似 | other_unparsed | S1T96, S1T171 |
| 脖子和腿长度近似。 | 2 | 脖子和腿长度近似 | other_unparsed | S1T92, S1T173 |
| 头和脖子长度近似。 | 1 | 头和脖子长度近似 | other_unparsed | S1T95 |
| 尾巴长，头，腿和脖子比较短。 | 1 | 头 | other_unparsed | S1T22 |
| 有两个部位的长度超过了躯干。 | 1 | 有两个部位的长度超过了躯干 | count_abstract, body_geometry | S1T107 |
| 脖子短，其余三个部位较长，且差不多长。 | 1 | 且差不多长 | global_balance | S1T53 |
| 脖子长度和躯干近似。 | 1 | 脖子长度和躯干近似 | body_geometry | S1T93 |
| 腿长，四个部位长度各不相同。 | 1 | 四个部位长度各不相同 | disjoint_inequality | S1T94 |

#### S228

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 比较均匀。 | 10 | 比较均匀 | global_balance | S1T146, S2T79, S2T82, S2T88, S2T96, S2T98, S2T102, S2T125 |
| 均匀。 | 5 | 均匀 | global_balance | S2T104, S2T107, S2T110, S2T112, S2T116 |
| 三个都在躯干上面。 | 1 | 三个都在躯干上面 | body_geometry | S1T270 |
| 三长一短。 | 1 | 三长一短 | count_abstract | S1T77 |
| 整体都比较短。 | 1 | 整体都比较短 | other_unparsed | S1T179 |
| 整体都比较长。 | 1 | 整体都比较长 | other_unparsed | S1T172 |
| 脖子不是最短。 | 1 | 脖子不是最短 | other_unparsed | S1T87 |

#### S231

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 各个部分差不多长。 | 2 | 各个部分差不多长 | global_balance | S1T302, S3T125 |
| 肘比腿长，脖子比尾巴长。 | 2 | 肘比腿长 | other_unparsed | S3T209, S3T302 |
| 各个部分差不多长，脖子、尾巴一样长。 | 1 | 各个部分差不多长 | global_balance | S1T286 |
| 各个部分看上去差不多，腿和尾巴略短。 | 1 | 各个部分看上去差不多 | global_balance | S1T160 |
| 各个部分都是中等长度。 | 1 | 各个部分都是中等长度 | other_unparsed | S2T125 |
| 头和脖子短，腿和尾巴长，两长，两短。 | 1 | 两长; 两短 | other_unparsed | S1T40 |
| 头最短，其他部位也短，比头长一些。 | 1 | 比头长一些 | other_unparsed | S2T100 |
| 头最短，脖子和尾巴差不多长，都长。 | 1 | 都长 | other_unparsed | S1T73 |
| 头最长，其他三个部分差不多，比较短。 | 1 | 比较短 | other_unparsed | S1T64 |
| 尾巴和脖子。 | 1 | 尾巴和脖子 | other_unparsed | S1T140 |
| 尾巴最长，其他部位稍短，并且长度差不多。 | 1 | 并且长度差不多 | global_balance | S1T30 |
| 尾巴最长，然后是脖子，其他两个部位差不多。 | 1 | 然后是脖子 | ordinal_or_secondary | S1T22 |
| 尾巴比较短，腿长，头和脖子也比较长，相差不大。 | 1 | 相差不大 | other_unparsed | S2T213 |
| 差不多长，腿略短一点。 | 1 | 差不多长 | global_balance | S2T143 |
| 手比腿长，脖子比尾巴长。 | 1 | 手比腿长 | other_unparsed | S3T314 |
| 脖子最短，腿和尾巴最长，并且差不多。 | 1 | 并且差不多 | global_balance | S1T31 |
| 脖子最长，腿可能和腿差不多。 | 1 | 腿可能和腿差不多 | global_balance | S1T60 |
| 脖子较短，各个部分都比较长。 | 1 | 各个部分都比较长 | other_unparsed | S1T214 |
| 腿最长，头和脖子都短，并且差不多。 | 1 | 并且差不多 | global_balance | S1T35 |
| 都差不多，头略短一些。 | 1 | 都差不多 | global_balance | S1T156 |
| 都比较短。 | 1 | 都比较短 | other_unparsed | S4T66 |

#### S301

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子和头都很长，长度相近。 | 3 | 长度相近 | other_unparsed | S1T309, S2T10, S2T11 |
| 头和脖子很长，腿也很长，三者长度相近。 | 1 | 三者长度相近 | other_unparsed | S1T110 |
| 头和脖子极长，比尾巴和腿都长。 | 1 | 比尾巴和腿都长 | other_unparsed | S1T188 |
| 头和脖子相对较长，腿和尾巴中等，稍短一些。 | 1 | 稍短一些 | other_unparsed | S1T160 |
| 头和脖子较长，均比腿长，尾巴较短。 | 1 | 均比腿长 | other_unparsed | S1T97 |
| 头和脖子非常长，均比腿长。 | 1 | 均比腿长 | other_unparsed | S1T78 |
| 头和脖子非常长，比尾巴长，腿较短。 | 1 | 比尾巴长 | other_unparsed | S1T217 |
| 头很长、比脖子长一些，尾巴和腿相对中等。 | 1 | 头很长、比脖子长一些 | other_unparsed | S1T133 |
| 头很长，比脖子长很多，腿也比较长。 | 1 | 比脖子长很多 | other_unparsed | S1T169 |
| 头很长，比脖子长，脖子比腿要短。 | 1 | 比脖子长 | other_unparsed | S1T252 |
| 头极长，比尾巴长。 | 1 | 比尾巴长 | other_unparsed | S1T192 |
| 头长，尾巴，脖子短，尾巴长。 | 1 | 尾巴 | other_unparsed | S1T317 |
| 头长，脖子短，相对来说，头和脖子都很长。 | 1 | 相对来说 | other_unparsed | S1T304 |
| 头长，脖子短，相对较小。 | 1 | 相对较小 | other_unparsed | S1T293 |
| 头非常长，比脖子长很多。 | 1 | 比脖子长很多 | other_unparsed | S1T241 |
| 尾巴很长，头很长，两者长度相近，腿很短。 | 1 | 两者长度相近 | other_unparsed | S1T138 |
| 相对来说，脖子比较长，腿非常长。 | 1 | 相对来说 | other_unparsed | S1T229 |
| 脖子和头都较短，长度相近，尾巴较长。 | 1 | 长度相近 | other_unparsed | S2T20 |
| 脖子和头长度相近，都偏短。 | 1 | 都偏短 | other_unparsed | S2T13 |
| 脖子很长、长于头，腿较长。 | 1 | 脖子很长、长于头 | other_unparsed | S1T112 |
| 脖子很长，比头和尾巴都长。 | 1 | 比头和尾巴都长 | other_unparsed | S1T281 |
| 脖子很长，相对来说，比头和尾巴长。 | 1 | 相对来说; 比头和尾巴长 | other_unparsed | S1T213 |
| 脖子比腿长，比头长。 | 1 | 比头长 | other_unparsed | S1T245 |
| 脖子非常长、长于头，腿较长，尾巴较短。 | 1 | 脖子非常长、长于头 | other_unparsed | S1T105 |
| 脖子非常长、长于尾巴，头极短，腿极短。 | 1 | 脖子非常长、长于尾巴 | other_unparsed | S1T100 |
| 脖子非常长，其余三个部位较长，长度相近。 | 1 | 长度相近 | other_unparsed | S1T114 |
| 腿很长，其余三个部位较长，长度相近。 | 1 | 长度相近 | other_unparsed | S1T94 |
| 腿很长，头和脖子相近，也都比较长。 | 1 | 也都比较长 | other_unparsed | S1T72 |
| 腿极短，头、脖子、尾巴都较长，长度相近。 | 1 | 长度相近 | other_unparsed | S1T42 |
| 腿极长，头、脖子和尾巴相近，稍短一些。 | 1 | 稍短一些 | other_unparsed | S1T36 |
| 腿比较长，脖子比较长，比头长。 | 1 | 比头长 | other_unparsed | S1T242 |
| 腿较长，脖子很长，比头长。 | 1 | 比头长 | other_unparsed | S1T139 |
| 腿较长，脖子相对较长、比头长。 | 1 | 脖子相对较长、比头长 | other_unparsed | S1T102 |
| 长度比较均衡，相对来说脖子长一些。 | 1 | 长度比较均衡 | global_balance | S1T270 |

#### S302

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿长，其他部位都不是很长，比较匀称。 | 2 | 比较匀称 | global_balance | S1T109, S1T136 |
| 上半身很长，腿很短。 | 1 | 上半身很长 | other_unparsed | S1T85 |
| 上半身比较长，腿相对短。 | 1 | 上半身比较长 | other_unparsed | S1T89 |
| 头比较短，其他部位比较匀称。 | 1 | 其他部位比较匀称 | other_reference, global_balance | S1T103 |
| 头相对长，其他部位不是很长，比较匀称。 | 1 | 比较匀称 | global_balance | S1T106 |
| 尾巴长，脖子、头、腿都不是很长，整体比较匀称。 | 1 | 整体比较匀称 | global_balance | S1T93 |
| 脖子和头都比较长，整体，腿和尾巴也相对长。 | 1 | 整体 | other_unparsed | S1T129 |
| 脖子比，头比脖子长。 | 1 | 脖子比 | other_unparsed | S1T156 |
| 脖子长，尾巴长，头短，腿中等，整体比较匀称。 | 1 | 整体比较匀称 | global_balance | S1T81 |
| 脖子，脖子长，尾巴短。 | 1 | 脖子 | other_unparsed | S1T42 |
| 腿很长，上半身都比较短。 | 1 | 上半身都比较短 | other_unparsed | S1T90 |
| 都比较中等，都不是很长。 | 1 | 都比较中等; 都不是很长 | other_unparsed | S1T146 |

#### S303

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头，脖子、腿都短，尾巴也短、比其他部位长一点。 | 1 | 头 | other_unparsed | S1T90 |

#### S304

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和腿长度相同。 | 2 | 头和腿长度相同 | other_unparsed | S1T23, S1T25 |
| 头和躯干长度相同，其他部位长度是躯干的0.7倍。 | 1 | 头和躯干长度相同; 其他部位长度是躯干的0.7倍 | other_reference, body_geometry | S1T6 |

#### S305

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 挺高大。 | 8 | 挺高大 | body_geometry, vague_size | S1T50, S1T51, S1T52, S1T54, S1T55, S1T56, S1T59, S1T61 |
| 身材高大。 | 7 | 身材高大 | body_geometry, vague_size | S1T70, S1T80, S1T82, S1T89, S1T92, S1T94, S1T96 |
| 很高大。 | 6 | 很高大 | body_geometry, vague_size | S1T1, S1T12, S1T23, S1T24, S1T31, S1T46 |
| 挺高大，脖子短。 | 5 | 挺高大 | body_geometry, vague_size | S1T37, S1T40, S1T42, S1T45, S1T47 |
| 中等身材，头很短。 | 1 | 中等身材 | body_geometry, vague_size | S1T4 |
| 很高。 | 1 | 很高 | other_unparsed | S1T3 |
| 比较适中，头和尾巴长，脖子短。 | 1 | 比较适中 | other_unparsed | S1T32 |
| 比较高大。 | 1 | 比较高大 | body_geometry, vague_size | S1T34 |
| 脖子高，脖子短。 | 1 | 脖子高 | other_unparsed | S1T35 |
| 腿适中，其实还挺短。 | 1 | 其实还挺短 | other_unparsed | S1T29 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S1T163 |

#### S306

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子比头长，和腿差不多，比腿稍长，尾巴最短。 | 1 | 和腿差不多; 比腿稍长 | other_unparsed | S1T44 |
| 腿和头差不多一样长，长度较长，尾巴和脖子相对来说较短。 | 1 | 长度较长 | other_unparsed | S1T42 |
| 腿和尾巴一样长，较长，脖子非常长，头较短。 | 1 | 较长 | other_unparsed | S1T11 |
| 腿很长，脖子，头较短，尾巴适中。 | 1 | 脖子 | other_unparsed | S2T83 |
| 腿明显很短，头比脖子长，比尾巴长很多。 | 1 | 比尾巴长很多 | other_unparsed | S1T214 |
| 腿明显比较长，头比脖子长，两者都很长，尾巴短。 | 1 | 两者都很长 | other_unparsed | S1T218 |
| 腿短，尾巴很长，脖子和头，较短。 | 1 | 脖子和头; 较短 | other_unparsed | S1T147 |
| 腿短，尾巴长，头和脖子，也都很长。 | 1 | 头和脖子; 也都很长 | other_unparsed | S2T66 |
| 腿短，脖子和头接近，长度较长，尾巴更长一些。 | 1 | 长度较长 | other_unparsed | S2T17 |
| 腿短，脖子和尾巴一样长，比头短。 | 1 | 比头短 | other_unparsed | S1T155 |
| 腿短，脖子比头稍长，比尾巴长很多。 | 1 | 比尾巴长很多 | other_unparsed | S1T88 |
| 腿短，脖子比尾巴稍长，也比头长一些。 | 1 | 也比头长一些 | other_unparsed | S1T73 |
| 腿短，脖子，脖子稍长。 | 1 | 脖子 | other_unparsed | S2T259 |
| 腿较短，其他三者较长，且长度接近。 | 1 | 且长度接近 | other_unparsed | S2T36 |
| 腿较短，头较长、略长于尾巴，脖子较短。 | 1 | 头较长、略长于尾巴 | other_unparsed | S2T43 |
| 腿较短，尾巴短，头，和脖子较长，脖子比头长一些。 | 1 | 头 | other_unparsed | S2T22 |
| 腿较短，脖子、头较长，尾巴略比前两者略短一些。 | 1 | 尾巴略比前两者略短一些 | other_unparsed | S1T247 |
| 腿较短，脖子比头子，脖子比头长，尾巴和脖子差不多一样长。 | 1 | 脖子比头子 | other_unparsed | S1T220 |
| 腿较短，脖子较短，头和尾巴长度接近，较长。 | 1 | 较长 | other_unparsed | S2T37 |
| 腿较短，脖子较长，头和尾巴差不多一样长，比脖子短。 | 1 | 比脖子短 | other_unparsed | S1T174 |
| 腿较长，头较长、略长于脖子，脖子长度略短于尾巴。 | 1 | 头较长、略长于脖子; 脖子长度略短于尾巴 | other_unparsed | S2T42 |
| 腿较长，尾巴较短，头和脖子长度接近，较长。 | 1 | 较长 | other_unparsed | S2T18 |
| 腿较长，尾巴，尾巴长，脖子比头短。 | 1 | 尾巴 | other_unparsed | S1T222 |
| 腿长，头，较长。 | 1 | 头; 较长 | other_unparsed | S2T95 |
| 腿长，脖子和尾巴一样长，比头长。 | 1 | 比头长 | other_unparsed | S1T72 |
| 腿长，脖子和尾巴差不多长，比头长一些。 | 1 | 比头长一些 | other_unparsed | S2T14 |
| 较躯干来说，腿短，尾巴长。 | 1 | 较躯干来说 | body_geometry | S1T65 |
| 较躯干来说，腿短，尾巴长，头比脖子稍长。 | 1 | 较躯干来说 | body_geometry | S1T67 |
| 较躯干来说，腿较短，其他部位均较长，脖子比头长。 | 1 | 较躯干来说 | body_geometry | S1T57 |
| 较躯干来说，腿较长，尾巴短，脖子和头都较长。 | 1 | 较躯干来说 | body_geometry | S1T63 |
| 较躯干来说，腿较长，脖子比头长，是头比脖子长，尾巴短。 | 1 | 较躯干来说 | body_geometry | S1T64 |
| 较躯干来说，腿适中，其他部位较长一些。 | 1 | 较躯干来说 | body_geometry | S1T58 |
| 较躯干而言，腿的适中，脖子较短，头较长，尾巴短。 | 1 | 较躯干而言 | body_geometry | S1T60 |
| 较躯干而言，腿短，头比脖子稍长一些，四个部位都短，尾巴最长。 | 1 | 较躯干而言 | body_geometry | S1T62 |
| 较躯干而言，腿适中，脖子比头长。 | 1 | 较躯干而言 | body_geometry | S1T61 |
| 较躯干而言，腿适中，脖子较短，其他部位适中。 | 1 | 较躯干而言 | body_geometry | S1T59 |
| 较躯干而言，腿长，尾巴短。 | 1 | 较躯干而言 | body_geometry | S1T66 |

#### S307

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和脖子差不多长，它们都是中等，腿较长，尾巴较长。 | 1 | 它们都是中等 | other_unparsed | S1T86 |
| 头和脖子差不多长，都是较长，尾巴中等，腿较长。 | 1 | 都是较长 | other_unparsed | S1T66 |
| 头和脖子，腿较短，尾巴很短。 | 1 | 头和脖子 | other_unparsed | S1T159 |
| 头较长，脖，脖子，中等，尾巴中等，腿较短。 | 1 | 脖子; 中等 | other_unparsed | S1T174 |
| 脖子是头的两倍。 | 1 | 脖子是头的两倍 | other_unparsed | S1T141 |
| 脖子比头长一点，两个都是较长，尾巴较长，腿很短。 | 1 | 两个都是较长 | other_unparsed | S2T4 |
| 腿、脖子和尾巴，四个部位都短，头较长。 | 1 | 腿、脖子和尾巴 | other_unparsed | S1T14 |
| 腿和尾巴很长，脖子很长，头很小。 | 1 | 头很小 | vague_size | S2T62 |
| 腿很长，脖子，脖子、尾巴较长，头很短。 | 1 | 脖子 | other_unparsed | S2T193 |

#### S308

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖子不是最长。 | 12 | 脖子不是最长 | other_unparsed | S2T235, S2T242, S2T243, S2T244, S2T257, S2T258, S2T273, S2T280 |
| 哪个最长，头和脖子长度相似，腿最短。 | 1 | 哪个最长 | other_unparsed | S1T160 |
| 头与尾巴长度相似，比脖子长。 | 1 | 比脖子长 | other_unparsed | S1T92 |
| 头最短，腿与尾巴等长，且更长。 | 1 | 且更长 | other_unparsed | S1T112 |
| 尾巴最长，其，头、脖子、腿长度相似。 | 1 | 其 | other_unparsed | S1T89 |
| 尾巴比腿，腿长。 | 1 | 尾巴比腿 | other_unparsed | S1T296 |
| 脖子不是最长，尾巴长。 | 1 | 脖子不是最长 | other_unparsed | S2T241 |
| 脖子不是最长，腿长。 | 1 | 脖子不是最长 | other_unparsed | S2T239 |
| 脖子与尾巴长度相似，且最长。 | 1 | 且最长 | other_unparsed | S1T63 |
| 脖子与腿长度相似，且长度长于头。 | 1 | 且长度长于头 | other_unparsed | S1T55 |
| 脖子比头长，脖子与尾巴长度无法分辨。 | 1 | 脖子与尾巴长度无法分辨 | other_unparsed | S2T14 |
| 脖子比头长，腿与尾巴长度相似，且比且比脖子更长。 | 1 | 且比且比脖子更长 | other_unparsed | S1T210 |
| 脖子比头长，腿和尾巴长度相似，且比脖子更长。 | 1 | 且比脖子更长 | other_unparsed | S1T194 |
| 脖子比尾巴长，比腿长。 | 1 | 比腿长 | other_unparsed | S2T35 |
| 腿与尾巴，最短。 | 1 | 腿与尾巴; 最短 | other_unparsed | S1T53 |
| 腿最长，尾巴最短，脖子比头长，短于腿。 | 1 | 短于腿 | other_unparsed | S1T135 |

#### S309

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 整体都较为均匀。 | 2 | 整体都较为均匀 | global_balance | S1T26, S1T27 |
| 比较均衡。 | 2 | 比较均衡 | global_balance | S1T258, S1T264 |
| 基本都很均衡。 | 1 | 基本都很均衡 | global_balance | S1T317 |
| 尾巴短，小小。 | 1 | 小小 | other_unparsed | S1T259 |
| 尾巴非常短，其他部位正常。 | 1 | 其他部位正常 | other_reference | S1T23 |
| 腿、头、尾巴都比脖子长。 | 1 | 腿、头、尾巴都比脖子长 | other_unparsed | S1T53 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S3T9 |
| 非常均衡。 | 1 | 非常均衡 | global_balance | S1T295 |

#### S310

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短，躯干长。 | 2 | 躯干长 | body_geometry | S3T300, S3T319 |
| 头、脖子和腿长度差不多，尾巴要稍短一些，和躯干的长度相当，全身比较均衡。 | 1 | 全身比较均衡 | global_balance | S2T37 |
| 头和尾巴长，脖子和腿，短。 | 1 | 脖子和腿; 短 | other_unparsed | S2T227 |
| 头和尾巴，相当躯干，差不多，脖子最长，腿最短。 | 1 | 头和尾巴; 相当躯干; 差不多 | body_geometry, global_balance | S3T69 |
| 头和腿长，脖子和腿，和尾巴短。 | 1 | 脖子和腿 | other_unparsed | S1T300 |
| 头很长，脖子和腿，长度差不多。 | 1 | 脖子和腿; 长度差不多 | global_balance | S1T160 |
| 头最长、比躯干长，脖子、腿、尾巴长度相当，且与躯干长度差不多。 | 1 | 且与躯干长度差不多 | body_geometry, global_balance | S3T49 |
| 头最长，也比较短，脖子、腿、尾巴都很短。 | 1 | 也比较短 | other_unparsed | S2T109 |
| 头最长，尾巴稍微比脖子和腿短一点。 | 1 | 尾巴稍微比脖子和腿短一点 | other_unparsed | S2T67 |
| 头最长，脖子、腿，差不多长，尾巴很短。 | 1 | 脖子、腿; 差不多长 | global_balance | S2T82 |
| 头最长，腿最短，相差较大。 | 1 | 相差较大 | other_unparsed | S1T41 |
| 头比躯干短，其余各部位长度比头长，且长度相当。 | 1 | 其余各部位长度比头长; 且长度相当 | other_reference, global_balance | S4T17 |
| 头比躯干短，尾巴短，躯干最长，且明显长于其他部位。 | 1 | 躯干最长 | body_geometry | S4T16 |
| 头比躯干短，腿比其他部位都长，且最长。 | 1 | 且最长 | other_unparsed | S4T13 |
| 头比躯干短，躯干较长，其余各部分较短。 | 1 | 躯干较长 | body_geometry | S4T18 |
| 头短，其余各肢平衡。 | 1 | 其余各肢平衡 | other_reference | S3T285 |
| 头短，尾巴、躯干较长，腿，脖子较短。 | 1 | 腿 | other_unparsed | S3T262 |
| 头短，脖子长，尾巴长，躯干长，腿最短。 | 1 | 躯干长 | body_geometry | S4T36 |
| 头短，躯干最长，其余各部位都较短且差不多长。 | 1 | 躯干最长 | body_geometry | S4T40 |
| 头稍短，脖子、躯干、尾巴、腿长度差不多，比头长。 | 1 | 比头长 | other_unparsed | S3T42 |
| 头稍长，躯干最长，脖子、尾巴、腿较短。 | 1 | 躯干最长 | body_geometry | S3T92 |
| 头，尾巴稍短，脖子最长，腿稍长。 | 1 | 头 | other_unparsed | S3T28 |
| 头，脖子和腿较长，尾巴稍短。 | 1 | 头 | other_unparsed | S2T103 |
| 头，腿、尾巴很长，脖子很短。 | 1 | 头 | other_unparsed | S3T77 |
| 尾巴和腿，长度比躯干更长，头和脖子很短。 | 1 | 尾巴和腿; 长度比躯干更长 | body_geometry | S3T75 |
| 尾巴长，腿第二，头第三，脖子，次。 | 1 | 头第三; 脖子; 次 | ordinal_or_secondary | S2T76 |
| 差不多长。 | 1 | 差不多长 | global_balance | S2T117 |
| 脖子很长，其他，差不多。 | 1 | 其他; 差不多 | other_reference, global_balance | S2T155 |
| 脖子最长，头，尾巴和腿稍短。 | 1 | 头 | other_unparsed | S2T224 |
| 脖子最，头最长，脖子最短，腿和尾巴中间。 | 1 | 脖子最 | other_unparsed | S2T226 |
| 脖子长，头和尾巴，中间腿最短。 | 1 | 头和尾巴 | other_unparsed | S1T235 |
| 腿很长，尾巴稍短，头和脖子要比尾巴更短一些。 | 1 | 头和脖子要比尾巴更短一些 | other_unparsed | S2T27 |
| 腿最短，躯干最长，头、脖子、尾巴长度中间且差不多。 | 1 | 躯干最长 | body_geometry | S3T71 |
| 腿最长，其他部位都相对较短，且长度差不多。 | 1 | 且长度差不多 | global_balance | S1T251 |
| 腿最长，头、尾巴、脖子较短，且长度差不多。 | 1 | 且长度差不多 | global_balance | S1T236 |
| 腿最长，头和躯干差不多，脖子最短，尾巴稍短。 | 1 | 头和躯干差不多 | body_geometry, global_balance | S3T78 |

#### S311

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和脖子最长，腿和尾巴。 | 2 | 腿和尾巴 | other_unparsed | S1T172, S1T174 |
| 头最长，尾巴、脖子和腿。 | 2 | 尾巴、脖子和腿 | other_unparsed | S1T92, S1T179 |
| 尾巴最长，腿、头和脖子。 | 2 | 腿、头和脖子 | other_unparsed | S1T110, S1T250 |
| 脖子最长，尾巴、头和腿。 | 2 | 尾巴、头和腿 | other_unparsed | S1T99, S1T294 |
| 脖子最长，腿、头和尾巴。 | 2 | 腿、头和尾巴 | other_unparsed | S1T93, S1T216 |
| 腿最长，头、尾巴和脖子。 | 2 | 头、尾巴和脖子 | other_unparsed | S1T61, S1T106 |
| 腿最长，尾巴、头和脖子。 | 2 | 尾巴、头和脖子 | other_unparsed | S1T96, S1T202 |
| 头、脖子、尾巴、腿略长，且都比较接近。 | 1 | 且都比较接近 | other_unparsed | S1T1 |
| 头和尾巴很长，脖子，腿最短。 | 1 | 脖子 | other_unparsed | S1T143 |
| 头和尾巴最长，脖子和腿。 | 1 | 脖子和腿 | other_unparsed | S1T242 |
| 头和尾巴略长，腿和脖子。 | 1 | 腿和脖子 | other_unparsed | S1T142 |
| 头和脖子最长，尾巴和腿。 | 1 | 尾巴和腿 | other_unparsed | S1T136 |
| 头和腿最长，尾巴和脖子。 | 1 | 尾巴和脖子 | other_unparsed | S1T286 |
| 头最长，尾巴、脖子和腿，很短。 | 1 | 尾巴、脖子和腿; 很短 | other_unparsed | S1T137 |
| 头最长，尾巴和腿。 | 1 | 尾巴和腿 | other_unparsed | S1T192 |
| 头最长，尾巴，脖子和腿略短。 | 1 | 尾巴 | other_unparsed | S1T25 |
| 头最长，脖子、腿和尾巴。 | 1 | 脖子、腿和尾巴 | other_unparsed | S1T251 |
| 头最长，腿、尾巴和脖子。 | 1 | 腿、尾巴和脖子 | other_unparsed | S1T98 |
| 头最长，腿，脖子和尾巴很短。 | 1 | 腿 | other_unparsed | S1T102 |
| 尾巴和头。 | 1 | 尾巴和头 | other_unparsed | S2T8 |
| 尾巴和脖子最长，腿和头。 | 1 | 腿和头 | other_unparsed | S1T163 |
| 尾巴和脖子比较长，头，腿最短。 | 1 | 头 | other_unparsed | S1T161 |
| 尾巴和腿也比较长，脖子，头最短。 | 1 | 脖子 | other_unparsed | S1T123 |
| 尾巴最长，脖子、头和腿，很短。 | 1 | 脖子、头和腿; 很短 | other_unparsed | S1T78 |
| 尾巴最长，脖子和腿，还有头。 | 1 | 脖子和腿; 还有头 | other_unparsed | S1T317 |
| 脖子、腿、尾巴比较长，且比较接近，头最短。 | 1 | 且比较接近 | other_unparsed | S1T6 |
| 脖子最长，其次是尾巴、头，和腿。 | 1 | 和腿 | other_unparsed | S1T115 |
| 脖子最长，头、腿和尾巴。 | 1 | 头、腿和尾巴 | other_unparsed | S1T135 |
| 脖子最长，头，腿和尾巴。 | 1 | 头; 腿和尾巴 | other_unparsed | S1T149 |
| 脖子最长，尾巴和头，还有腿。 | 1 | 尾巴和头; 还有腿 | other_unparsed | S1T240 |
| 脖子略长，尾巴，再是腿和头。 | 1 | 尾巴; 再是腿和头 | ordinal_or_secondary | S1T270 |
| 腿和尾巴比较长，头，脖子最短。 | 1 | 头 | other_unparsed | S1T160 |
| 腿最长，其次是头和脖子，腿。 | 1 | 腿 | other_unparsed | S1T222 |
| 腿最长，头、脖子和尾巴。 | 1 | 头、脖子和尾巴 | other_unparsed | S1T121 |
| 腿最长，头和尾巴，还有脖子。 | 1 | 头和尾巴; 还有脖子 | other_unparsed | S1T156 |
| 腿最长，尾巴、脖子和头。 | 1 | 尾巴、脖子和头 | other_unparsed | S1T148 |
| 腿最长，脖子、头和尾巴。 | 1 | 脖子、头和尾巴 | other_unparsed | S1T112 |

#### S312

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 一样长。 | 1 | 一样长 | other_unparsed | S1T197 |
| 中等长度，头和尾巴略短。 | 1 | 中等长度 | other_unparsed | S3T137 |
| 中等长度，腿略长，脖子很短。 | 1 | 中等长度 | other_unparsed | S3T138 |
| 头、腿、尾巴，很短。 | 1 | 头、腿、尾巴; 很短 | other_unparsed | S3T54 |
| 差不多长。 | 1 | 差不多长 | global_balance | S2T254 |
| 选错了。 | 1 | 选错了 | meta_or_uncertain | S2T140 |
| 都略短，头比尾巴长。 | 1 | 都略短 | other_unparsed | S3T18 |
| 都较短，头略长。 | 1 | 都较短 | other_unparsed | S3T253 |
| 都较短，尾巴和腿更短一点。 | 1 | 都较短 | other_unparsed | S3T315 |

#### S313

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头、脖子、尾巴、腿均小于或等于，躯干。 | 1 | 头、脖子、尾巴、腿均小于或等于; 躯干 | body_geometry | S3T53 |
| 头和腿，脖子和尾巴比躯干短。 | 1 | 头和腿 | other_unparsed | S3T37 |
| 头很长，尾巴、腿、脖子差不多长，都比头稍微短一点。 | 1 | 都比头稍微短一点 | other_unparsed | S2T70 |
| 尾巴短，腿短，脖子比头短，都比尾巴和腿长。 | 1 | 都比尾巴和腿长 | other_unparsed | S2T281 |
| 尾巴，巴长，腿短，头短。 | 1 | 尾巴; 巴长 | other_unparsed | S1T165 |
| 腿短，尾巴长，头较短，和脖子相比。 | 1 | 和脖子相比 | other_unparsed | S2T176 |
| 腿长，尾巴长，头和脖子差不多长，头和脖子都比尾巴和腿短。 | 1 | 头和脖子都比尾巴和腿短 | other_unparsed | S2T79 |

#### S314

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 躯干长于脖子，且短于腿。 | 6 | 且短于腿 | other_unparsed | S3T137, S3T139, S3T141, S3T142, S3T145, S3T146 |
| 朝向向左。 | 4 | 朝向向左 | other_unparsed | S1T37, S1T39, S1T40, S1T42 |
| 有一个部位比躯干长。 | 3 | 有一个部位比躯干长 | count_abstract, body_geometry | S1T43, S1T44, S1T45 |
| 躯干短于脖子，且长于尾巴。 | 2 | 且长于尾巴 | other_unparsed | S3T136, S3T140 |
| 去判断于脖子和尾巴。 | 1 | 去判断于脖子和尾巴 | other_unparsed | S3T190 |
| 去干短于脖子，且去干短于尾巴。 | 1 | 去干短于脖子; 且去干短于尾巴 | other_unparsed | S3T84 |
| 去干长于脖子，且去干长于腿。 | 1 | 去干长于脖子; 且去干长于腿 | other_unparsed | S2T290 |
| 朝向向右。 | 1 | 朝向向右 | other_unparsed | S1T41 |
| 脖子长于头、长于躯干，长于尾巴、长于腿。 | 1 | 长于尾巴、长于腿 | other_unparsed | S1T67 |
| 腿长于头、长于躯干，长于脖子。 | 1 | 长于脖子 | other_unparsed | S1T68 |
| 躯干短于脖子，且短于尾巴。 | 1 | 且短于尾巴 | other_unparsed | S3T138 |
| 躯干短于，脖子和尾巴。 | 1 | 躯干短于; 脖子和尾巴 | body_geometry | S3T179 |

#### S315

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 脖。 | 1 | 脖子 | other_unparsed | S1T95 |

#### S316

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿短，且是最短，头比脖子长。 | 4 | 且是最短 | other_unparsed | S1T249, S1T250, S1T253, S1T269 |
| 腿短，且是最短，头比脖子短。 | 3 | 且是最短 | other_unparsed | S1T254, S1T265, S1T270 |
| 腿短，且脖子不是头、脖子、尾巴里最短。 | 2 | 且脖子不是头、脖子、尾巴里最短 | other_unparsed | S1T137, S1T139 |
| 腿短，尾巴不比脖子长。 | 2 | 尾巴不比脖子长 | other_unparsed | S1T178, S1T179 |
| 腿长，尾巴明显比腿短。 | 2 | 尾巴明显比腿短 | other_unparsed | S1T62, S1T63 |
| 尾巴和脖子很长，腿很长，这题。 | 1 | 这题 | other_unparsed | S1T47 |
| 脖子不是最短，脖子比尾巴长，尾巴是头、脖子、尾巴里最短。 | 1 | 脖子不是最短 | other_unparsed | S1T153 |
| 脖子和尾巴不比头长，腿短，头最长。 | 1 | 脖子和尾巴不比头长 | other_unparsed | S1T213 |
| 脖子和尾巴都比头长。 | 1 | 脖子和尾巴都比头长 | other_unparsed | S1T155 |
| 脖子和腿的长度在2/3以上，头和尾巴几乎是最小长度。 | 1 | 脖子和腿的长度在2/3以上 | other_unparsed | S1T2 |
| 脖子比头长，尾巴也比头长，脖子比尾巴长。 | 1 | 尾巴也比头长 | other_unparsed | S1T96 |
| 腿不是很短，脖子和尾巴比较长，比腿略长一些。 | 1 | 比腿略长一些 | other_unparsed | S1T69 |
| 腿中等，腿比尾巴长，脖子是尾巴的两倍以上，脖子也是头的两倍以上。 | 1 | 脖子是尾巴的两倍以上; 脖子也是头的两倍以上 | other_unparsed | S1T170 |
| 腿比较短，脖子不是最短。 | 1 | 脖子不是最短 | other_unparsed | S1T109 |
| 腿没有明显长于脖子和尾巴，且头和脖子长度相似。 | 1 | 腿没有明显长于脖子和尾巴 | other_unparsed | S1T51 |
| 腿没有那么长，腿的长度已经很长了，还是。 | 1 | 还是 | other_unparsed | S1T61 |
| 腿短不是最短，脖子和尾巴比脖子长一点，头是最短。 | 1 | 腿短不是最短 | other_unparsed | S1T258 |
| 腿短，且不是最短，脖子不是最长。 | 1 | 且不是最短; 脖子不是最长 | other_unparsed | S1T257 |
| 腿短，且不是最短，脖子最长。 | 1 | 且不是最短 | other_unparsed | S1T271 |
| 腿短，且最短，头比脖子长。 | 1 | 且最短 | other_unparsed | S1T243 |
| 腿短，且脖子不是另外三个里最短。 | 1 | 且脖子不是另外三个里最短 | other_reference | S1T129 |
| 腿短，且脖子不是最短。 | 1 | 且脖子不是最短 | other_unparsed | S1T136 |
| 腿短，尾巴和脖子都比腿短，头比腿长。 | 1 | 尾巴和脖子都比腿短 | other_unparsed | S1T231 |
| 腿短，尾巴比脖子长，头也比脖子长。 | 1 | 头也比脖子长 | other_unparsed | S1T181 |
| 腿短，尾巴比脖子长，有，类了。 | 1 | 有; 类了 | other_unparsed | S1T206 |
| 腿短，是最短，尾巴比腿短，脖子是最长。 | 1 | 是最短 | other_unparsed | S1T251 |
| 腿短，是最短，脖子不是最长。 | 1 | 是最短; 脖子不是最长 | other_unparsed | S1T268 |
| 腿短，是最短，脖子是最长。 | 1 | 是最短 | other_unparsed | S1T248 |
| 腿短，脖子不明显，短于尾巴和头。 | 1 | 脖子不明显; 短于尾巴和头 | other_unparsed | S1T119 |
| 腿短，脖子不是另外三个里最短。 | 1 | 脖子不是另外三个里最短 | other_reference | S1T117 |
| 腿短，脖子不是头、脖子、尾巴里最短。 | 1 | 脖子不是头、脖子、尾巴里最短 | other_unparsed | S1T112 |
| 腿短，脖子不是头、脖子和尾巴里最短。 | 1 | 脖子不是头、脖子和尾巴里最短 | other_unparsed | S1T141 |
| 腿短，脖子与头相近，脖子肯定不是最短。 | 1 | 脖子肯定不是最短 | other_unparsed | S1T152 |
| 腿短，脖子和尾巴几乎一样长，远长于头和腿。 | 1 | 远长于头和腿 | other_unparsed | S1T66 |
| 腿短，脖子和尾巴都不比腿短。 | 1 | 脖子和尾巴都不比腿短 | other_unparsed | S1T237 |
| 腿短，脖子是尾巴的两倍以上，脖子比头长很多。 | 1 | 脖子是尾巴的两倍以上 | other_unparsed | S1T168 |
| 腿短，脖子比头长，比尾巴短。 | 1 | 比尾巴短 | other_unparsed | S1T149 |
| 腿短，脖子比腿短，且，尾巴和头非常长。 | 1 | 且 | other_unparsed | S1T189 |
| 腿短，脖子短，头远长于脖子，尾巴中等长度。 | 1 | 头远长于脖子 | other_unparsed | S1T74 |
| 腿短，腿不是最短。 | 1 | 腿不是最短 | other_unparsed | S1T236 |
| 腿短，腿不是最短，头比腿短。 | 1 | 腿不是最短 | other_unparsed | S1T244 |
| 腿短，腿是最短，头、脖子、尾巴都比腿长，头是最长。 | 1 | 头、脖子、尾巴都比腿长 | other_unparsed | S1T234 |
| 腿长，尾巴最短，脖子和头分不清楚。 | 1 | 脖子和头分不清楚 | other_unparsed | S1T145 |
| 腿长，尾巴比脖子长，头也比脖子长。 | 1 | 头也比脖子长 | other_unparsed | S1T188 |

#### S317

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 有部位达到最长或最短长度。 | 6 | 有部位达到最长或最短长度 | count_abstract, extreme_endpoint | S2T109, S2T110, S2T113, S2T316, S2T317, S2T319 |
| 脖子不是最长的部位。 | 5 | 脖子不是最长的部位 | other_unparsed | S2T176, S2T177, S2T178, S2T179, S2T181 |
| 腿不是最短的部位。 | 5 | 腿不是最短的部位 | other_unparsed | S2T185, S2T186, S2T187, S2T189, S2T190 |
| 没有部位达到最长或最短长度。 | 4 | 没有部位达到最长或最短长度 | count_abstract, extreme_endpoint | S2T111, S2T112, S2T315, S2T318 |
| 腿没有达到最大长度。 | 4 | 腿没有达到最大长度 | extreme_endpoint | S2T71, S2T154, S2T155, S2T156 |
| 有比腿更短的部位。 | 3 | 有比腿更短的部位 | other_unparsed | S2T172, S2T173, S2T174 |
| 两个以上部位比躯干的一半长。 | 2 | 两个以上部位比躯干的一半长 | body_geometry | S2T62, S2T63 |
| 头不比腿短。 | 2 | 头不比腿短 | other_unparsed | S2T124, S2T125 |
| 小于两个部位一样长。 | 2 | 小于两个部位一样长 | count_abstract | S2T313, S2T314 |
| 尾巴不短于脖子。 | 2 | 尾巴不短于脖子 | other_unparsed | S2T67, S2T68 |
| 有两个部位达到最长长度。 | 2 | 有两个部位达到最长长度 | count_abstract, extreme_endpoint | S2T286, S2T288 |
| 没有部位达到最长长度。 | 2 | 没有部位达到最长长度 | count_abstract, extreme_endpoint | S2T83, S2T94 |
| 腿没有达到最长或最短。 | 2 | 腿没有达到最长或最短 | extreme_endpoint | S2T76, S2T77 |
| 一个部位达到最长长度。 | 1 | 一个部位达到最长长度 | count_abstract, extreme_endpoint | S2T89 |
| 出现两个最长长度。 | 1 | 出现两个最长长度 | extreme_endpoint | S2T93 |
| 出现最长长度，且出现了三个最长长度。 | 1 | 出现最长长度; 且出现了三个最长长度 | extreme_endpoint | S2T92 |
| 大于两个部位一样长。 | 1 | 大于两个部位一样长 | count_abstract | S2T312 |
| 头不是最短的部位。 | 1 | 头不是最短的部位 | other_unparsed | S2T192 |
| 头长度不变。 | 1 | 头长度不变 | other_unparsed | S3T105 |
| 尾巴大约3/4躯干，脖子大约3/4躯干。 | 1 | 尾巴大约3/4躯干; 脖子大约3/4躯干 | body_geometry | S3T294 |
| 有三个部位和躯干一样长。 | 1 | 有三个部位和躯干一样长 | count_abstract, body_geometry | S1T15 |
| 有两个部位比躯干长。 | 1 | 有两个部位比躯干长 | count_abstract, body_geometry | S1T18 |
| 有部位达到最长或最短。 | 1 | 有部位达到最长或最短 | count_abstract, extreme_endpoint | S2T79 |
| 没有两个部位达到最长长度。 | 1 | 没有两个部位达到最长长度 | count_abstract, extreme_endpoint | S2T287 |
| 没有出现最小长度，尾巴小于脖子。 | 1 | 没有出现最小长度 | extreme_endpoint | S2T91 |
| 没有出现最长长度。 | 1 | 没有出现最长长度 | extreme_endpoint | S2T90 |
| 没有达到最长长度。 | 1 | 没有达到最长长度 | extreme_endpoint | S2T95 |
| 没有部位是最大或最小长度。 | 1 | 没有部位是最大或最小长度 | count_abstract, extreme_endpoint | S2T80 |
| 没有部位是最长或者最短。 | 1 | 没有部位是最长或者最短 | count_abstract | S2T78 |
| 没有部位是最长的。 | 1 | 没有部位是最长的 | count_abstract | S2T108 |
| 没有部位达到最长或最短的长度。 | 1 | 没有部位达到最长或最短的长度 | count_abstract, extreme_endpoint | S2T114 |
| 腿和头长度变化。 | 1 | 腿和头长度变化 | other_unparsed | S3T107 |
| 腿长度变化。 | 1 | 腿长度变化 | other_unparsed | S3T106 |
| 达到最长或最短长度的部位数小于等于一。 | 1 | 达到最长或最短长度的部位数小于等于一 | extreme_endpoint | S2T320 |

#### S318

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 判断不出哪一部分更长，尾巴不短。 | 1 | 判断不出哪一部分更长 | other_unparsed | S1T248 |
| 头不比尾巴长，腿比较长。 | 1 | 头不比尾巴长 | other_unparsed | S1T241 |
| 头和尾巴差不多长，腿不短，脖子比较长，算太长。 | 1 | 算太长 | other_unparsed | S1T263 |
| 头和尾巴相比，头更长，脖子比较短。 | 1 | 头和尾巴相比 | other_unparsed | S1T260 |
| 头和尾巴相比，尾巴更长，脖子和尾巴都很长，躯干腿很短。 | 1 | 头和尾巴相比 | other_unparsed | S1T259 |
| 头很长，头和尾巴相差很大，脖子也很长。 | 1 | 头和尾巴相差很大 | other_unparsed | S1T203 |
| 头最长，躯干头和尾巴相差很大。 | 1 | 躯干头和尾巴相差很大 | body_geometry | S1T202 |
| 头没有比尾巴长很多，脖子很长。 | 1 | 头没有比尾巴长很多 | other_unparsed | S1T215 |
| 尾巴和躯干看不清。 | 1 | 尾巴和躯干看不清 | body_geometry | S2T110 |
| 尾巴比腿长，头和脖子，脖子更长。 | 1 | 头和脖子 | other_unparsed | S1T72 |
| 尾巴比腿长，差距不大，脖子和头都很长。 | 1 | 差距不大 | other_unparsed | S1T104 |
| 尾巴比腿长，躯干差距很大。 | 1 | 躯干差距很大 | body_geometry | S1T102 |
| 尾巴比躯干看不清，脖子比较短，腿比较长。 | 1 | 尾巴比躯干看不清 | body_geometry | S2T107 |
| 尾巴比躯干长，腿和躯干差不多。 | 1 | 腿和躯干差不多 | body_geometry, global_balance | S2T116 |
| 脖子最短，尾巴比腿短，比脖子长，头也比较长。 | 1 | 比脖子长 | other_unparsed | S1T46 |
| 脖子没有明显的优势，腿比较长，腿和尾巴都很长。 | 1 | 脖子没有明显的优势 | other_unparsed | S1T272 |
| 腿最短，脖子和尾巴一样长，比腿长，头最长。 | 1 | 比腿长 | other_unparsed | S1T63 |

#### S319

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头和脖子的比例大于腿和尾巴。 | 26 | 头和脖子的比例大于腿和尾巴 | proportion_or_ratio | S1T88, S1T100, S1T103, S1T108, S1T109, S1T111, S1T113, S1T115 |
| 头和脖子的比例小于腿和尾巴。 | 10 | 头和脖子的比例小于腿和尾巴 | proportion_or_ratio | S1T98, S1T101, S1T102, S1T104, S1T112, S1T118, S1T122, S1T123 |
| 选错了。 | 6 | 选错了 | meta_or_uncertain | S2T201, S2T267, S2T283, S3T66, S3T102, S3T123 |
| 头和脖子都比腿长。 | 3 | 头和脖子都比腿长 | other_unparsed | S2T252, S3T52, S3T53 |
| 三个部位都比躯干长。 | 2 | 三个部位都比躯干长 | count_abstract, body_geometry | S4T50, S4T51 |
| 三个部位长于躯干。 | 2 | 三个部位长于躯干 | count_abstract, body_geometry | S2T305, S2T307 |
| 四个部位长度比较平衡。 | 2 | 四个部位长度比较平衡 | other_unparsed | S2T219, S2T225 |
| 三个部位都比较长。 | 1 | 三个部位都比较长 | count_abstract | S3T286 |
| 四个部位比较均等。 | 1 | 四个部位比较均等 | global_balance | S2T320 |
| 四个部位长度都小于躯干，且差不多长。 | 1 | 且差不多长 | global_balance | S3T78 |
| 四个部位长度都很平衡。 | 1 | 四个部位长度都很平衡 | other_unparsed | S2T279 |
| 四个部位长度都相同。 | 1 | 四个部位长度都相同 | other_unparsed | S2T76 |
| 头和脖子比例小于腿和尾巴。 | 1 | 头和脖子比例小于腿和尾巴 | proportion_or_ratio | S1T134 |
| 头和脖子的比例大于尾巴和腿。 | 1 | 头和脖子的比例大于尾巴和腿 | proportion_or_ratio | S1T151 |
| 头和脖子的比例小于腿和尾巴，腿长。 | 1 | 头和脖子的比例小于腿和尾巴 | proportion_or_ratio | S1T161 |
| 头和脖子的比例等于腿和尾巴。 | 1 | 头和脖子的比例等于腿和尾巴 | proportion_or_ratio | S1T138 |
| 头和脖子都长于腿和尾巴。 | 1 | 头和脖子都长于腿和尾巴 | other_unparsed | S1T33 |
| 头有点小。 | 1 | 头有点小 | other_unparsed | S1T255 |
| 尾巴和腿的比例大于头和脖子的比例。 | 1 | 尾巴和腿的比例大于头和脖子的比例 | proportion_or_ratio | S1T187 |
| 尾巴长，腿很小。 | 1 | 腿很小 | vague_size | S1T295 |
| 比较均匀。 | 1 | 比较均匀 | global_balance | S1T278 |
| 腿和尾巴的比例大于头和脖子。 | 1 | 腿和尾巴的比例大于头和脖子 | proportion_or_ratio | S1T147 |
| 腿和尾巴的比例大于脖子和头的比例。 | 1 | 腿和尾巴的比例大于脖子和头的比例 | proportion_or_ratio | S1T172 |
| 长度都很平均。 | 1 | 长度都很平均 | global_balance | S4T108 |
| 除了头以外，其他部位都很长，很平均。 | 1 | 很平均 | global_balance | S3T197 |
| 除了尾巴以外，其他三个部位都很长，而且很平均。 | 1 | 而且很平均 | global_balance | S3T145 |
| 除了尾巴，都小于躯干。 | 1 | 除了尾巴; 都小于躯干 | body_geometry | S4T62 |
| 除了脖子，都很平衡。 | 1 | 除了脖子; 都很平衡 | other_unparsed | S4T95 |
| 除了腿以外，其他部位都很长，且平均。 | 1 | 且平均 | global_balance | S3T198 |

#### S321

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿不是最短的，头比脖子长。 | 2 | 腿不是最短的 | other_unparsed | S2T92, S2T109 |
| 头比脖子短，尾巴也比腿短。 | 1 | 尾巴也比腿短 | other_unparsed | S2T35 |
| 头比脖子短，腿略微比尾巴短。 | 1 | 腿略微比尾巴短 | other_unparsed | S2T13 |
| 头相对，脖子较短。 | 1 | 头相对 | other_unparsed | S2T71 |
| 脖子比头短，比尾巴也短，腿相对较短。 | 1 | 比尾巴也短 | other_unparsed | S2T138 |
| 脖子比头短，比尾巴长，腿相对较短。 | 1 | 比尾巴长 | other_unparsed | S2T139 |
| 脖子比头短，腿与尾巴都比脖子长。 | 1 | 腿与尾巴都比脖子长 | other_unparsed | S2T87 |
| 腿不是最短的，头比脖子短。 | 1 | 腿不是最短的 | other_unparsed | S2T90 |
| 腿不是最短的，头短于脖子。 | 1 | 腿不是最短的 | other_unparsed | S2T93 |
| 腿不是最短，头比脖子和尾巴短。 | 1 | 腿不是最短 | other_unparsed | S2T106 |
| 腿不是最短，头比脖子长。 | 1 | 腿不是最短 | other_unparsed | S2T108 |
| 腿不是最短，头较脖子更短。 | 1 | 腿不是最短 | other_unparsed | S2T102 |
| 腿不是最短，尾巴比头和脖子都长。 | 1 | 腿不是最短 | other_unparsed | S2T103 |
| 腿不是最短，腿较长。 | 1 | 腿不是最短 | other_unparsed | S2T148 |
| 腿超级短，头、脖子和尾巴都比腿长。 | 1 | 头、脖子和尾巴都比腿长 | other_unparsed | S1T29 |
| 腿较短，头比尾巴轻。 | 1 | 头比尾巴轻 | other_unparsed | S2T150 |
| 腿较短，是最短。 | 1 | 是最短 | other_unparsed | S2T147 |
| 腿较长，头比尾巴重。 | 1 | 头比尾巴重 | other_unparsed | S2T149 |
| 腿较，尾巴和脖子都长，头也长。 | 1 | 腿较 | other_unparsed | S1T78 |

#### S322

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头不是最长。 | 21 | 头不是最长 | other_unparsed | S1T186, S1T187, S1T223, S1T226, S1T227, S1T230, S1T233, S1T236 |
| 四个部位长度各不相同，脖子长于躯干。 | 1 | 四个部位长度各不相同 | disjoint_inequality | S2T111 |
| 四个部位长度都不一样，尾巴最长，头和腿短于躯干，脖子和尾巴长于躯干。 | 1 | 四个部位长度都不一样 | disjoint_inequality | S2T106 |
| 头不是最长，头和脖子一样长，腿和尾巴一样长。 | 1 | 头不是最长 | other_unparsed | S2T7 |
| 头不是最长，尾巴最长。 | 1 | 头不是最长 | other_unparsed | S2T6 |
| 头不是最长，脖子最长， | 1 | 头不是最长 | other_unparsed | S2T4 |
| 头不是最长，腿最长， | 1 | 头不是最长 | other_unparsed | S2T3 |
| 头和尾巴长度相等，略短于脖子。 | 1 | 略短于脖子 | other_unparsed | S1T191 |
| 头和脖子一样长，与腿相近。 | 1 | 与腿相近 | other_unparsed | S1T205 |
| 头和脖子一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S2T26 |
| 头和脖子长度相等，略长于尾巴。 | 1 | 略长于尾巴 | other_unparsed | S1T207 |
| 头和腿一样长，且不是最长。 | 1 | 且不是最长 | other_unparsed | S1T302 |
| 头和腿一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S2T32 |
| 头和腿一样长，中等偏长，脖子和尾巴一样长，略长于前两者。 | 1 | 中等偏长; 略长于前两者 | other_unparsed | S1T39 |
| 头比腿长，四个部位长度不相等。 | 1 | 四个部位长度不相等 | other_unparsed | S2T158 |
| 尾巴很短，腿和头差不多长，略长于尾巴，脖子最长。 | 1 | 略长于尾巴 | other_unparsed | S1T30 |
| 尾巴略长于头，四个部位长度不相等。 | 1 | 四个部位长度不相等 | other_unparsed | S1T85 |
| 短于躯干，两个长于躯干，脖子是最长。 | 1 | 短于躯干; 两个长于躯干 | body_geometry | S2T17 |
| 短于躯干，头和躯干一样长。 | 1 | 短于躯干 | body_geometry | S2T203 |
| 短于躯干，尾巴和躯干一样长。 | 1 | 短于躯干 | body_geometry | S2T202 |
| 短于躯干，有两个长于躯干。 | 1 | 短于躯干; 有两个长于躯干 | body_geometry | S2T20 |
| 短于躯干，腿最长，脖子很长。 | 1 | 短于躯干 | body_geometry | S2T204 |
| 脖子和头差不多长，腿和尾巴差不多长，且长于前二者。 | 1 | 且长于前二者 | other_unparsed | S1T26 |
| 脖子和头非常长，腿略短于二者，尾巴最短。 | 1 | 腿略短于二者 | other_unparsed | S1T19 |
| 脖子和尾巴一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S2T27 |
| 脖子和尾巴一样长，和头一样长。 | 1 | 和头一样长 | other_unparsed | S2T174 |
| 脖子和尾巴一样长，头不是最长。 | 1 | 头不是最长 | other_unparsed | S2T18 |
| 脖子和尾巴一样长，头略短于二者，腿最短。 | 1 | 头略短于二者 | other_unparsed | S1T18 |
| 脖子和腿一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S1T320 |
| 脖子和腿差不多长，头和尾巴差不多长，且长于前二者。 | 1 | 且长于前二者 | other_unparsed | S1T22 |
| 脖子和腿长度一样，头不是最长。 | 1 | 头不是最长 | other_unparsed | S2T9 |
| 脖子和腿长度相等，头不是最长。 | 1 | 头不是最长 | other_unparsed | S1T289 |
| 脖子和腿，头比脖子和腿长，尾巴很短。 | 1 | 脖子和腿 | other_unparsed | S1T2 |
| 脖子最长，头不是最长。 | 1 | 头不是最长 | other_unparsed | S2T8 |
| 腿和尾巴一样长，且头不是最长。 | 1 | 且头不是最长 | other_unparsed | S2T31 |
| 腿长，头、脖子、尾巴都不一样长。 | 1 | 头、脖子、尾巴都不一样长 | disjoint_inequality | S3T89 |

#### S323

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 选错了。 | 3 | 选错了 | meta_or_uncertain | S1T146, S1T156, S1T157 |

#### S324

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 腿和头较长，脖子长度中，脖子长度中等，尾巴较短。 | 1 | 脖子长度中 | other_unparsed | S1T47 |
| 腿和脖子较长，头和尾巴长中等，较短。 | 1 | 较短 | other_unparsed | S1T44 |
| 腿较短，前已经较长。 | 1 | 前已经较长 | other_unparsed | S1T28 |
| 腿较长，下巴为中等。 | 1 | 下巴为中等 | other_unparsed | S1T88 |

#### S325

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 每个都长。 | 3 | 每个都长 | other_unparsed | S1T133, S1T134, S1T140 |
| 每一个都很长。 | 1 | 每一个都很长 | other_unparsed | S1T194 |

#### S326

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 假设脖子无关。 | 6 | 假设脖子无关 | meta_or_uncertain | S1T243, S1T244, S1T245, S1T246, S1T247, S1T248 |
| 朝右。 | 4 | 朝右 | other_unparsed | S1T81, S1T82, S1T83, S1T84 |
| 假设尾巴无关。 | 2 | 假设尾巴无关 | meta_or_uncertain | S1T241, S1T242 |
| 腿和躯干差不多，头比脖子长。 | 2 | 腿和躯干差不多 | body_geometry, global_balance | S1T32, S1T33 |
| 腿比尾巴短，头在躯干以下。 | 2 | 头在躯干以下 | body_geometry | S1T125, S1T126 |
| 分布均匀，像猫。 | 1 | 分布均匀; 像猫 | global_balance | S1T222 |
| 头和脖子一样长，尾巴也差不多长，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T8 |
| 头和脖子很长，尾巴很短，头在躯干以上。 | 1 | 头在躯干以上 | body_geometry | S1T9 |
| 头和脖子比尾巴长，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T11 |
| 头在躯干以上，腿长，尾巴短。 | 1 | 头在躯干以上 | body_geometry | S1T40 |
| 头在躯干以下，尾巴和腿差不多。 | 1 | 头在躯干以下 | body_geometry | S1T47 |
| 头很长，脖子、尾巴很短，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T10 |
| 头最长，尾巴短，像狗。 | 1 | 像狗 | other_unparsed | S1T220 |
| 头长，尾巴短，脖子短，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T6 |
| 头长，尾巴短，脖子长，头在躯干以上。 | 1 | 头在躯干以上 | body_geometry | S1T5 |
| 头长，脖子一般，腿长，尾巴短，右朝向。 | 1 | 右朝向 | other_unparsed | S1T4 |
| 头长，脖子短，尾巴短，头在躯干以下、腿以上。 | 1 | 头在躯干以下、腿以上 | body_geometry | S1T1 |
| 头长，脖子长，尾巴短，头在躯干和腿之间，右朝向。 | 1 | 头在躯干和腿之间; 右朝向 | body_geometry | S1T3 |
| 头长，腿长，像狗。 | 1 | 像狗 | other_unparsed | S1T221 |
| 尾巴最长，依次是脖子、腿、头。 | 1 | 依次是脖子、腿、头 | other_unparsed | S1T209 |
| 尾巴长，头在躯干和腿之间。 | 1 | 头在躯干和腿之间 | body_geometry | S1T122 |
| 左朝向，头在躯干以下、腿以上，脖子和尾巴一般。 | 1 | 左朝向; 头在躯干以下、腿以上 | body_geometry | S1T2 |
| 朝右，脖子长，尾巴长。 | 1 | 朝右 | other_unparsed | S1T78 |
| 朝左。 | 1 | 朝左 | other_unparsed | S1T80 |
| 朝左，头短。 | 1 | 朝左 | other_unparsed | S1T79 |
| 朝左，脖子短，头短。 | 1 | 朝左 | other_unparsed | S1T77 |
| 狗状。 | 1 | 狗状 | other_unparsed | S1T295 |
| 脖子和头一样长，头在躯干以上。 | 1 | 头在躯干以上 | body_geometry | S1T16 |
| 脖子和尾巴一样长，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T15 |
| 脖子最长，依次是头、腿、尾巴。 | 1 | 依次是头、腿、尾巴 | other_unparsed | S1T210 |
| 腿最长，依次是脖子、头、尾巴。 | 1 | 依次是脖子、头、尾巴 | other_unparsed | S1T211 |
| 腿比较短，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T48 |
| 腿比较长，头在躯干以上，尾巴最长。 | 1 | 头在躯干以上 | body_geometry | S1T39 |
| 腿比较长，头在躯干以下。 | 1 | 头在躯干以下 | body_geometry | S1T38 |
| 腿短，头在躯干以上，尾巴长。 | 1 | 头在躯干以上 | body_geometry | S1T46 |
| 腿长，像狗。 | 1 | 像狗 | other_unparsed | S1T224 |
| 腿长，头在躯干以下，尾巴长。 | 1 | 头在躯干以下 | body_geometry | S1T41 |
| 腿长，头在躯干以下，脖子、尾巴长。 | 1 | 头在躯干以下 | body_geometry | S1T42 |
| 腿长，脖子最短，然后是头和尾巴。 | 1 | 然后是头和尾巴 | ordinal_or_secondary | S1T21 |
| 蜥蜴状。 | 1 | 蜥蜴状 | other_unparsed | S1T294 |

#### S327

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 头短，尾巴短，脖子长，腿中上。 | 1 | 腿中上 | other_unparsed | S1T43 |

#### S328

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 尾巴和腿都比较长，差不多，头加脖子比较短。 | 1 | 差不多 | global_balance | S1T95 |
| 尾巴特别短，尾巴明显比腿短，头和脖子加在一起比较长。 | 1 | 尾巴明显比腿短 | other_unparsed | S1T23 |
| 尾巴短，头和脖子都比腿长。 | 1 | 头和脖子都比腿长 | other_unparsed | S2T82 |
| 尾巴短，脖子比对长。 | 1 | 脖子比对长 | other_unparsed | S2T227 |
| 尾巴长，头和脖子都很短，都比腿短。 | 1 | 都比腿短 | other_unparsed | S2T39 |
| 腿和尾巴差不多长，都是中等长度，头和脖子都比较长。 | 1 | 都是中等长度 | other_unparsed | S1T52 |

#### S329

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 按错了。 | 2 | 按错了 | other_unparsed | S1T185, S1T251 |
| 全身都长。 | 1 | 全身都长 | other_unparsed | S1T70 |
| 尾巴长，其次是腿，然后是头和脖子。 | 1 | 然后是头和脖子 | ordinal_or_secondary | S1T87 |

#### S331

| text | count | un_pro_items | categories | sample_trials |
| --- | --- | --- | --- | --- |
| 均等。 | 1 | 均等 | global_balance | S1T154 |
| 头长，尾巴长，腿也还行。 | 1 | 腿也还行 | other_unparsed | S1T25 |
| 躯干长，头短。 | 1 | 躯干长 | body_geometry | S1T29 |

## 编码规则改进建议

1. 本轮已加入 `其他/其余/另外/剩下/剩余` 的补集指代：如 `腿短，其他长` 会把 `其他` 解析为未点名的三个部位；`只有腿很长` 会编码为腿长、其他短。后续需要重点检查残留的 `other_reference` 是否属于更复杂的比较句或计数句。
2. 本轮已把 `达到最长/最短长度` 简化为绝对 `长/短`：center 分别为 0.75/0.25，region 分别为 `>0.5`/`<0.5`。若之后需要更严格端点，可再把这类规则单独改成 0.9/0.1。
3. 本轮已加入跨分句排序和 `次之` 逻辑：如 `腿最长，脖子第二，尾巴第三，头最短` 编码为 strict ranking；如 `头最长，脖子次之，腿、尾巴极短` 编码为 `头 > 脖子 > 腿/尾巴`；如 `脖子和尾巴明显长，头最短，腿次之` 编码为 `脖子/尾巴 > 腿 > 头`。并列部位只作为同一层级参与组间比较，不额外加入组内相等或大小关系。
4. 本轮已加入否定 direct 描述和限定范围最高级：`不长/不是很长/并非很长` 编码为短，`不短/不是很短/并非很短` 编码为长；`不是最短/并非最长` 暂不编码；`脖子是头、尾巴、脖子里最短` 只编码 `脖子 < 头`、`脖子 < 尾巴`，不涉及未出现在限定范围内的腿。
5. 计数抽象先不要强行编码为单个 `A,b`：例如 `两长两短`、`三个部位很长`、`最长的两个/最短的两个`。这类语义通常是多个区域的并集，单个凸 region 表达不了；若要编码，需要扩展为 multi-region 或结合刺激上下文。
6. 全局形态词需要人工定义语义：`比较均衡/匀称/协调/高大/体型中等`。其中 `均衡/匀称` 可考虑映射为四维 pairwise equality；`高大` 可考虑映射为多数或全部维度偏长，但这需要你确认。
7. `比例` 句式要加强 group-sum/ratio parser：如 `头和脖子的比例大于腿和尾巴` 可近似编码为 `head + neck > leg + tail`，但如果被试真的在说比例而非总和，需要另定语义。
8. 身体几何/方位描述需要任务图像约定：如 `头在躯干之下`、`躯体下方比上面高`。这可能涉及 `body_ori` 或视觉布局，不应只靠四个长度维度猜。
9. `不一样/各不相同` 是非凸或补集语义：当前单个 `A,b` 难以表达，应标记为 unsupported 或引入 disjunctive region 表示。
10. `假设X无关`、`选错了`、`不知道` 更像 meta 策略或信心报告，建议继续不编码为 center/region，但在诊断表中保留。

