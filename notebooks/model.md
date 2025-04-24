1.符号定义

特征空间  假设每个刺激（stimuli)都由$D$个标准化后的特征参数决定，特征空间就是所有可能刺激的集合$\mathbf{X} = [0, 1]^D = \{(x_i)_{i=1}^D \mid x_i \in [0, 1]\}$，其中${{x}_{i}}$表示第$i$个维度的特征值。在当前的任务中，$D=4$，因此在几何上可以视为一个四维单位超立方体。

类别判断空间  假设一共有$N$个类别，所有可能的类别集合为$\mathbf{C}=\{{{C}_{1}},...,{{C}_{N}}\}$，则类别判断空间就是分布在这$N$个类别上的所有概率分布所构成的集合
${{\Delta }^{N-1}}=\mathcal{P}(\mathbf{C})=\{(p({{C}_{i}}))_{i=1}^{N}\mid p({{C}_{i}})\ge 0,\sum\limits_{i=1}^{N}{p}({{C}_{i}})=1\}$。

假设空间  被试的分类标准可以表示为从特征空间到类别判断空间的映射函数，即$b:\mathbf{X}\to {{\Delta }^{N-1}}$，所有连续映射函数的集合就是被试的假设空间$\mathcal{H}={{C}_{0}}(\mathbf{X},{{\Delta }^{N-1}})$。由于$\mathcal{H}={{C}_{0}}(\mathbf{X},{{\Delta }^{N-1}})$是一个无穷维的函数空间，难以直接用于计算建模。
因此，我们采用其有限维子空间$\mathcal{H_0}$作为可行的假设空间。

接下来对该子空间进行参数化，用$k$表示从有限的特征空间分割方法集合中选择的第$k$种具体方法，$k \in \{1, 2, \dots, K\}$，其中$K$是可选分割方法的总数。用$\beta$刻画分割边界的软硬程度，也就是分类的不确定性。因此，给定特定的分割方法，能够计算任一刺激$\mathbf{x} \in \mathbf{X}$属于类别${C}_{i}$的概率：
	$p(C_i | \mathbf{x},\theta) = \frac{e^{-\beta d_i(x,\theta)}}{\sum_{j=1}^{N} e^{-\beta d_j(x,\theta)}}
    $
其中$d_i$表示刺激$\mathbf{x} \in \mathbf{X}$到分割区域${{C}_{i}}$的中心点${{\mathbf{c}}_{i}}=({{c}_{i1}},...,{{c}_{iD}})$的欧几里得距离 $d_i(x,\theta) = \sqrt{{\sum_{j=1}^{D} (x_j - c_{i,j})^2}}$。


2.贝叶斯更新过程

在实验开始前，假设被试持有“无偏好”或“相对无信息”的先验假设。
即假设$k$的先验是均匀分布，$p(k)=\frac{1}{K},\quad k\in \{1,2,\ldots ,K\}$，即被试对所有可选分割方法没有偏好；同时假设$\beta$也满足在指定区间$[\beta_{\min},\beta_{\max}]$（${{\beta }_{\min }},{{\beta }_{\max }}>0$)上的均匀分布。

实验包括三个步骤：被试看到刺激$\mathbf{x}$、基于当前假设$\theta$做出选择$c$、获得实验系统生成的反馈$r$。那么，似然函数$p(\mathbf{x}, c, r | \theta) = p(r | \mathbf{x}, c, \theta) \cdot p(c | \mathbf{x}, \theta) \cdot p(\mathbf{x} | \theta)$就是在给定假设参数$\theta$的条件下观测到数据$(\mathbf{x}, c, r)$的概率。

因此，对似然函数$p(\mathbf{x}, c, r | \theta)$可以进行如下分解： 
	$p(\mathbf{x}, c, r | \theta) = p(r | \mathbf{x}, c, \theta) \cdot p(c | \mathbf{x}, \theta) \cdot p(\mathbf{x} | \theta)$	
其中$p(\mathbf{x} | \theta)$是一个与$\theta$无关的常数，因为在实验中刺激是由固定程序独立生成的。而$p(r | \mathbf{x}, c, \theta)$反映的是实验对于反馈系统的设计，在本实验中反馈是确定性的，只取决于选择$c$与刺激$\mathbf{x}$的真实类别$C_{\text{true}}$是否一致：

[0-1反馈条件]
	$p(r | c, \mathbf{x}, \theta) = \begin{cases}1, & \text{if } r = 1 \text{ and } c = C_{\text{true}} \\1, & \text{if } r = 0 \text{ and } c \neq C_{\text{true}} \\0, & \text{else}\end{cases}$	

[0-0.5-1反馈条件]
$p(r | c, \mathbf{x}, \theta) = \begin{cases}1, & \text{if } r = 1 \text{ and } c = C_{\text{true}} \\1, & \text{if } r = 0.5 \text{ and } c \neq C_{\text{true}} \text{ and } Family(c) = Family(C_{\text{true}}) \\1, & \text{if } r = 0 \text{ and } c \neq C_{\text{true}} \text{ and } Family(c) \neq Family(C_{\text{true}})\\0, & \text{else}\end{cases}$

因此，似然函数$p(\mathbf{x}, c, r | \theta)$可以写作：
	$p(\mathbf{x}, c, r | \theta) = \begin{cases}p(c | \mathbf{x}, \theta), & \text{if } r = 1 \\1-p(c | \mathbf{x}, \theta), & \text{if } r = 0 \end{cases}$

根据贝叶斯定理，单试次的更新过程可以表示为：
	$p(\theta |{{\mathbf{x}}_{t}},{{c}_{t}},{{r}_{t}})\ =\ \frac{p({{\mathbf{x}}_{t}},{{c}_{t}},{{r}_{t}}|\theta )p(\theta )}{p({{\mathbf{x}}_{t}},{{c}_{t}},{{r}_{t}})}$
对于$T$个试次的连续学习过程，数据则变为$D=({{\mathbf{x}}_{t}},{{c}_{t}},{{r}_{t}})_{t=1}^{T}$，将似然进行累乘：
	$p(\theta |{{D}_{1:T}})=\frac{p(\theta )\prod\limits_{t=1}^{T}{p}({{D}_{t}}|\theta )}{p({{D}_{1:T}})}$

3.模型拟合

我们先计算对数后验概率，其中${{Z}_{T}}=p({{D}_{1:t}})$是一个归一化常数：
	$\log p(\theta |{{D}_{1:T}})=\log p(\theta )+\sum\limits_{t=1}^{T}{\log }p({{D}_{t}}|\theta )-\log {{Z}_{T}}$
然后通过最大后验概率(maximum a posteriori, MAP)的方法来估计模型参数，将目标函数写作：
	${{\mathcal{L}}_{\text{MAP}}}(\theta )=-[\log p(\theta )+\sum\limits_{t=1}^{T}{\log }p({{D}_{t}}|\theta )]$

在实际拟合中，由于$k$是一个离散变量，$\beta$是一个连续变量，因此，我们在固定$k$的情况下，采用数值优化方法寻找能够最小化目标函数的${{\hat{\beta }}_{k}}$值：
	${{\hat{\beta }}_{k}}\ =\ \arg {{\min }_{\beta \in [{{\beta }_{\min }},{{\beta }_{\max }}]}}{{\mathcal{L}}_{\text{MAP}}}(k,\beta )$
计算得到所有k对应的后验概率$p(k,{{\hat{\beta }}_{k}}|{{D}_{1:T}})$，选择整体后验概率最大的$k$和相应的${{\hat{\beta }}_{k}}$作为MAP估计值。最后，为了追踪每个k的后验概率的变化，我们逐渐增加用于拟合的${{D}_{1:T}}$的试次数量。


4.非理性模块

(1)知觉

被试在知觉过程中可能对刺激的感知不完全准确，导致决策和学习并不基于准确的刺激特征。假设真实刺激特征向量为$\mathbf x\in[0,1]^D$，被试无法直接读取，需要经历一次感知变换：
	$\tilde{\mathbf x}\;=\;\mathbf x+\boldsymbol\varepsilon,\qquad
\boldsymbol\varepsilon\sim\mathcal N\!\bigl(\mathbf 0,\;\Sigma\bigr)$
其中$\Sigma=\operatorname{diag}(\sigma_1^{2},\dots,\sigma_D^{2})$是通过任务1b测量得到的被试对于每个特征的知觉误差。

(2)记忆

被试无法完全记住每个试次的信息，可能会出现记忆衰退或者只记住部分关键信息。因此，我们以一个指数衰减的形式来降低个体对于之前试次的记忆权重：
	$\log {{p}^{\text{(forget)}}}(\theta |{{\mathcal{D}}_{1:T}})=\log p(\theta )+\sum\limits_{j=1}^{T}{(}{{w}_{0}}+{{\gamma }^{T-j}})\log p({{D}_{j}}|\theta )-\log {{Z}_{T}}$
其中γ和w0都是被试层面的参数，$\gamma \in [0,1]$表示记忆衰减率，${{w}_{0}}\in [0,1]$表示基础记忆强度。

此时最大后验估计的目标函数也会相应地变为：
	$\mathcal{L}_{\text{MAP}}^{\text{(forget)}}(k,\beta )=-[\log p(k)+\log p(\beta )+\sum\limits_{j=1}^{T}{(}{{w}_{0}}+{{\gamma }^{T-j}})\log p({{D}_{j}}|k,\beta )]$


(3)假设集跳转

为了模拟从行为上发现的“顿悟”等现象，我们假设被试不是像基线模型一样，在一开始就产生完备的假设集，而是每次只能拥有部分假设集，并在学习中不断发生假设集之间的跳转：
	$\mathcal H_t=\{k_{(1)},\dots ,k_{(M)}\}\subset\mathcal K,
\qquad |\mathcal H_t|=M$
每个试次的假设集$\mathcal H_t=\{k_{(1)},\dots ,k_{(M)}\}\subset\mathcal K,
\qquad |\mathcal H_t|=M$是假设全集$\mathcal H_t=\{k_{(1)},\dots ,k_{(M)}\}\subset\mathcal K, \qquad |\mathcal H_t|=M$的一个子集，M是事先设置的假设个数。

通过考虑探索(exploration)和利用(exploitation)这两种取向之间的权衡，设置以下三种假设跳转规则，得到下一试次的假设集：
	$\mathcal H_{t+1}
   =\mathcal S_1^{(t)}\;\cup\;
    \mathcal S_2^{(t)}\;\cup\;
    \mathcal S_3^{(t)},\qquad
|\mathcal H_{t+1}|=M$
$M=m_1+m_2+m_3,\qquad 
m_i\in\mathbb N,\; 0\le m_i\le K$

第一种规则为保留上一试次中后验概率最高，即表现足够好的假设：
	$\mathcal{S}_{1}^{(t)}=\text{arg}\text{To}{{\text{p}}_{{{m}_{1}}}}{{\{p(k,{{\hat{\beta }}_{k}}|{{\mathcal{D}}_{1:T}})\}}_{k\in \mathcal{K}}}$

第二种规则为联想到一些与当前刺激相关的假设：
	$\mathcal{S}_{2}^{(t)}=\text{arg}\text{To}{{\text{p}}_{{{m}_{2}}}}{{\{{{s}_{t}}(k)\}}_{k\in {{\mathcal{C}}_{t}}}}$
其中${{s}_{t}}(k)$表示当前刺激与分割方法k的同一类别的相似度，相似度越大，表示越能通过当前刺激联想到k。

第三种规则为随机从假设全集中抽取一些新的假设：
	$\mathcal S_3^{(t)}
   \;\sim\;\text{UniformWithoutReplacement}\Bigl(
          \mathcal K\setminus\bigl(\mathcal S_1^{(t)}\cup\mathcal S_2^{(t)}\bigr),\;
          m_3\Bigr).$