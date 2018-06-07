### chapter 1: 线性回归

## 符号介绍

输入数据 $ X = [x^{(1)}, x^{(2)}, \ldots, x^{(n)}]^T $ <br>
输出标签 $ Y = [y^{(1)}, y^{(2)}, \ldots, y^{(n)}]^T $ <br>
第$i$个输入数据 $x^{(i)}$ <br>
第$i$个输出数据 $y^{(i)}$ <br>
第$i$个数据的第$j$个特征 $x_j^{(i)} (1 <= j <= m)$ <br>

## 概念

- 线性判别分析


## 知识点

Q1. X必须是满秩的, 倘若不是, 怎么办 <br>
A1. 倘若不是满秩的, 则$X^TX$不存在逆, 无法求解,只能通过正则化限制$\theta$的范围来获得可行解

Q2. 正则化的几何意义以及其公式 <br>
A2. 见下文推导过程

Q3. 非参数学习算法与参数学习算法的定义, 区别 <br>
A3. 定义: 训练完成后是否需要保存训练数据 区别:非参数学习算法训练完成后不需要保存参数(LR), 而参数学习算法需要(LWR)

Q4. 简述LWR算法过程 <br>
A4. 代价函数$J(\theta)=\frac{1}{2}\sum_{i=1}^{i=n} {w_i(h_{\theta}(x^{(i)}) - y^{(i)})^2}$其中$w_i=exp(-\frac{(x^{(i)} - x)}{2\gamma^2})$<br>每当有新数据加入train data时候原数据仍然需要保存

Q5. 二分类问题下逻辑回归的代价函数 <br>
A5. 
$$h_{\theta}(x)=\frac{1}{1+e^{\theta^Tx}}$$
$$P(y|x;\theta)=(h_{\theta}(x^{(i)}))^{y^{(i)}}(1 - h_{\theta}(x^{(i)}))^{1- y^{(i)}}$$
$$J(\theta)=logL(\theta)=\prod_{i=1}^{i=n}{logP(y|x,\theta)}=\sum_{i=1}^{i=n}{y^{(i)}{\rm{log}}(h_{\theta}(x^{(i)})) + (1-y^{(i)}){\rm{log}}(1-h_{\theta}(x^{(i)}))}$$

> $P(y|x,\theta)$与$P(y|x;\theta)$区别:
> 前者$\theta$是变量而后者是常量或者给定值

Q6. 牛顿法的求解过程 <br>
A6. 牛顿法采用不断迭代$x_{(i+1)}=x_i - \frac{f^{i阶导数}(x_i)}{f^{i+1阶导数}(x_{(i+1)})}$求解出代价函数导数的零点获得最优解

Q7. 从二分类到多分类的几种方法以及其各自的优缺点 <br>
A7. 1对其余: 分出单独某一类以及其他类; 1对1: 区分待判定样本是第$i$类还是第$j$类(需要$n(n-1)/2$个分类器然后再统计最高票数)

Q8. 简述LDA的算法过程<br>
A8. 面试不常考,在此仅仅给出引用, 西瓜书P60

## 公式推导以及证明
> 预备知识: (需翻墙) <br>
> [矩阵的迹][1] <br>
> [矩阵求导][2] <br>

- 线性回归问题描述以及采用LMS作为cost function其最优解 <br><br>
假设函数$$h_{\theta}(x)=\sum_{i=1}^{i=m} {\theta_ix_i}=\theta^Tx$$
代价函数$$J(\theta)=\frac{1}{2}\sum_{i=1}^{i=n} {(h_{\theta}(x^{(i)}) - y^{(i)})^2}=\frac{1}{2}\sum_{i=1}^{i=n} {(\theta^Tx^{(i)} - y^{(i)})^2}$$
$$ \nabla_{\theta}J(\theta) = \frac{1}{2}\nabla_{\theta}(X\theta - Y)^T(X\theta - Y)$$
$$ =\frac{1}{2}\nabla_{\theta}tr(\theta^TX^TX\theta - Y^TX\theta - \theta^TX^TY + Y^TY)$$
$$ =\frac{1}{2}\nabla_{\theta}tr(\theta^TX^TX\theta - 2Y^TX\theta)$$
$$ =\frac{1}{2}\nabla_{\theta}tr(\theta^TX^TX\theta - 2Y^TX\theta)$$
$$ =\frac{1}{2}tr(X^TX\theta + {\theta^TX^TX}^T - 2Y^TX)$$
$$ =X^TX\theta + Y^TX$$
$$ =X^TX\theta + X^TY$$
令 $\nabla_{\theta}J(\theta) = 0$可得 $$\theta=(X^TX)^{-1}X^TY$$

- 加入L1, L2正则其表达式以及其最优解
$$J(\theta)=\frac{1}{2}(X\theta - Y)^T(X\theta - Y) + \frac{\lambda}{2}\sum_{j=1}^{j=n}|\theta_j|^q$$
采用拉格朗日乘子可得
$$\theta=(X^TX)^{-1}X^TY \quad \rm{s.t. \,} \it{\sum_{j=1}^{j=n}|\theta_j|^q < \lambda}$$

- 使用概率的方法推导LR求解过程
$y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}$其中$\epsilon^{(i)}$表示经验误差或者非模型影响因素.<br>
由于$y^{(i)}$的分布与$\epsilon$保持一致,则有
$$P(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp(\frac{(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2})$$
采用MLE作为代价函数可得:
$$J(\theta)=L(\theta)=log\prod_{i=1}^{i=n}{P(y^{(i)}|x^{(i)};\theta)}$$
$$=\sum_{i=1}^{i=n}{log\frac{1}{\sqrt{2\pi}\sigma}exp(\frac{(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2})}$$
$$=nlog\frac{1}{\sqrt{2\pi}\sigma}+\frac{1}{2\sigma^2}\sum_{i=1}^{i=n}{(y^{(i)}-\theta^Tx)^2}$$
> 由上可知, 采用MLE的方法计算LR与LMS做为代价函数是等价的


[1]: https://en.wikipedia.org/wiki/Trace_(linear_algebra)
[2]: https://en.wikipedia.org/wiki/Matrix_calculus

