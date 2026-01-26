假设某个策略为最优策略，在通常的定义中，这等价于该策略在任何状态的状态价值函数的值都为最大

在**Policy Gradient**中，我们使用  $J(\theta)$  来衡量策略的优劣，一个策略的 $J(\theta)$ 定义为在所有轨迹上的 $R(\tau)$ 的期望：

$$
J(\theta)=\mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]
$$

其中 $\theta$ 代表策略函数的参数， $\tau$ 为一条轨迹， $R(\tau)$ 则是这条轨迹的总的折扣回报

则 $J(\theta)$ 对于 $\theta$ 的梯度为：

$$
\begin{aligned}
\nabla_{\theta}J(\theta)&=\nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]\\
&=\nabla_{\theta}\sum_\tau P(\tau|\theta)R(\tau)\\
&=\sum_\tau \nabla_\theta [P(\tau|\theta)R(\tau)]\\
&=\sum_\tau R(\tau) \nabla_\theta P(\tau|\theta)\\
&=\sum_\tau R(\tau)[P(\tau|\theta) \nabla_\theta \ln P(\tau|\theta)]\\
&=\sum_\tau P(\tau|\theta) R(\tau) \nabla_\theta \ln P(\tau|\theta)\\
&=\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \nabla_\theta \ln P(\tau|\theta)] \qquad \qquad (1)
\end{aligned}
$$

我们定义 $T$ 为轨迹的长度，意为轨迹中动作的数量，例如轨迹 $\{s_0,a_0,r_0,s_1,a_1,r_1,s_2\}$ 的长度 $T$ 为 $2$ ，则：

$$
\begin{aligned}
\ln P(\tau|\theta)&=\ln \left[P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\right]\\
&=\ln P(s_0) + \ln \left[\prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\right]\\
&=\ln P(s_0) + \sum_{t=0}^{T-1}\ln[\pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)]\\
&=\ln P(s_0) + \sum_{t=0}^{T-1} [\ln \pi_\theta(a_t|s_t) + \ln P(s_{t+1}|s_t,a_t)]\\
&=\ln P(s_0) + \sum_{t=0}^{T-1} \ln \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \ln P(s_{t+1}|s_t,a_t)
\end{aligned}
$$

则：

$$
\begin{aligned}
\nabla_\theta \ln P(\tau|\theta)&=\nabla_\theta\left[\ln P(s_0) + \sum_{t=0}^{T-1} \ln \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \ln P(s_{t+1}|s_t,a_t)\right]\\
&=\nabla_\theta \sum_{t=0}^{T-1} \ln \pi_\theta(a_t|s_t) + \nabla_\theta \ln P(s_0) + \nabla_\theta \sum_{t=0}^{T-1} \ln P(s_{t+1}|s_t,a_t)\\
&=\nabla_\theta \sum_{t=0}^{T-1} \ln \pi_\theta(a_t|s_t)\\
&=\sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t|s_t)
\end{aligned}
$$

代入到**公式（1）**，有：

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \nabla_\theta \ln P(\tau|\theta)]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right] \qquad \qquad (2)
\end{aligned}
$$

对于上述公式的直观理解是，如果一条轨迹的总折扣回报越大，则越发强化此轨迹上的所有动作选择

基于此理解，更好的方式是每个动作选择应该由其 $Q$ 值进行评判，若 $Q$ 值大则强化之

此公式还能进行进一步的优化：

由于不同的轨迹长短不一，我们定义序列 $X_\tau(t)$ :

$$
X_\tau(t)=
\begin{cases}
R(\tau) \nabla_\theta \ln \pi_\theta(a_t|s_t)=\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{T-1}\gamma^k r(k),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$

将序列 $X_\tau(t)$ 进行分解：

$$
X_\tau(t)=Past_\tau(t)+Future_\tau(t)
$$

其中：

$$
Past_\tau(t)=
\begin{cases}
\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$


$$
Future_\tau(t)=
\begin{cases}
\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$


将 $X_\tau(t)$ 代入**公式（2）**：

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau)\sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} R(\tau) \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}
\left[\sum_{t=0}^\infty X_\tau(t)\right]\\
&=\sum_{\tau}P(\tau|\theta)\sum_{t=0}^\infty X_\tau(t)\\
&=\sum_{\tau}\sum_{t=0}^\infty P(\tau|\theta)X_\tau(t)\\
&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)X_\tau(t)\\
&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)[Past_\tau(t)+Future_\tau(t)]\\
&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Past_\tau(t)+\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Future_\tau(t)\qquad \qquad (3)
\end{aligned}
$$

而：

$$
\begin{aligned}
\sum_{\tau}P(\tau|\theta)Past_\tau(t)&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]+\sum_{\tau:T(\tau) \le t}P(\tau|\theta)\cdot 0\\
&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]\qquad \qquad (4)
\end{aligned}
$$

接下来需要定义**吸收态**，对于任意的终止态，即使其外显属性不同，但仍将其视作同一个状态 $s_{\perp}$ ，称作**吸收态**

**吸收态不同于任何一个中间态**，状态空间由 $\\{s_{\perp}\\}$ 和所有中间态构成的集合并成

对于任意 $T(\tau)>t$ 的轨迹，都可以将其看作三元组 $(h_t,a_t,\tau')$ ，分别代表**轨迹的历史部分**、**当前动作**、**轨迹的剩余部分**， $h_t$ 不可包含吸收态

具体来说，轨迹集合和三元组集合存在一个双射，则求和可以分解到三个维度上，且其概率可以分解为三者的概率之积

对于任何一个 $h_t$ ，其概率为多步的概率之积，任何长度不足的轨迹都没有 $h_t$ ，这就隐式包含了对轨迹的长度约束

则：

$$
\begin{aligned}
\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]&=\sum_{\tau:T(\tau) > t}P(h_t)\pi(a_t|h_t)P(\tau'|h_t,a_t)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]\\
&=\sum_{\tau:T(\tau) > t}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\\
&=\sum_{h_t}\sum_{a_t}\sum_{\tau'}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\sum_{a_t}\pi(a_t|s_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\sum_{a_t}\pi(a_t|s_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\sum_{a_t}\pi(a_t|s_t)\left[\frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)}\right]\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\sum_{a_t}\nabla_\theta \pi_\theta(a_t|s_t)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\nabla_\theta \sum_{a_t}\pi_\theta(a_t|s_t)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\nabla_\theta 1\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right] \cdot 0\\
&=0
\end{aligned}
$$

代回**公式（4）**，得：

$$
\begin{aligned}
\sum_{\tau}P(\tau|\theta)Past_\tau(t)&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]\\
&=0
\end{aligned}
$$

再代回**公式（3）**，得：

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Past_\tau(t)+\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Future_\tau(t)\\
&=\sum_{t=0}^\infty 0+\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Future_\tau(t)\\
&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Future_\tau(t)\qquad \qquad (5)
\end{aligned}
$$

单看 **公式（5）** 中的：

$$
\begin{aligned}
\sum_{\tau}P(\tau|\theta)Future_\tau(t)&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]+\sum_{\tau:T(\tau) \le t}P(\tau|\theta)\cdot 0\\
&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]\qquad \qquad (6)
\end{aligned}
$$

和上述的分解类似，可以将其分解为：

$$
\begin{aligned}
\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]&=\sum_{\tau:T(\tau) > t}P(h_t)\pi(a_t|h_t)P(\tau'|h_t,a_t)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]\\
&=\sum_{\tau:T(\tau) > t}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\\
&=\sum_{h_t}\sum_{a_t}\sum_{\tau'}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\sum_{k=t}^{T-1} \gamma^k r(k)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\sum_{k=t}^{T-1} \gamma^{k-t} r(k)\qquad \qquad (7)
\end{aligned}
$$

观察这部分：

$$
\sum_{\tau'}P(\tau'|h_t,a_t)\sum_{k=t}^{T-1} \gamma^{k-t} r(k)
$$

会发现它就是：

$$
Q(s_t,a_t)
$$

代入**公式（7）**，得：

$$
\begin{aligned}
\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\sum_{k=t}^{T-1} \gamma^{k-t} r(k)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\\
&=\sum_{h_t}\sum_{a_t}\sum_{\tau'}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\\
&=\sum_{h_t}\sum_{a_t}\sum_{\tau'}P(h_t)\pi(a_t|h_t)P(\tau'|h_t,a_t)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\\
&=\sum_{\tau:T(\tau) > t}P(h_t)\pi(a_t|h_t)P(\tau'|h_t,a_t)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\\
&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\qquad \qquad (8)
\end{aligned}
$$

显然可以定义序列 $Y_\tau (t)$ ：

$$
Y_\tau (t)=
\begin{cases}
\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$

则：

$$
\begin{aligned}
\sum_\tau P(\tau|\theta) Y_\tau (t)&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)+\sum_{\tau:T(\tau) \le t}P(\tau|\theta)\cdot 0\\
&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)
\end{aligned}
$$

代入 **公式（8）** 得：

$$
\begin{aligned}
\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\\
&=\sum_\tau P(\tau|\theta) Y_\tau (t)
\end{aligned}
$$

再代回 **公式（6）** 得：

$$
\begin{aligned}
\sum_{\tau}P(\tau|\theta)Future_\tau(t)&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]\\
&=\sum_\tau P(\tau|\theta) Y_\tau (t)
\end{aligned}
$$

再代回**公式（5）**，得：

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Future_\tau(t)\\
&=\sum_{t=0}^\infty \sum_\tau P(\tau|\theta) Y_\tau (t)\\
&=\sum_\tau \sum_{t=0}^\infty P(\tau|\theta) Y_\tau (t)\\
&=\sum_\tau P(\tau|\theta) \sum_{t=0}^\infty Y_\tau (t)\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^\infty Y_\tau (t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)+\sum_{t=T}^\infty 0\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\qquad \qquad (9)
\end{aligned}
$$

***

下面介绍**baseline**：

我们要证明：

$$
\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]=0 \qquad \qquad (10)
$$

其中 $b(s_t)$ 是一个只依赖于 $s_t$ 的函数，我们定义序列 $Z_\tau (t)$ ：

$$
Z_\tau (t)=
\begin{cases}
\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$

则：

$$
\begin{aligned}
\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]&=\sum_\tau P(\tau)\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_\tau P(\tau)\sum_{t=0}^\infty Z_\tau (t)\\
&=\sum_{t=0}^\infty \sum_\tau P(\tau) Z_\tau (t)\qquad \qquad (11)
\end{aligned}
$$

而：

$$
\begin{aligned}
\sum_\tau P(\tau)Z_\tau (t)&=\sum_{\tau:T(\tau)>t}P(\tau)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)+\sum_{\tau:T(\tau) \le t}P(\tau) \cdot 0\\
&=\sum_{\tau:T(\tau)>t}P(\tau)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{\tau:T(\tau)>t}P(h_t)\pi(a_t|h_t)P(\tau'|h_t,a_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{\tau:T(\tau)>t}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{h_t}\sum_{a_t}\sum_{\tau'}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t b(s_t)\left[\frac{\nabla_\theta \pi_\theta (a_t|s_t)}{\pi_\theta (a_t|s_t)}\right]\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\gamma^t b(s_t)\nabla_\theta \pi_\theta (a_t|s_t)\\
&=\gamma^t \sum_{h_t}P(h_t)\sum_{a_t}b(s_t)\nabla_\theta \pi_\theta (a_t|s_t)\\
\end{aligned}
$$

因为 $b(s_t)$ 只依赖于 $s_t$ ，而 $h_t$ 确定后 $s_t$ 就确定了，这意味着 $h_t$ 确定了后 $b(s_t)$ 就确定了，则：

$$
\begin{aligned}
\sum_\tau P(\tau)Z_\tau (t)&=\gamma^t \sum_{h_t}P(h_t)\sum_{a_t}b(s_t)\nabla_\theta \pi_\theta (a_t|s_t)\\
&=\gamma^t \sum_{h_t}P(h_t)b(s_t)\sum_{a_t}\nabla_\theta \pi_\theta (a_t|s_t)\\
&=\gamma^t \sum_{h_t}P(h_t)b(s_t)\nabla_\theta \sum_{a_t}\pi_\theta (a_t|s_t)\\
&=\gamma^t \sum_{h_t}P(h_t)b(s_t)\nabla_\theta 1\\
&=\gamma^t \sum_{h_t}P(h_t)b(s_t) \cdot 0\\
&=0
\end{aligned}
$$

将此结果代入**公式（11）**，得：

$$
\begin{aligned}
\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]&=\sum_{t=0}^\infty \sum_\tau P(\tau) Z_\tau (t)\\
&=\sum_{t=0}^\infty 0\\
&=0
\end{aligned}
$$

这就证明了**等式（10）**，由于其成立，我们可以在 **公式（9）** 中任意加减它：

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t Q(s_t,a_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]-\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t [Q(s_t,a_t)-b(s_t)]\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]
\end{aligned}
$$

显然的， $V(s_t)$ 是一个 $b(s_t)$ （而且它是一个被普遍采用的**baseline**），则有：

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t [Q(s_t,a_t)-b(s_t)]\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t [Q(s_t,a_t)-V(s_t)]\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]\qquad \qquad (12)
\end{aligned}
$$

其中 $Q(s_t,a_t)-V(s_t)$ 被称为**优势函数（Advantage Function）**，记作：

$$
A(s_t,a_t)=Q(s_t,a_t)-V(s_t)
$$

#### **Actor-Critic方法（AC）**：

这是一类方法的统称，回顾上述的**公式（9）**，会发现它需要一个策略网络和一个价值网络，这个策略网络就是所谓的**Actor**，意为行动者，价值网络则称之为**Critic**，意为评判者，这个名字是因为它给出的值将被用来衡量动作选择的好坏

对于**公式（12）**，它似乎需要两个价值网络来分别拟合 $Q$ 和 $V$ ，但实际上它可以进行变形：

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t [Q(s_t,a_t)-V(s_t)]\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t [r(t)+\gamma V(s_{t+1})-V(s_t)]\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]\qquad \qquad (13)
\end{aligned}
$$

这样只需要一个价值网络即可，它仍然是**AC**，由于历史原因，**公式（13）**被称为**标准AC**、单步**AC**，其对应的算法就是**A2C（Advantage Actor-Critic）**，算法名字来自于其使用的**优势函数（Advantage Function）**

因为**Critic**是神经网络，给不出准确的价值函数，所以**AC**方法是有偏差的

**Critic**的更新使用时序差分即可

#### 多步A2C

前面的 **公式（7）** 能导出另一种更普遍的形式

具体来说，可以对其中轨迹的未来部分 $\tau'$ 进行进一步的划分，划分为前面 $n$ 个**reward**及其中间的部分，例如取 $n=3$ ，有：

$$
\begin{aligned}
\tau'&=\{r_5,s_6,a_6,r_6,s_7,a_7,r_7,s_8,a_8,r_8,s_\perp\}\\
&=\{r_5,s_6,a_6,r_6,s_7,a_7,r_7|s_8,a_8,r_8,s_\perp\}\\
&=\{\tau'_n|\tau''\}
\end{aligned}
$$

特别的，如果 $\tau'$ 剩余的**reward**不足 $n$ 个，则取全部**reward**及其中间的部分，如 $n=3$ 时，有：

$$
\begin{aligned}
\tau'&=\{r_5,s_6,a_6,r_6,s_\perp\}\\
&=\{r_5,s_6,a_6,r_6|s_\perp\}\\
&=\{\tau'_n|\tau''\}
\end{aligned}
$$

则 **公式（7）** 中的：

$$
\begin{aligned}
\sum_{\tau'}P(\tau'|h_t,a_t)\sum_{k=t}^{T-1} \gamma^{k-t} r(k)&=\sum_{\tau'}P(\tau'_n|h_t,a_t)P(\tau''|h_t,a_t,\tau'_n)\sum_{k=t}^{T-1} \gamma^{k-t} r(k)\\
&=\sum_{\tau'_n}\sum_{\tau''}P(\tau'_n|h_t,a_t)P(\tau''|h_t,a_t,\tau'_n)\sum_{k=t}^{T-1} \gamma^{k-t} r(k)\\
&=\sum_{\tau'_n}\sum_{\tau''}P(\tau'_n|h_t,a_t)P(\tau''|h_t,a_t,\tau'_n)\left[\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k) + \sum_{k=t+n}^{T-1} \gamma^{k-t} r(k)\right]\\
&=\sum_{\tau'_n}\sum_{\tau''}P(\tau'_n|h_t,a_t)P(\tau''|h_t,a_t,\tau'_n)\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k) + \sum_{\tau'_n}\sum_{\tau''}P(\tau'_n|h_t,a_t)P(\tau''|h_t,a_t,\tau'_n)\sum_{k=t+n}^{T-1} \gamma^{k-t} r(k)\\
&=\sum_{\tau'_n}P(\tau'_n|h_t,a_t)\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\sum_{\tau''}P(\tau''|h_t,a_t,\tau'_n) + \sum_{\tau'_n}P(\tau'_n|h_t,a_t)\sum_{\tau''}P(\tau''|h_t,a_t,\tau'_n)\sum_{k=t+n}^{T-1} \gamma^{k-t} r(k)\\
&=\sum_{\tau'_n}P(\tau'_n|h_t,a_t)\underbrace{\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)}_{A} + \sum_{\tau'_n}P(\tau'_n|h_t,a_t)\gamma^n\sum_{\tau''}P(\tau''|h_t,a_t,\tau'_n)\underbrace{\sum_{k=t+n}^{T-1} \gamma^{k-t-n} r(k)}_{B}\\
&=\sum_{\tau'_n}P(\tau'_n|h_t,a_t)\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k) + \sum_{\tau'_n}P(\tau'_n|h_t,a_t)\gamma^n V(s_{t+n})\\
&=\sum_{\tau'_n}P(\tau'_n|h_t,a_t)\left[\gamma^n V(s_{t+n}) + \sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\right]\\
\end{aligned}
$$

 $A$ 、 $B$ 两部分都只取存在的**reward**求和，特别的，若 $B$ 中所有**reward**都不存在，即轨迹的剩余**reward**数量小于等于 $n$ 时，则其下方的 $V$ 视作 $0$ 

代回 **公式（7）** 得：

$$
\begin{aligned}
\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\sum_{k=t}^{T-1} \gamma^{k-t} r(k)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'_n}P(\tau'_n|h_t,a_t)\left[\gamma^n V(s_{t+n}) + \sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\right]\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'_n}P(\tau'_n|h_t,a_t)\left[\gamma^n V(s_{t+n}) + \sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\right]\sum_{\tau''}P(\tau''|h_t,a_t,\tau'_n)\\
&=\sum_{h_t}\sum_{a_t}\sum_{\tau'_n}\sum_{\tau''}P(h_t)\pi(a_t|s_t)P(\tau'_n|h_t,a_t)P(\tau''|h_t,a_t,\tau'_n)\gamma^t\nabla_\theta \ln \pi_\theta(a_t|s_t)\left[\gamma^n V(s_{t+n}) + \sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\right]\\
&=\sum_{\tau:T(\tau) > t} P(\tau|\theta) \gamma^t\nabla_\theta \ln \pi_\theta(a_t|s_t)\left[\gamma^n V(s_{t+n}) + \sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\right]\\
\end{aligned}
$$

同上从 **公式（8）** 到 **公式（9）** 的推导，可以推得：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\gamma^n V(s_{t+n}) + \sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\right]\nabla_\theta \ln \pi_\theta(a_t|s_t)\right]
$$

上述公式中对于动作的评判多取了几步真实的**reward**（**n-step return**），将其应用到**A2C**中就得到了**多步A2C（n-step A2C）**：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\gamma^n V(s_{t+n}) - V(s_t) + \sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\right]\nabla_\theta \ln \pi_\theta(a_t|s_t)\right]
$$

同样的，为了避免梯度消失，最前面的 $\gamma^t$ 往往省略

**n-step A2C**就是取**n**个真实**reward**的**A2C**，当**n**取 $1$ 的时候，就是普通的**A2C**

对于轨迹剩余部分不足的情况，根据上面提到过的， $V$ 项为 $0$ ，则只取存在的所谓**reward**计算即可

另一个要注意的点就是对于截断情况的处理，即轨迹足够长但是我们实际采样中对其进行了切断，这就导致终止态实际上是一个中间态，并且轨迹不完整，这时候往往采用剩余所有的**reward**的折扣和加上“终止态”的带折扣 $V$ 值作为评判。对于上述公式中的**n**，将其视为变量即可推出可以任意混用**x-step return**，所以对截断情况做这种处理本身不会带来额外的偏差，偏差仍然是截断本身带来的（以及对于最前面折扣的省略、采样本身、**Critic**不准确这三者）

#### **关于方差**：

这是**RL**中的一个重要概念，在实际训练时，采样往往是昂贵的，难以真正采得大量的数据去求期望，所以需要在每条轨迹上做训练，具体来说，当一个方法被称之为方差高时，具体是指两方面：

1. 对于同一状态的同一个动作的好坏判断过大，例如**公式（2）**，对于两条不同的轨迹，若其中包含对于同一状态选择了同一动作的部分，对于此选择好坏的估计取决于 $R(\tau)$ ，而 $R(\tau)$ 则是取决于整个轨迹，具有很大的随机性，这意味着对于此状态此动作的评价可能差别很大

    **AC**的存在能缓解这个问题，拿**标准AC**举例，对于两条轨迹中同一个状态的同一个动作，若其走到了同一个 $s_{t+1}$ ，则两条轨迹中的评判是相同的，若走到了不同的 $s_{t+1}$ ，则二者的差值来源于此步拿到的奖励和下一步的状态价值函数，随机性来源被限制到了一步内，由于神经网络的连续性质，相近的状态往往输出相近的值，而两个不同的 $s_{t+1}$ 由于是同一状态转移而来，往往比较相近，这大大降低了方差，即使考虑网络的更新带来的评价偏移，因为学习率的限制，这个偏移往往会缓慢进行，在短期内方差仍然较小

2. 对于同一状态不同动作，例如说状态 $s$ 具有 $a$ 、 $b$ 俩个动作选择，假设对于二者的评判分别是 $1005$ 和 $1007$ ，这二者的差很小，但是因为 $a$ 、 $b$ 是两个不同的动作，一者的概率增大另一者需要相应减少，因此它们的梯度方向往往是相反的，在**agent**若干次经过此状态时，可能会因为采样而选取不同的动作，在对 $a$ 的梯度下降中走了很大的步子，在对 $b$ 的梯度下降中则同样走了方向相反的很大的步子，这就导致了梯度方差大

    **baseline**就是旨在解决这个问题的，对于上述例子，若**baseline**为 $1000$ ，则二者都减去其之后变为 $5$ 和 $7$ ，再去做梯度下降，对 $a$ 来说它迈了很小一步，对 $b$ 来说它往反方向迈了很小一步，因此方差较小


#### 关于折扣

对于**公式（13）**，它们内部对于一个动作好坏的评价前面都有系数 $\gamma^t$ ，但实际应用中，往往会去除这个系数

去除此系数后在数学上已经不是原本的梯度了，见[Is the Policy Gradient a Gradient?](https://arxiv.org/abs/1906.07073)，但是能避免轨迹中长远步骤的梯度消失

#### 关于  $\ln$ 

直观上理解上述各种公式，它们都在做同一件事：评判某个状态下的某个选择

评判的实现各不相同，但是后面的 $\nabla_\theta \ln \pi_\theta (a_t|s_t)$ 一直不变，直观上理解，即使我们使用 $\nabla_\theta \pi_\theta (a_t|s_t)$ ，在梯度上升后也能实现增大此选择的效果，如果我们不关心数学上的严谨，为什么不使用后者呢？

对 $\nabla_\theta \ln \pi_\theta (a_t|s_t)$ ，其等价于 $\frac{\nabla_\theta \pi_\theta (a_t|s_t)}{\pi_\theta (a_t|s_t)}$ ，这意味着概率越大的选择其梯度越小，反之则相反，即相同评价下小概率动作的选择将被更多的鼓励
