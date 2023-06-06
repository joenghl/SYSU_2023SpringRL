# 实践作业二 (期末)：Multi-Agent Reinforcement Learning (MARL)

本次作业部分参考了以下开源仓库：
- OpenAI Baselines: https://github.com/openai/baselines
- OpenAI Multi-Agent Particle Environments: https://github.com/openai/multiagent-particle-envs

## 概述

### 环境任务

本次作业在多智能体粒子环境中的 `simple_spread` 协作任务中进行。粒子环境是由 OpenAI 开发的专用于多智能体强化学习的仿真环境，包括一系列不同场景的子任务，如多智能体竞争、合作、通信等不同的任务场景。在 `simple_spread` 任务场景中，有 3 个 Landmark (黑色小圆) 和 3 个 Agent (紫色大圆)，如下图所示：

<div align=center><img width = '300' height ='300' src ="https://github.com/joenghl/SYSU_2023SpringRL/blob/master/docs/images/mpe_demo.png?raw=true"/></div>

在此场景中，3 个 Agent 需要学会相互协作去分别导航至 3 个 Landmark，并在此过程中避免相互碰撞。此任务为完全合作式任务，Agent 间仅有合作关系，不存在竞争关系。

### 观测空间

Agent 观测维度: Box(18,)

Agent 观测信息: [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]

其中,
- `self_vel`: 自己的速度信息 (2,) (x,y轴，二维，下同)
- `self_pos`: 自己的位置信息 (2,)
- `landmark_rel_positions`: 地标和当前 Agent 的相对位置 (2,)*3
- `other_agent_rel_positions`: 其他 Agent 和当前 Agent 的相对位置 (2,)*2
- `communication`: 其他 Agent 和当前 Agent 的通信信息 (2,)*2 (此任务中不需要通信信息，所以通信信息置为0)

### 动作空间

Agent 动作维度: Discrete(5,)

Agent 动作信息: [no_action, move_left, move_right, move_down, move_up]

在 `env.step(actions)` 中，`actions` 需要转换为 one-hot 表示，具体形式可在 `agents/random/submission` 参考动作格式示例。


### 奖励函数

奖励值为计算每个 Landmark 到距离其最近的 Agent 的距离 * -1.0 并求和，此外，每次 Agent 间发生碰撞会得到 -1.0 的惩罚。具体代码可在粒子环境源码仓库 `multiagent-particle-envs/multiagent/scenarios/simple_spread` 中 `reward` 函数查看。


## 环境配置

整个项目使用 Python 语言，在配置环境前，推荐使用 `conda` 工具管理 Python 编程环境。

#### 1. 安装 `conda` 工具 (推荐, 可选)

推荐从 [Miniconda Official](https://docs.conda.io/en/latest/miniconda.html) 下载 `Miniconda` ，安装时勾选自动添加环境变量选项，若没有勾选则需要手动添加，其他选项默认即可。

#### 2. 新建并激活用于此项目的虚拟环境 

**Option a. 使用 `conda` :**

```shell
conda create -n mpe python=3.7
conda activate mpe
```

**Option b. 使用 Python 工具 :**

```shell
python -m venv mpe
source mpe/bin/activate
```

#### 3. 配置项目仓库

**Option a. 更新之前的仓库 :**

```shell
cd SYSU_2023SpringRL
git pull origin master
cd Assignment2
```

**Option b. 重新拉取仓库 :**

```shell
git clone https://github.com/joenghl/SYSU_2023SpringRL.git
cd SYSU_2023SpringRL/Assignment2
```

#### 4. 配置仿真环境 Multi-Agent Particle Environments (MPE)

```shell
git clone https://github.com/shariqiqbal2810/multiagent-particle-envs.git
cd multiagent-particle-envs
pip install -e .
```

#### 5. 安装依赖库

```shell
python -m pip install seaborn gym==0.9.4 pyglet==1.2.4 torch==1.13.1
```

## 代码示例

代码结构如下所示：

```
Assignment2
│   README.md
│   run_test.py  # 本地测试文件
│
└───agents  # 提交样例
│   │ 
│   └─random  # 样例一: 随机采样模型
│   │   submission.py
│   │  
│   └─random_network  # 样例二: 随机网络模型
│       │   agent1.pth
│       │   agent2.pth
│       │   agent3.pth
│       │   submission.py
│
└───utils
    |   make_env.py  # 环境封装  
```

作业目标即为使用多智能体强化学习算法训练一个 `agents` 提交样例，将训练好的网络模型**前向计算**部分打包为类似 `agents/random_network` 的文件夹 (需要包含 `submission.py` 文件)，在本地使用 `run_test.py` 测试文件自行测试通过 (在 `run_test.py` 中将模型替换为自己的模型)，且保证每步耗时不超过 10 ms (`run_test.py` 中有判别基准)。`run_test.py` 文件中已将需要替换的代码前加 `TODO` 注释和说明。

注意：提交模型样例文件夹名字命名为 `学号_姓名拼音`，如 `22000000_zhangsan`。需要保证文件夹内有 `submission.py` 文件，内含有 `Agents` 类，`Agents` 类中必须有 `act` 方法。这样才可以在 `run_test.py` 中测试通过。

## 基准

| Level     | Reward | Score|
| ----------- | ----------- | --- |
| Weak Baseline      | -6.5      | 70+  |
| Medium Baseline   | -5.5        | 80+ |
| Strong Baseline   | -4.8        | 95+ |

具体分数和最终评估性能、代码规范、报告等相关。

补充：这里的 `Reward` 为 `run_test.py` 文件中评估 100 局的平均回报，每步的回报为 3 个 Agents 回报的加和。

## 提交要求


需要提交两个压缩包，分别为：

### 1. 模型压缩包
将自己训练好的提交样例放入 `agents` 文件夹中测试通过后，将文件夹打包为 `.zip`，如将测试样例打包 `random_network.zip`。**压缩包大小禁止超过50MB。**

命名规则：学号\_姓名拼音。若提交后需更新代码，请加后缀 \_v1, \_v2，以此类推。

如: `22000000_zhangsan.zip`, `22000000_zhangsan_v1.zip`

模型压缩包需上传至 FTP 服务器根目录下的 `研究生强化学习作业2_code` 文件夹，截止日期: 2023.7.16 23:59。

### 2. 实验报告压缩包
将完整训练的代码、实验报告，以及自己的模型在 `run_test.py` 中测试的渲染录制一个 5~10 s 的视频，格式为 `.mp4`，打包为一个压缩包提交。渲染时可将 `run_test.py` 中的 `render()` 注释取消。

命名规则：学号\_姓名拼音。若提交后需更新代码，请加后缀 \_v1, \_v2，以此类推。

如: `22000000_zhangsan.zip`, `22000000_zhangsan_v1.zip`

实验报告压缩包需上传至 FTP 服务器根目录下的 `研究生强化学习作业2_report` 文件夹，截止日期: 2023.7.16 23:59。

### FTP 服务器

IP: 222.200.177.152

Port: 1021

User: ftpstu

Password: 123456

Tips: Windows 下在文件夹路径填入 `ftp://222.200.177.152:1021/` 后输入用户名和密码即可连接至 FTP 服务器（校园网内网）。

## 问题反馈

有任何疑问可直接在课程群中提问，或者在此提 Issues，或者发邮件至助教邮箱: yanghlin7@mail2.sysu.edu.cn，每天固定时间检查并回复邮件，请勿直接添加助教个人微信。

