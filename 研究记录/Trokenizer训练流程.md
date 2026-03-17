1. **熟悉这些数据格式（后面需要用这些数据训练我们的PFNN模型,到时候我会让你将数据处理成pfnn的格式）**
2. Train Tokenizer(学习他如何利用FSQ对动作编码的，你需要理解并给我讲清楚FSQ的好处，为啥不用VQ-VAE，是你自己的想法，因为他们可能只是为了凑个新的点作为自己的创新点)

### 数据：

#### 原始数据：

![image-20251009084052836](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251009084052836.png)

##### 1. 动作数据格式 (motion_data)

文件格式：.npy (NumPy二进制格式)

数据维度：(时间步长, 272)

数据类型：float32

272维向量组成：

- 根关节位置：3维 (x, y, z)

- 关节旋转：22个关节 × 6维旋转表示 = 132维

- 其他特征：**137维**（可能包括速度、加速度等）（需要看一下）

##### 2. 文本数据格式 (texts)

文件格式：.txt (纯文本)

内容：多行文本描述，每行一个动作描述

##### 3. 数据集划分文件 (split)

文件格式：.txt (纯文本)

内容：每行一个动作ID，对应motion_data和texts中的文件名

这里用的是tokenizer_96 版本

- train.txt: ~（70%）（训练集）作用：用于模型训练的主要数据；特点：数量最多，包含大部分动作样本；用途：训练VQ-VAE的编码器和解码器

- val.txt: ~（10%）（验证集）作用：用于训练过程中的模型验证；特点：数量较少，用于监控训练进度；用途：计算MPJPE等评估指标、防止过拟合、调整超参数

- test.txt: ~（20%）（测试集）作用：用于最终模型性能评估；特点：数量适中，保持数据独立性；用途：最终性能评估、与训练集和验证集完全独立、报告最终结果

##### 4. 统计信息文件 (mean_std)

文件格式：.npy (NumPy二进制格式)

内容：

- mean.npy：272维的均值向量

- std.npy：272维的标准差向量

用途：用于数据标准化 (data - mean) / std

#### 数据获取

数据加载器——>初始化Init——>获取数据GetItem

##### 问题1：数据集格式(t2m, kit, motionmillion)配置分别有什么区别？在两个脚本中是什么样的

在 dataset_VQ.py, dataset_TM_eval.py 等文件中可以看到

![image-20251010191605473](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251010191605473.png)

![image-20251010191635850](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251010191635850.png)

在两个脚本里的处理感觉差不多。

![image-20251010191809521](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251010191809521.png)

###### 1. t2m (HumanML3D) 数据集

- 关节数量: 22个关节

- 姿态维度: 263维

- 最大运动长度: 196帧

- 帧率: 20 FPS

- 数据根目录: ./dataset/HumanML3D

###### 2. kit (KIT-ML) 数据集

- 关节数量: 21个关节（比HumanML3D少1个）

- 姿态维度: 251维

- 最大运动长度: 196帧

- 帧率: 12.5 FPS

- 数据根目录: ./dataset/KIT-ML

###### 3. motionmillion 数据集

- 关节数量: 22个关节

- 姿态维度: 272维（最丰富）

- 最大运动长度: 300-600帧（支持更长序列）（训练的时候）

- 帧率: 20-60 FPS（可变）

- 数据根目录: ./dataset/MotionMillion



##### 问题2：作为训练用的数据和作为评估用的数据在处理上有什么区别？为什么这么做？

1.train的数据只存储ID，不加载实际数据，而evl的数据是要加载的

![image-20251010195406289](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251010195406289.png)

2.训练数据是固定窗口随机采样能最大化数据利用率，评估数据需要**相对完整**的运动来模拟真实生成场景；

返回值也不同，训练只返回motion数据，评估返回数据、真实长度和名称。训练时，优化器只需要运动数据本身。评估时，因为评估序列被填充到了统一长度，在计算损失或指标时，必须知道原始有效长度，以便只计算有效部分，忽略填充部分。`name`则用于保存示例、调试和记录哪些样本表现好/差。

![image-20251010195630539](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251010195630539.png)

![image-20251009084436985](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251009084436985.png)

3.动态长度过滤 (`reset_max_len`)是评估数据集独有的功能，在评估生成模型时，可以先在较短的序列上测试其性能（因为生成长序列更难）。通过调整 `max_length`，可以逐渐增加评估难度，观察模型性能随序列长度变化的情况，从而更全面地评估模型的能力。训练码本时不需要这个，因为它处理的是固定窗口。

4.词向量化器WordVectorizer被评估数据集MotionMillionFSQDATALoader所使用



#### 调用链：

启动命令
    ↓
train_tokenizer.py (main函数)
    ↓
option_vq.get_args_parser() → 解析参数
    ↓
WordVectorizer('./glove', 'our_vab') → 词向量化器
（文本向量化工具，专门用于将文本描述转换为数值向量）
    ↓
dataset_VQ.DATALoader() → 训练数据加载
    ↓
dataset_TM_eval.MotionMillionFSQDATALoader() → 验证数据加载(使用词向量化器)
    ↓
vqvae.HumanVQVAE() → 模型初始化
    ↓
optim.AdamW() + MultiStepLR() → 优化器设置
    ↓
Accelerator().prepare() → 分布式训练准备
    ↓
losses.ReConsLoss() → 损失函数
    ↓
Warm-up训练循环 (40次迭代)
    ↓
主训练循环 (200次迭代)
    ↓
eval_trans.evaluation_vqvae_motionmillion() → 定期评估
    ↓
torch.save() → 模型保存



#### 流程描述：

1.通过launch.json配置启动

2.导入模块后进行参数解析（36行）

![image-20251009081527636](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251009081527636.png)

3.词向量化器

![image-20251009081545505](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251009081545505.png)

3.训练数据加载

![image-20251009081646076](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251009081646076.png)

4.验证数据加载

![image-20251009081501868](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251009081501868.png)

5.VQVAE模型初始化

![image-20251009081752180](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251009081752180.png)







### 10/12

#### 1.需要了解motion的具体shape 272*？

维度上，在loss.py中有对应的代码：

![image-20251012004157033](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251012004157033.png)

还有其他9维的特征没有在注释中明确说明，很奇怪？

添加print输出：

![image-20251011105728313](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251011105728313.png)

```python
------ warm-up -------
nb_iter:  1
训练数据 motion.shape:  (294, 272)训练数据 motion.shape: 
训练数据 motion.shape: 训练数据 motion.shape:   (298, 272)
训练数据 motion.shape:  训练数据 motion.shape: (178, 272)训练数据 motion.shape:  (322, 272)

(126, 272)训练数据 motion.shape: (221, 272)
 (494, 272)
训练数据 motion.shape:  (444, 272)

(297, 272)
训练数据 motion.shape:  (160, 272)
训练数据 motion.shape: 训练数据 motion.shape: 训练数据 motion.shape:  训练数据 motion.shape: 训练数据 motion.shape:  (600, 272)
(232, 272)
 训练数据 motion.shape:  (286, 272)
(98, 272)
训练数据 motion.shape: (316, 272)
 训练数据 motion.shape:   (356, 272)训练数据 motion.shape: (138, 272)
训练数据 motion.shape:   (194, 272)
(600, 272)
(96, 272)

训练数据 motion.shape:  (600, 272)
训练数据 motion.shape:  (432, 272)
训练数据 motion.shape:  (497, 272)
训练数据 motion.shape:  (252, 272)
nb_iter:  2
nb_iter:  3
nb_iter:  4
nb_iter:  5
nb_iter:  6
训练数据 motion.shape:  (205, 272)
训练数据 motion.shape:  (600, 272)
训练数据 motion.shape:  (129, 272)
训练数据 motion.shape:  (280, 272)
nb_iter:  7
训练数据 motion.shape:  (185, 272)
训练数据 motion.shape:  (223, 272)
训练数据 motion.shape:  (500, 272)
训练数据 motion.shape:  (500, 272)
nb_iter:  8
训练数据 motion.shape:  (110, 272)
训练数据 motion.shape:  (103, 272)
训练数据 motion.shape:  (394, 272)
训练数据 motion.shape:  (100, 272)
nb_iter:  9
训练数据 motion.shape:  (150, 272)
训练数据 motion.shape:  (599, 272)
训练数据 motion.shape:  (111, 272)
训练数据 motion.shape:  (299, 272)
nb_iter:  10
训练数据 motion.shape:  (180, 272)
训练数据 motion.shape:  (143, 272)
训练数据 motion.shape:  (234, 272)
训练数据 motion.shape:  (189, 272)
nb_iter:  11
训练数据 motion.shape:  (96, 272)
训练数据 motion.shape:  (115, 272)
训练数据 motion.shape:  (500, 272)
训练数据 motion.shape:  (275, 272)
nb_iter:  12
训练数据 motion.shape:  (171, 272)
训练数据 motion.shape:  (429, 272)
训练数据 motion.shape:  (319, 272)
训练数据 motion.shape:  (177, 272)
nb_iter:  13
训练数据 motion.shape:  (600, 272)
训练数据 motion.shape:  (109, 272)
训练数据 motion.shape:  (102, 272)
训练数据 motion.shape:  (257, 272)
nb_iter:  14
训练数据 motion.shape:  (97, 272)
训练数据 motion.shape:  (298, 272)
训练数据 motion.shape:  (305, 272)
训练数据 motion.shape:  (99, 272)
nb_iter:  15
训练数据 motion.shape:  (427, 272)
训练数据 motion.shape:  (152, 272)
训练数据 motion.shape:  (480, 272)
训练数据 motion.shape:  (287, 272)
nb_iter:  16
训练数据 motion.shape:  (98, 272)
训练数据 motion.shape:  (361, 272)
训练数据 motion.shape:  (129, 272)
训练数据 motion.shape:  (600, 272)
nb_iter:  17
训练数据 motion.shape:  (127, 272)
训练数据 motion.shape:  (145, 272)
训练数据 motion.shape:  (123, 272)
训练数据 motion.shape:  (167, 272)
nb_iter:  18
训练数据 motion.shape:  (159, 272)
训练数据 motion.shape:  (341, 272)
nb_iter:  19
训练数据 motion.shape:  (219, 272)
训练数据 motion.shape:  (193, 272)
nb_iter:  20
训练数据 motion.shape:  (119, 272)
训练数据 motion.shape:  (500, 272)
nb_iter:  21
训练数据 motion.shape:  (600, 272)
训练数据 motion.shape:  (500, 272)
nb_iter:  22
训练数据 motion.shape:  (260, 272)
训练数据 motion.shape:  (451, 272)
nb_iter:  23
训练数据 motion.shape:  (141, 272)
训练数据 motion.shape:  (351, 272)
nb_iter:  24
nb_iter:  25
nb_iter:  26
nb_iter:  27
nb_iter:  28
nb_iter:  29
nb_iter:  30
nb_iter:  31
nb_iter:  32
nb_iter:  33
nb_iter:  34
nb_iter:  35
nb_iter:  36
nb_iter:  37
nb_iter:  38
nb_iter:  39
准备开始训练
开始评估
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
评估数据处理后 motion.shape:  (600, 272)
```

- 最小长度：120帧

- 最大长度：600帧

- 典型长度：150-300帧

此外，batch_size上（在一次前向传播和反向传播中同时处理的样本数量），训练数据通过命令配置，这里是2，评估数据写死了是32。



#### 2.warm_up部分做了什么，有什么用？

- Warm-up只是预热阶段，不计入正式的训练迭代

在训练初期使用较小的学习率，然后逐渐增加到目标学习率。避免训练开始时学习率过大导致的梯度爆炸和不稳定

![image-20251012211653190](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251012211653190.png)

让模型逐渐适应多种损失函数的组合，避免初始阶段某个损失项过大导致训练偏向

![image-20251012211830698](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251012211830698.png)

此外

![image-20251012211904774](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251012211904774.png)





#### 3.训练的时候是怎么解决motion帧数序列长短不一的问题

![image-20251011110133387](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251011110133387.png)

训练部分采取了定长的64帧，从长序列中随机选择64帧的连续片段，（最小数据有120帧）保证了所有训练样本都是64×272的固定尺寸

![image-20251012213952585](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251012213952585.png)

评估数据采用了零填充，并返回了原始长度信息

![image-20251012005348481](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251012005348481.png)





#### 3.tokenizer的用处（Motion tokenizer）

![image-20251012215141629](Trokenizer%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251012215141629.png)

VQ-VAE tokenizer训练是纯运动数据的自监督学习：

- 输入：运动数据 (batch_size, 64, 272)

- 输出：重构的运动数据 (batch_size, 64, 272)

- 目标：学习运动数据的离散表示（codebook）

- 损失：重构损失 + commit损失 + 速度损失



#### 4.Quantizer (models/quantize_cnn.py) - 量化器的作用

模型组件：

- Encoder (models/encdec.py) - 编码器：将连续运动数据压缩为低维潜在表示
- Decoder (models/encdec.py) - 解码器：将量化后的特征重构为连续运动数据
- Quantizer (FSQ (models/FSQ.py)) - 量化器：将连续潜在特征转换为离散码本索引