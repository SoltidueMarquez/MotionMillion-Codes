import torch.nn as nn  # PyTorch神经网络模块
from models.encdec import Encoder, Decoder  # 编码器和解码器模块
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset  # 量化器模块
from models.LFQ import LFQ  # 学习有限量化器
from models.FSQ import FSQ  # 有限标量量化器
from models.causal_cnn import CausalEncoder, CausalDecoder  # 因果卷积编码器和解码器


class VQVAE_251(nn.Module):  # VQ-VAE模型类，支持251维动作数据
    """
    功能：实现Vector Quantized Variational AutoEncoder，用于动作数据的编码和重建
    输入参数：
        args: 配置参数对象
        nb_code: int - 码本大小，默认为1024
        code_dim: int - 编码维度，默认为512
        output_emb_width: int - 输出嵌入宽度，默认为512
        down_t: int - 时间下采样层数，默认为3
        stride_t: int - 时间步长，默认为2
        width: int - 网络宽度，默认为512
        depth: int - 网络深度，默认为3
        dilation_growth_rate: int - 膨胀增长率，默认为3
        activation: str - 激活函数，默认为'relu'
        norm: str - 归一化方法，默认为None
        kernel_size: int - 卷积核大小，默认为3
        use_patcher: bool - 是否使用补丁器，默认为False
        patch_size: int - 补丁大小，默认为1
        patch_method: str - 补丁方法，默认为"haar"
        use_attn: bool - 是否使用注意力机制，默认为False
    主要组件：
        - encoder: 编码器，将动作序列编码为潜在表示
        - decoder: 解码器，将潜在表示重建为动作序列
        - quantizer: 量化器，将连续表示量化为离散码本
    实现过程：动作序列 -> 编码器 -> 量化器 -> 解码器 -> 重建动作序列
    """
    def __init__(self,  # 初始化函数
                 args,  # 配置参数
                 nb_code=1024,  # 码本大小
                 code_dim=512,  # 编码维度
                 output_emb_width=512,  # 输出嵌入宽度
                 down_t=3,  # 时间下采样层数
                 stride_t=2,  # 时间步长
                 width=512,  # 网络宽度
                 depth=3,  # 网络深度
                 dilation_growth_rate=3,  # 膨胀增长率
                 activation='relu',  # 激活函数
                 norm=None,  # 归一化方法
                 kernel_size=3,  # 卷积核大小
                 use_patcher=False,  # 是否使用补丁器
                 patch_size=1,  # 补丁大小
                 patch_method="haar",  # 补丁方法
                 use_attn=False,  # 是否使用注意力机制
                 ):
        
        super().__init__()  # 调用父类初始化
        self.code_dim = code_dim  # 保存编码维度
        self.num_code = nb_code  # 保存码本大小
        self.quant = args.quantizer  # 保存量化器类型
        
        
        if args.causal:  # 如果使用因果卷积
            print('use causal conv !!!')  # 打印使用因果卷积的提示
            self.encoder = CausalEncoder(251 if args.dataname == 'kit' else 272, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, kernel_size=kernel_size, use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method, use_attn=use_attn)  # 创建因果编码器
            self.decoder = CausalDecoder(251 if args.dataname == 'kit' else 272, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, kernel_size=kernel_size, use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method, use_attn=use_attn)  # 创建因果解码器
        else:  # 否则使用标准编码器和解码器
            self.encoder = Encoder(251 if args.dataname == 'kit' else 272, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, kernel_size=kernel_size, use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method)  # 创建标准编码器
            self.decoder = Decoder(251 if args.dataname == 'kit' else 272, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, kernel_size=kernel_size, use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method)  # 创建标准解码器

        if args.quantizer == "ema_reset":  # 如果使用EMA重置量化器
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)  # 创建EMA重置量化器
        elif args.quantizer == "orig":  # 如果使用原始量化器
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)  # 创建原始量化器
        elif args.quantizer == "ema":  # 如果使用EMA量化器
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)  # 创建EMA量化器
        elif args.quantizer == "reset":  # 如果使用重置量化器
            self.quantizer = QuantizeReset(nb_code, code_dim, args)  # 创建重置量化器
        elif args.quantizer == "LFQ":  # 如果使用学习有限量化器
            self.quantizer = LFQ(codebook_size=args.nb_code, dim=code_dim)  # 创建LFQ量化器
        elif args.quantizer == 'BSQ':  # 如果使用球面量化器
            self.quantizer = LFQ(codebook_size=args.nb_code, dim=code_dim, spherical=True)  # 创建球面LFQ量化器
        elif args.quantizer == 'FSQ':  # 如果使用有限标量量化器
            if args.nb_code == 256:  # 如果码本大小为256
                levels = [8, 6, 5]  # 设置量化级别
            elif args.nb_code == 512:  # 如果码本大小为512
                levels = [8, 8, 8]  # 设置量化级别
            elif args.nb_code == 1024:  # 如果码本大小为1024
                levels = [8, 5, 5, 5]  # 设置量化级别
            elif args.nb_code == 2048:  # 如果码本大小为2048
                levels = [8, 8, 6, 5]  # 设置量化级别
            elif args.nb_code == 4096:  # 如果码本大小为4096
                levels = [7, 5, 5, 5, 5]  # 设置量化级别
            elif args.nb_code == 16384:  # 如果码本大小为16384
                levels = [8, 8, 8, 6, 5]  # 设置量化级别
            elif args.nb_code == 65536:  # 如果码本大小为65536
                levels = [8, 8, 8, 5, 5, 5]  # 设置量化级别
            else:  # 其他情况
                raise ValueError('Unsupported number of codebooks')  # 抛出不支持码本大小的错误
            self.quantizer = FSQ(levels=levels, dim=code_dim)  # 创建FSQ量化器

    def preprocess(self, x):  # 预处理函数
        """
        功能：将输入数据从时间序列格式转换为通道优先格式
        输入参数：
            x: torch.Tensor - 输入动作数据，形状为(batch_size, time, joints*3)
        输出：
            torch.Tensor - 预处理后的数据，形状为(batch_size, joints*3, time)
        实现过程：调整维度顺序并转换为浮点型
        """
        # (bs, T, Jx3) -> (bs, Jx3, T)  # 从(批次, 时间, 关节*3)转换为(批次, 关节*3, 时间)
        x = x.permute(0,2,1).float()  # 调整维度顺序并转换为浮点型
        return x  # 返回预处理后的数据


    def postprocess(self, x):  # 后处理函数
        """
        功能：将输出数据从通道优先格式转换回时间序列格式
        输入参数：
            x: torch.Tensor - 输出数据，形状为(batch_size, joints*3, time)
        输出：
            torch.Tensor - 后处理后的数据，形状为(batch_size, time, joints*3)
        实现过程：调整维度顺序
        """
        # (bs, Jx3, T) ->  (bs, T, Jx3)  # 从(批次, 关节*3, 时间)转换为(批次, 时间, 关节*3)
        x = x.permute(0,2,1)  # 调整维度顺序
        return x  # 返回后处理后的数据


    def encode(self, x):  # 编码函数
        """
        功能：将动作序列编码为离散码本索引
        输入参数：
            x: torch.Tensor - 输入动作数据，形状为(batch_size, time, joints*3)
        输出：
            torch.Tensor - 编码后的码本索引，形状为(batch_size, encoded_length)
        实现过程：预处理 -> 编码器 -> 量化器 -> 返回码本索引
        """
        
        if self.quant in ["LFQ", "BSQ", "FSQ"]:  # 如果使用现代量化器
            N, T, _ = x.shape  # 获取批次大小和时间长度
            x_in = self.preprocess(x)  # 预处理输入数据
            x_encoder = self.encoder(x_in)  # 通过编码器编码
            _, code_idx, _, _, _, _ = self.quantizer(x_encoder)  # 通过量化器获取码本索引
            code_idx = code_idx.view(N, -1)  # 重塑为(batch_size, encoded_length)
            return code_idx  # 返回码本索引
        else:  # 如果使用传统量化器
            N, T, _ = x.shape  # 获取批次大小和时间长度
            x_in = self.preprocess(x)  # 预处理输入数据
            x_encoder = self.encoder(x_in)  # 通过编码器编码
            x_encoder = self.postprocess(x_encoder)  # 后处理编码结果
            x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)  # 重塑为(批次*时间, 通道)
            code_idx = self.quantizer.quantize(x_encoder)  # 通过量化器量化
            code_idx = code_idx.view(N, -1)  # 重塑为(batch_size, encoded_length)
            return code_idx  # 返回码本索引


    def forward(self, x):  # 前向传播函数
        """
        功能：执行完整的VQ-VAE前向传播，包括编码、量化和解码
        输入参数：
            x: torch.Tensor - 输入动作数据，形状为(batch_size, time, joints*3)
        输出：
            x_out: torch.Tensor - 重建的动作数据，形状为(batch_size, time, joints*3)
            loss: torch.Tensor - 量化损失
            perplexity: torch.Tensor - 困惑度（码本使用情况）
            activate: torch.Tensor - 激活的码本数量
            indices: torch.Tensor - 量化索引
        实现过程：预处理 -> 编码 -> 量化 -> 解码 -> 后处理
        """
        
        x_in = self.preprocess(x)  # 预处理输入数据
        # Encode  # 编码阶段
        x_encoder = self.encoder(x_in)  # 256, 512, 16  # 通过编码器编码
         
        ## quantization  # 量化阶段
        if self.quant in ["LFQ", "BSQ"]:  # 如果使用LFQ或BSQ量化器
            x_quantized, _, loss, perplexity, activate, indices = self.quantizer(x_encoder)  # 通过量化器量化
        elif self.quant == "FSQ":  # 如果使用FSQ量化器
            x_quantized, _, loss, perplexity, activate, indices = self.quantizer(x_encoder)  # 通过量化器量化
        else:  # 如果使用传统量化器
            x_quantized, loss, perplexity, activate, indices  = self.quantizer(x_encoder)  # (256, 512, 16)  # 通过量化器量化
        
        ## decoder  # 解码阶段
        x_decoder = self.decoder(x_quantized)  # 通过解码器解码
        x_out = self.postprocess(x_decoder)  # 后处理输出数据

        return x_out, loss, perplexity, activate, indices  # 返回重建数据、损失、困惑度、激活数量和索引


    def forward_decoder(self, x):  # 解码器前向传播函数
        """
        功能：从码本索引直接解码为动作序列（用于推理阶段）
        输入参数：
            x: torch.Tensor - 码本索引，形状为(encoded_length,)
        输出：
            torch.Tensor - 重建的动作数据，形状为(1, time, joints*3)
        实现过程：反量化 -> 重塑维度 -> 解码器 -> 后处理
        """
        x_d = self.quantizer.dequantize(x)  # 从码本索引反量化为连续表示
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()  # 重塑为(1, code_dim, encoded_length)
        
        # decoder  # 解码器
        x_decoder = self.decoder(x_d)  # 通过解码器解码
        x_out = self.postprocess(x_decoder)  # 后处理输出数据
        return x_out  # 返回重建的动作数据



class HumanVQVAE(nn.Module):  # 人体VQ-VAE模型类
    """
    功能：人体动作数据的VQ-VAE模型包装器，基于VQVAE_251实现
    输入参数：
        args: 配置参数对象
        nb_code: int - 码本大小，默认为512
        code_dim: int - 编码维度，默认为512
        output_emb_width: int - 输出嵌入宽度，默认为512
        down_t: int - 时间下采样层数，默认为3
        stride_t: int - 时间步长，默认为2
        width: int - 网络宽度，默认为512
        depth: int - 网络深度，默认为3
        dilation_growth_rate: int - 膨胀增长率，默认为3
        activation: str - 激活函数，默认为'relu'
        norm: str - 归一化方法，默认为None
        kernel_size: int - 卷积核大小，默认为3
        use_patcher: bool - 是否使用补丁器，默认为False
        patch_size: int - 补丁大小，默认为1
        patch_method: str - 补丁方法，默认为"haar"
        use_attn: bool - 是否使用注意力机制，默认为False
    主要组件：
        - vqvae: VQVAE_251实例，执行实际的编码和解码操作
        - nb_joints: 关节数量，根据数据集确定
    实现过程：将人体动作数据传递给VQVAE_251进行处理
    """
    def __init__(self,  # 初始化函数
                 args,  # 配置参数
                 nb_code=512,  # 码本大小
                 code_dim=512,  # 编码维度
                 output_emb_width=512,  # 输出嵌入宽度
                 down_t=3,  # 时间下采样层数
                 stride_t=2,  # 时间步长
                 width=512,  # 网络宽度
                 depth=3,  # 网络深度
                 dilation_growth_rate=3,  # 膨胀增长率
                 activation='relu',  # 激活函数
                 norm=None,  # 归一化方法
                 kernel_size=3,  # 卷积核大小
                 use_patcher=False,  # 是否使用补丁器
                 patch_size=1,  # 补丁大小
                 patch_method="haar",  # 补丁方法
                 use_attn=False,  # 是否使用注意力机制
                 ):
        
        super().__init__()  # 调用父类初始化
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22  # 根据数据集确定关节数量
        self.vqvae = VQVAE_251(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, kernel_size=kernel_size, use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method, use_attn=use_attn)  # 创建VQVAE_251实例

    def encode(self, x):  # 编码函数
        """
        功能：将人体动作序列编码为离散码本索引
        输入参数：
            x: torch.Tensor - 输入人体动作数据，形状为(batch_size, time, joints*3)
        输出：
            torch.Tensor - 编码后的码本索引，形状为(batch_size, encoded_length)
        实现过程：调用VQVAE_251的编码方法
        """
        b, t, c = x.size()  # 获取批次大小、时间长度和通道数
        quants = self.vqvae.encode(x)  # (N, T)  # 调用VQVAE_251的编码方法
        return quants  # 返回码本索引

    def forward(self, x):  # 前向传播函数
        """
        功能：执行完整的人体VQ-VAE前向传播
        输入参数：
            x: torch.Tensor - 输入人体动作数据，形状为(batch_size, time, joints*3)
        输出：
            x_out: torch.Tensor - 重建的人体动作数据，形状为(batch_size, time, joints*3)
            loss: torch.Tensor - 量化损失
            perplexity: torch.Tensor - 困惑度（码本使用情况）
            activate: torch.Tensor - 激活的码本数量
            indices: torch.Tensor - 量化索引
        实现过程：调用VQVAE_251的前向传播方法
        """

        x_out, loss, perplexity, activate, indices = self.vqvae(x)  # 调用VQVAE_251的前向传播方法
        
        return x_out, loss, perplexity, activate, indices  # 返回重建数据、损失、困惑度、激活数量和索引

    def forward_decoder(self, x):  # 解码器前向传播函数
        """
        功能：从码本索引直接解码为人体动作序列（用于推理阶段）
        输入参数：
            x: torch.Tensor - 码本索引，形状为(encoded_length,)
        输出：
            torch.Tensor - 重建的人体动作数据，形状为(1, time, joints*3)
        实现过程：调用VQVAE_251的解码器前向传播方法
        """
        x_out = self.vqvae.forward_decoder(x)  # 调用VQVAE_251的解码器前向传播方法
        return x_out  # 返回重建的人体动作数据
        