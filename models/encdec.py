import torch.nn as nn
from models.resnet import Resnet1D
from models.modules import Patcher1D, UnPatcher1D
# class Encoder(nn.Module):
#     def __init__(self,
#                  input_emb_width = 3,
#                  output_emb_width = 512,
#                  down_t = 3,
#                  stride_t = 2,
#                  width = 512,
#                  depth = 3,
#                  dilation_growth_rate = 3,
#                  activation='relu',
#                  norm=None,
#                  block_width = [512,512,1024,1024]
#                  ):
#         super().__init__()
        
#         blocks = []
#         filter_t, pad_t = stride_t * 2, stride_t // 2
#         blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
        
#         input_dim = width
#         for i in range(down_t):
#             # block = nn.Sequential(
#             #     nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
#             #     Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
#             # )
#             # blocks.append(block)
#             block = nn.Sequential(
#                 nn.Conv1d(input_dim, block_width[i], filter_t, stride_t, pad_t),
#                 Resnet1D(block_width[i], depth, dilation_growth_rate, activation=activation, norm=norm),
#             )
#             blocks.append(block)
#             input_dim = block_width[i]
#         # blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
#         blocks.append(nn.Conv1d(block_width[-1], block_width[-1], 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         # x.shape (b, 263, len)
#         return self.model(x)

# class Decoder(nn.Module):
#     def __init__(self,
#                  input_emb_width = 3,
#                  output_emb_width = 512,
#                  down_t = 3,
#                  stride_t = 2,
#                  width = 512,
#                  depth = 3,
#                  dilation_growth_rate = 3, 
#                  activation='relu',
#                  norm=None,
#                  block_width = [1024,1024,512,512]
#                  ):
#         super().__init__()
#         blocks = []
        
#         filter_t, pad_t = stride_t * 2, stride_t // 2
#         # blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
#         blocks.append(nn.Conv1d(block_width[0], block_width[0], 3, 1, 1))
#         blocks.append(nn.ReLU())
#         input_dim = block_width[0]
#         for i in range(down_t):
#             # out_dim = width
#             # block = nn.Sequential(
#             #     Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
#             #     nn.Upsample(scale_factor=2, mode='nearest'),
#             #     nn.Conv1d(width, out_dim, 3, 1, 1)
#             # )
#             # blocks.append(block)
            
#             block = nn.Sequential(
#                 Resnet1D(input_dim, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 nn.Conv1d(input_dim, block_width[i], 3, 1, 1)
#             )
#             blocks.append(block)
#             input_dim = block_width[i]
            
#         blocks.append(nn.Conv1d(width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         return self.model(x)




class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 kernel_size=3,
                 use_patcher=False,
                 patch_size=1,
                 patch_method="haar",
                 ):
        super().__init__()
        
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if use_patcher:
            blocks.append(Patcher1D(patch_size, patch_method))
        blocks.append(nn.Conv1d(input_emb_width, width, kernel_size, 1, (kernel_size-1)//2))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, kernel_size=kernel_size),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, kernel_size, 1, (kernel_size-1)//2))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # x.shape (b, 263, len)
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None,
                 kernel_size=3,
                 use_patcher=False,
                 patch_size=1,
                 patch_method="haar",
                 ):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        blocks.append(nn.Conv1d(output_emb_width, width, kernel_size, 1, (kernel_size-1)//2))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, kernel_size=kernel_size),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
            
        blocks.append(nn.Conv1d(width, width, kernel_size, 1, (kernel_size-1)//2))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, kernel_size, 1, (kernel_size-1)//2))
        
        if use_patcher:
            blocks.append(UnPatcher1D(patch_size, patch_method))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    
