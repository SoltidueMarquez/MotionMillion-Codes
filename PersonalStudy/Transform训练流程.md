#### 10/14:

添加启动项并修改代码：

![image-20251014200950136](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251014200950136.png)

![image-20251014201037129](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251014201037129.png)

遇到了问题：

![image-20251014200654876](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251014200654876.png)

```python
Exception has occurred: FileNotFoundError
[Errno 2] No such file or directory: './dataset/MotionMillion\\all_data.pkl'
  File "D:\Desktop\动画项目\MotionMillion-Codes\dataset\dataset_TM_train_motionmillion.py", line 97, in __init__
    with open(os.path.join(self.data_root, "all_data.pkl"), "rb") as f:
  File "D:\Desktop\动画项目\MotionMillion-Codes\dataset\dataset_TM_train_motionmillion.py", line 214, in DATALoader
    train_loader = torch.utils.data.DataLoader(Text2MotionDataset_motionmillion(dataset_name, clip_model = clip_model, text_encode = text_encode, text_sum_way = text_sum_way, comp_device = comp_device, split = split, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, debug=debug, motion_type=motion_type, text_type=text_type, version=version),
  File "D:\Desktop\动画项目\MotionMillion-Codes\train_t2m_llama.py", line 324, in main
    train_loader = dataset_TM_train_motionmillion.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, args.train_split, clip_model, args.text_encode, args.text_sum_way, comp_device, motion_type=args.motion_type, text_type=args.text_type, version=args.version, unit_length=2**args.down_t, debug=args.debug, num_workers=args.num_workers)
  File "D:\Desktop\动画项目\MotionMillion-Codes\train_t2m_llama.py", line 435, in <module>
    main()
FileNotFoundError: [Errno 2] No such file or directory: './dataset/MotionMillion\\all_data.pkl'
```

![image-20251014201403412](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251014201403412.png)

得先跑这个。

同样的添加执行项修改代码

![image-20251014230446937](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251014230446937.png)

![image-20251014230522390](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251014230522390.png)

得到了对应文件。





#### 10/15：

![image-20251015180439294](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251015180439294.png)

```python
Exception has occurred: OutOfMemoryError
CUDA out of memory. Tried to allocate 90.00 MiB. GPU 0 has a total capacity of 15.92 GiB of which 0 bytes is free. Of the allocated memory 28.16 GiB is allocated by PyTorch, and 1.59 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
  File "D:\Desktop\动画项目\MotionMillion-Codes\models\lit_llama\model_hf.py", line 718, in forward
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # 将投影结果分割为Q、K、V
  File "D:\Desktop\动画项目\MotionMillion-Codes\models\lit_llama\model_hf.py", line 605, in forward
    x = x + self.attn(self.rms_1(x), y_mask)  # 自注意力：先归一化，再注意力，最后残差连接
  File "D:\Desktop\动画项目\MotionMillion-Codes\models\lit_llama\model_hf.py", line 244, in forward
    x = block(x, y_mask)  # 通过Transformer块处理
  File "D:\Desktop\动画项目\MotionMillion-Codes\train_t2m_llama.py", line 131, in train_one_iter
    cls_pred = trans_encoder(a_indices, feat_clip_text, y_mask)
  File "D:\Desktop\动画项目\MotionMillion-Codes\train_t2m_llama.py", line 356, in main
    cls_pred, target = train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder, args, comp_device)
  File "D:\Desktop\动画项目\MotionMillion-Codes\train_t2m_llama.py", line 435, in <module>
    main()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 90.00 MiB. GPU 0 has a total capacity of 15.92 GiB of which 0 bytes is free. Of the allocated memory 28.16 GiB is allocated by PyTorch, and 1.59 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

减小了训练参数，

![image-20251015180905112](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251015180905112.png)
```python
Exception has occurred: AssertionError
Cannot forward sequence of length 301, block size is only 150
  File "D:\Desktop\动画项目\MotionMillion-Codes\models\lit_llama\model_hf.py", line 231, in forward
    assert (  # 检查序列长度是否超过模型限制
  File "D:\Desktop\动画项目\MotionMillion-Codes\train_t2m_llama.py", line 131, in train_one_iter
    cls_pred = trans_encoder(a_indices, feat_clip_text, y_mask)
  File "D:\Desktop\动画项目\MotionMillion-Codes\train_t2m_llama.py", line 376, in main
    cls_pred, target = train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder, args, comp_device)
  File "D:\Desktop\动画项目\MotionMillion-Codes\train_t2m_llama.py", line 455, in <module>
    main()
AssertionError: Cannot forward sequence of length 301, block size is only 150
```

修改blocksize参数

还是超了内存

![image-20251015181403288](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251015181403288.png)





添加一些debug看看：

![image-20251016000823112](Transform%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.assets/image-20251016000823112.png)

内存一直爆，没什么头绪，可能是梯度那边计算的问题