# Sparse-Vit
FDU NNDL final project task2: Sparse ViT。使用python3.7与torch1.8编写。
本项目实现了两种Sparsity DeiT，一种是基础的根据L1范数选择一定比例的参数置零，另一种是根据参数较小但不一定作用小这一特点，定义了新的敏感度与不确定度，参考文章ICML2022 [PLATON](https://github.com/QingruZhang/PLATON)。
