# 图像去雾入门与应用 · ZRAM

> 一本面向计算机视觉入门者的中文去雾教程：从大气散射物理模型讲起，系统梳理 **CNN / Transformer / Mamba / SNN** 四条骨干路线，覆盖 **有监督 / 半监督 / 无监督** 三类学习范式，并配套可运行的 PyTorch Notebook 与工程落地指引。

<p align="center">
  <img alt="pages" src="https://img.shields.io/badge/PDF-56%20页-16263f">
  <img alt="notebooks" src="https://img.shields.io/badge/Notebooks-7%20个-1fb6a6">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-ef7b54">
  <img alt="author" src="https://img.shields.io/badge/作者-ZRAM-e9c46a">
  <img alt="lang" src="https://img.shields.io/badge/语言-中文-16263f">
</p>

---

## 一、这本书讲什么

图像去雾是典型的**低层视觉逆问题**：给定一张退化的有雾图像，恢复清晰无雾的场景。本仓库把"入门 → 理解方法动机 → 动手搭网络 → 落地部署"串成一条完整思想路线，不堆砌公式，而是回答三个问题：

1. **问题从哪来** —— 雾是怎么形成的？经典物理模型与先验方法做了什么、局限在哪？
2. **网络怎么搭** —— 同一去雾任务，CNN、Transformer、Mamba、SNN 各自如何建模依赖、效率与精度如何权衡？
3. **数据怎么解决** —— 真实场景缺成对数据时，半监督、无监督、域适应、零样本分别如何应对？

### 思想路线（推荐阅读顺序）

```
物理模型与经典先验  →  数据集与评价指标  →  训练范式与损失
        │
        ├── CNN（基线 + FFA 注意力）
        ├── Transformer（自注意力 + 窗口注意力，DehazeFormer）
        ├── Mamba（状态空间模型，线性复杂度长程建模）
        └── SNN（脉冲驱动，面向低能耗场景）
        │
半监督 / 无监督  →  前沿专题（频域 · 生成式 · 多模态）  →  夜间专题
        │
工程落地与边缘部署  →  实战路线与开放问题
```

## 二、成书内容（13 章 + 2 附录）

| 章节 | 标题 | 核心内容 |
|---|---|---|
| 第 1 章 | 导论：认识图像去雾 | 问题定义、病态逆问题、方法谱系与学习地图 |
| 第 2 章 | 大气散射模型与经典先验方法 | 成像模型推导、透射率/大气光、暗通道先验（DCP）复现与局限 |
| 第 3 章 | 数据集与评价指标 | RESIDE、I/O/Dense/NH-HAZE、Haze1K、遥感数据集；PSNR / SSIM / 无参考指标 |
| 第 4 章 | 深度学习去雾的总体范式 | 参数估计 vs 端到端、像素/感知/结构/对抗损失、数据流水线与训练骨架 |
| 第 5 章 | CNN 去雾网络：从基线到注意力 | U-Net 基线、FFA-Net 通道与像素注意力、多组特征融合 |
| 第 6 章 | Transformer 去雾：全局依赖建模 | 自注意力、窗口注意力、DehazeFormer 的归一化/激活/移位改造 |
| 第 7 章 | Mamba 状态空间模型去雾 | 线性递推、双向扫描、局部-全局混合、频域分解等视觉落地设计 |
| 第 8 章 | 脉冲神经网络（SNN）去雾 | LIF 神经元、代理梯度、脉冲卷积块与轻量骨架、能耗 vs 精度 |
| 第 9 章 | 半监督与无监督去雾 | 伪标签/师生互学习、域迁移、图像分解、零样本与物理约束 |
| 第 10 章 | 前沿专题：频域、生成式与多模态 | 频域分解、扩散模型、GAN、CLIP/多模态语义引导 |
| 第 11 章 | 专项场景：夜间去雾专题 | 低照度+光晕+多色光+噪声复合退化、夜间数据集、联合增强-去雾网络 |
| 第 12 章 | 工程落地与边缘部署 | 推理流程、大图切片（Tiling）、轻量化、蒸馏、量化部署与问题排查 |
| 第 13 章 | 入门实战路线与开放问题 | 从零复现清单、论文阅读路径、值得关注的开放问题 |
| 附录 A | 开源仓库与资源清单 | 数据集、代码仓库、论文入口汇总 |
| 附录 B | 完整 PyTorch 训练 / 测试模板 | 可直接改造的数据加载、训练循环、评估脚本 |

## 三、配套可运行 Notebook

`book/notebooks/` 下 7 个自包含 Notebook，每个都含原理讲解、模型定义与随机张量前向验证（打印输出形状与参数量），CPU 即可跑通形状检查；接入真实数据后可按附录 B 模板扩展为完整训练。

| Notebook | 对应章节 | 内容 |
|---|---|---|
| `00_metrics_demo.ipynb` | 第 3 章 | PSNR / SSIM 指标实现与验证 |
| `01_cnn_unet_baseline.ipynb` | 第 5 章 | U-Net 去雾基线 + 参数量统计 |
| `02_ffa_attention.ipynb` | 第 5 章 | FFA-Net 通道/像素注意力与多组融合 |
| `03_transformer.ipynb` | 第 6 章 | RescaleNorm + 窗口注意力复原块 |
| `04_mamba.ipynb` | 第 7 章 | 双向扫描状态空间块 |
| `05_snn.ipynb` | 第 8 章 | LIF 神经元 + 代理梯度 + 脉冲骨架 |
| `06_night_dehaze.ipynb` | 第 11 章 | 夜间联合增强-去雾网络与专用损失 |

> 已打包为 [`去雾配套Notebook_ZRAM.zip`](去雾配套Notebook_ZRAM.zip)，解压后在 PyTorch 环境下逐格运行即可。

## 四、仓库结构

```
.
├── 图像去雾入门与应用_ZRAM.pdf      # 成书 PDF（56 页，作者 ZRAM）
├── 去雾配套Notebook_ZRAM.zip       # 7 个配套 Notebook 打包
├── README.md
│
├── book/
│   ├── build.py                    # HTML 组装 + 代码高亮 + WeasyPrint 渲染 PDF
│   ├── style.css                   # 全书排版样式与配色
│   ├── figures.py                   # 生成原理配图（大气散射、谱系、路线图等）
│   ├── cover.py                    # 生成封面背景图
│   ├── make_notebooks.py           # 生成上述 7 个 Jupyter Notebook
│   ├── chapters/                   # 各章 HTML 源文件（front / ch01–ch13 / 附录 / 参考文献）
│   ├── notebooks/                  # 配套可运行 Notebook 源文件
│   └── assets/                     # 配图与封面背景 PNG
│
└── sources/                         # 摘自官方开源仓库的参考副本（版权归原作者，仅供代码提炼对照）
    ├── DehazeFormer/               # IDKiro/DehazeFormer（IEEE TIP 2023，MIT）
    ├── FFA-Net/                    # zhilin007/FFA-Net（AAAI 2020）
    └── DehazeSNN/                  # HaoranLiu507/DehazeSNN（arXiv:2512.23950，MIT）
```

## 五、环境与安装

### 1. 基础环境

- Python 3.10+
- 建议使用虚拟环境（conda / venv）

### 2. 运行 Notebook（核心依赖）

```bash
pip install torch torchvision numpy matplotlib jupyter nbformat
```

### 3. 从源码重建 PDF（可选）

```bash
pip install weasyprint matplotlib numpy nbformat
# WeasyPrint 系统依赖（Debian/Ubuntu）：
# sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0
```

> 中文排版需要系统安装 Noto Sans/Serif CJK 等中文字体。

### 4. 训练 / 评测扩展（可选）

```bash
pip install pillow scipy opencv-python scikit-image thop tqdm
```

## 六、快速开始

### 直接阅读

打开 [`图像去雾入门与应用_ZRAM.pdf`](图像去雾入门与应用_ZRAM.pdf) 即可，目录页码、章节交叉引用与参考文献均已排好。

### 运行 Notebook

```bash
cd book/notebooks
jupyter notebook
# 从 00_metrics_demo.ipynb 开始按编号顺序运行
```

### 从源码重建全书

在 `book/` 目录下按顺序执行：

```bash
cd book
python figures.py        # 1. 生成 assets/ 下的原理配图
python cover.py          # 2. 生成封面背景 cover_bg.png
python make_notebooks.py # 3. 生成 notebooks/*.ipynb
python build.py          # 4. 组装 HTML 并渲染 PDF（输出到上级目录）
```

## 七、数据集与指标指引

书中涉及的主要基准数据集：

- **合成成对**：RESIDE-ITS / OTS / SOTS、Dense-HAZE、I-HAZE、O-HAZE、NH-HAZE、Haze1K
- **夜间专项**：Reside-β Night、GTA5 夜间雾、RealNightHaze、NT-HAZE
- **遥感场景**：RICE、SateHaze1K

常用评价指标：

- **全参考**：PSNR、SSIM、CIEDE2000
- **无参考 / 自然度**：NIQE、MUSIQ 等
- **任务导向**：下游检测/分割等上游任务表现

数据集下载与组织方式见第 3 章与附录 A；成对目录建议按 `hazy/` 与 `gt/` 结构存放。

## 八、参考仓库与论文

本书代码骨架在理解以下官方开源实现的基础上提炼改写，**版权归原作者所有**：

| 仓库 | 方法 / 出处 | 协议 | 链接 |
|---|---|---|---|
| DehazeFormer | DehazeFormer, IEEE TIP 2023, 32:1927–1941 | MIT | [github.com/IDKiro/DehazeFormer](https://github.com/IDKiro/DehazeFormer) |
| FFA-Net | FFA-Net, AAAI 2020 | 见原仓库条款 | [github.com/zhilin007/FFA-Net](https://github.com/zhilin007/FFA-Net) |
| DehazeSNN | DehazeSNN, arXiv:2512.23950 | MIT | [github.com/HaoranLiu507/DehazeSNN](https://github.com/HaoranLiu507/DehazeSNN) |

书中参考文献共 25 条，涵盖经典先验、各骨干代表方法、半/无监督范式、夜间去雾与相关综述，完整条目见 PDF 末尾「参考文献」一章。

## 九、说明与免责

- 本书定位为**教学入门**：书中代码为可独立理解、可前向验证的骨架实现，重在讲清网络结构与设计动机；复现论文级指标需配合完整数据集、训练策略与调参。
- `sources/` 目录为方便对照而保存的官方开源仓库副本，如用于研究或商业用途，请遵守各原仓库对应的开源协议并引用原论文。
- 若发现内容错误、链接失效或代码问题，欢迎提 Issue 交流。

## 十、作者

**ZRAM**

- 主题：图像去雾 / 低层视觉复原 / 高效网络设计
- 欢迎围绕入门路线、网络结构选型与工程部署交流

---

<p align="center"><i>图像去雾入门与应用 · 从物理模型到高效网络，再到真实场景落地。</i></p>
