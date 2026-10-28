# CS5495 Assignment 2：CNN 错误预测的可解释性研究方案

## 推荐题目

**预训练 CNN 分类错误时，模型关注的是目标物体还是背景？不同 XAI 方法给出的解释是否一致？**

**English title:** *When Pre-trained CNNs Make Mistakes: Object or Background? A Comparison of Explainability Methods*

这个方案结合了作业中关于 CNN 错误案例和模型关注区域的两个方向。分析应根据实验结果判断模型可能使用了哪些线索，不应预先假设模型一定依赖背景。

## 推荐的方法组合

| 类型 | 方法 | 项目中的作用 |
| --- | --- | --- |
| 课内 XAI | Grad-CAM | 查看 CNN 中哪些区域对目标类别有较大贡献 |
| 课内 XAI | Integrated Gradients（IG） | 从输入像素角度分析哪些位置影响预测 |
| 课外 XAI 候选 | Score-CAM | 根据激活图和模型分数生成区域解释，不依靠梯度计算通道权重 |
| 课外 XAI 候选 | XRAI | 结合 Integrated Gradients 和图像分割，分析重要的图像区域 |
| 验证方法 | Insertion / Deletion | 检查解释图标记为重要的区域是否会影响模型输出 |

**课外方法的认定范围：**在本次提供的 Lecture 5a–8b 八份课件中，没有发现对 Score-CAM 和 XRAI 的具体介绍。由于尚未核对全部 tutorial、其他讲义和课堂口头内容，二者暂定为课外候选；是否满足评分要求，应结合课程完整材料及老师的认定确认。XRAI 建立在课堂讲过的 IG 之上，需要在报告中说明两者的关系。

## 课件中已经出现的方法

| 课件 | 与本项目有关的内容 |
| --- | --- |
| `Lecture5a.pdf` | CNN、卷积层与特征提取基础 |
| `Lecture5b.pdf` | Feature Visualization、Network Dissection |
| `Lecture6a.pdf` | Vanilla Gradient、Input × Gradient、Integrated Gradients、SmoothGrad、DeepLIFT、LRP、Grad-CAM、Guided Grad-CAM、LIME 和 KernelSHAP；Insertion / Deletion 评价 |
| `Lecture6b.pdf` | 对抗样本、反事实解释、影响样本、FGSM、PGD、Influence Functions、TracIn |
| `Lecture7a.pdf` | Transformer 和注意力解释；注意力或高亮区域并不自动构成对模型决策的充分解释 |
| `Lecture7b.pdf` | 在语言模型示例中使用 GradientSHAP 和 Occlusion 等方法 |
| `Lecture8a-ODAM.pdf` | 目标检测解释方法 ODAM；比较图和表格中出现 Grad-CAM++ 与 D-RISE |
| `Lecture8b-DAAM.pdf` | DAAM、Stable Diffusion 中的文本与生成图像关系 |

**不要把 Grad-CAM、IG、GradientSHAP、Occlusion 直接当作确定的课外方法。** Grad-CAM 和 IG 在 Lecture 6a 中明确讲解；GradientSHAP 和 Occlusion 出现在 Lecture 7b 的示例中。Grad-CAM++ 和 D-RISE 出现在 Lecture 8a 的比较图及表格中，是否只在图中出现就算“介绍过”，应由课程规则或老师确定。

Assignment 2 要求选择 CNN-based model，不选择 Transformer-based model。基于 CNN 的目标检测器原则上是可能方向，但还需要处理检测框和目标实例；本方案选择图像分类，实施更直接。

## 研究问题和设计

### 研究问题

1. 当预训练 CNN 对图片预测错误时，哪些区域对错误类别的分数有较大贡献？这些区域在目标物体上还是在目标外？
2. 课内和课外 XAI 方法对同一张图片、同一类别产生的解释是否相似？
3. 对选定的物体或背景区域进行遮挡后，目标类别的分数如何变化？两个 CNN 是否出现类似现象？

前两个问题是项目核心；跨模型比较可在主要流程跑通后加入。归因图和遮挡实验能提供线索，但单凭一张热力图或一次遮挡，不能确定错误的完整原因。

### 模型

| 模型 | 用途 |
| --- | --- |
| ResNet-18 | 主要 CNN；先在此模型上跑通完整流程 |
| MobileNetV2 | 轻量化 CNN；用相同图片检查观察结果是否相似 |

两个模型都可使用 torchvision 的 ImageNet 预训练权重，不需要从头训练。建议固定并记录各自的 `IMAGENET1K_V1` 权重和对应预处理。即使结果不同，也不能仅凭这次比较断言差异一定由模型架构造成，因为权重和训练过程等因素也可能不同。

### 数据

推荐使用 [ImageNetV2 Matched Frequency](https://github.com/modestyachts/ImageNetV2)，它与 ImageNet 分类任务相匹配。建议先随机抽取约 100–200 张图片，记录真实标签、各模型预测类别与分数，再按照预先写明的规则选择约 12–20 张代表性图片进行详细 XAI 分析，覆盖正确、错误和模型预测不同等情况。

这些数量是建议，并非作业规定。特意挑出的案例适合解释，但不能直接用它们推断模型在全部图片上的错误比例。对候选错误案例，还应人工检查：如果图像中有多个明显物体，预测出另一个实际存在的物体不一定是清晰的错误。

**标签映射注意：**ImageNetV2 的数字文件夹如果直接用普通 `ImageFolder` 读取，类别编号可能因字符串排序错位。可参考 [ImageNetV2 PyTorch 数据加载工具](https://github.com/modestyachts/ImageNetV2_pytorch)，并检查标签与图片类别是否对应；[数据集仓库中的讨论](https://github.com/modestyachts/ImageNetV2/issues/10)说明了这个问题。

## 四种解释方法

### 1. Grad-CAM（课内）

利用目标类别相对于 CNN 卷积层的梯度，生成较粗的区域热力图。对错误预测的类别生成解释，也可对真实类别生成另一张图，比较它们关注的位置。

课程来源：`Lecture6a.pdf`，第 11–12 页。

### 2. Integrated Gradients（课内）

从基准图像逐步过渡到原图并累积梯度，估计输入像素对目标类别输出的贡献。应与 Grad-CAM 使用同一图片和同一目标类别，并记录基准图像及相关参数。

如果图片已经标准化，输入张量的零值不一定代表黑色；需要说明基准在实际图片中对应什么颜色。

课程来源：`Lecture6a.pdf`，第 7–8 页。

### 3. Score-CAM（课外候选）

利用 CNN 特征图构造输入遮罩，通过模型对遮罩后图片的目标类别分数确定权重。与 Grad-CAM 相比，它不依赖梯度计算这些权重。适合比较：对同一错误案例，基于梯度的 Grad-CAM 和基于前向分数的 Score-CAM 是否指出相似区域。

Score-CAM 通常需要多次前向计算，运行可能比 Grad-CAM 慢。

- 论文：[Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPRW_2020/html/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.html)
- 实现：[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

### 4. XRAI（课外候选）

一种基于图像区域的归因方法，建立在 Integrated Gradients 基础上，并结合图像分割生成区域解释。适合比较 IG 的像素归因与 XRAI 的区域归因是否有助于判断物体和背景。

接入 PyTorch 模型时，需要正确处理输入格式、目标类别与梯度。它与课堂 IG 的关系必须在报告中说明，是否认作课外方法需要核对课程其他材料。

- 论文：[XRAI: Better Attributions Through Regions](https://research.google/pubs/xrai-better-attributions-through-regions/)
- 实现：[PAIR-code/saliency](https://github.com/PAIR-code/saliency)
- PyTorch 示例：[Examples_pytorch.ipynb](https://github.com/PAIR-code/saliency/blob/master/Examples_pytorch.ipynb)

### 备选方法

如果 XRAI 接入困难，可考虑 **Ablation-CAM**：通过消融 CNN 特征通道并观察目标输出变化构建解释，`pytorch-grad-cam` 提供其实现。它与 Score-CAM 同属 CAM 类方法；若希望方法思路更丰富，优先尝试 Score-CAM 和 XRAI。是否属于课外方法，仍需核对未提供的材料。

Lecture 8a 的比较图中出现了 **D-RISE**。图像分类方法 RISE 与 D-RISE 有关系但面向不同任务；是否把 RISE 算作“课堂没有介绍过”可能存在争议，因此不作为首选课外方法。

## 实验步骤与验证

### 1. 记录模型预测

对同一批图片分别运行两个模型。保存图片 ID、来源、真实类别、预测类别、目标类别的分数和预测是否正确。可以选择正确案例、两个模型结果不同的案例和共同错误案例进行解释。

### 2. 生成方法对照图

对同一模型、同一张图片和同一目标类别，分别展示 Grad-CAM、IG、Score-CAM 和 XRAI。

如果真实类别是“狗”，模型错误预测为“猫”，则可以分别解释：

- **猫**：哪些区域与错误类别分数有关？
- **狗**：哪些区域与真实类别分数有关？

两个目标回答不同问题，图中应明确标注。不同方法的图可能有不同分辨率、数值尺度和正负含义；统一说明可视化和排名处理方式，不应直接把各自归一化后的颜色强度当作可跨方法比较的绝对数值。

### 3. 用 Insertion / Deletion 验证

- **Deletion**：按照解释图从高到低遮挡区域，观察目标类别分数变化。
- **Insertion**：从基准图开始，逐步加入解释图认为重要的区域，观察目标类别分数如何恢复。
- **随机基线**：按随机顺序操作，与各解释方法进行比较。

使用统一的目标类别、遮挡比例和填充值，并记录设置。如果画的是目标类别概率随操作比例变化的常见曲线，通常比较曲线面积；如果直接复用 Lecture 6a 的代码，要注意它画的是与原始预测的绝对差值，指标方向与概率曲线不同，不能混用。

还可以手工标出部分图片中目标物体的大致范围，对目标附近和背景附近的位置做面积相近的遮挡，并记录目标类别分数如何改变。框内也可能包含背景；遮挡本身可能使图片不自然，因此这些结果只能提供线索，不能单独证明模型的因果决策过程。

## 五人分工

| 成员 | 主要负责 | 具体成果 |
| --- | --- | --- |
| A：模型和数据 | 准备图片、标签映射和两个预训练 CNN；运行基础预测 | 图片清单、标签检查、模型预测结果表 |
| B：课内 XAI | 实现 Grad-CAM 和 IG，整理正确与错误案例 | 两种方法的代码、热力图、初步解释 |
| C：课外方法 1 | 实现 Score-CAM，并和 Grad-CAM 对比 | Score-CAM 代码、对照图、方法原理和结果 |
| D：课外方法 2 | 实现 XRAI，并和 IG 对比 | XRAI 代码、区域解释图、方法原理和结果 |
| E：结果验证与展示整合 | 负责 insertion/deletion、随机基线、结果汇总；组织视频与幻灯片 | 验证曲线、结果总结、展示初稿 |

两张 meme 可以由 B/C 和 D/E 分别提出初稿，A 帮忙检查是否符合实际结果。每个人都应为自己负责的 notebook 部分写说明，并参与课堂问答准备。

## 与评分和提交的关系

本方案旨在覆盖作业中的多种 XAI 方法、课堂外方法尝试、对研究问题的解释分析、报告和展示。两张 meme 应建立在真实观察上，既容易理解又能对应 notebook 中的证据。

视频要求 2 分钟，**先展示两张 meme，再讨论形成这些 meme 的 XAI 分析**。视频会在 11 月 4 日课堂播放，因此实验需要在课堂前有足够完整的结果。其他提交包括 notebook、包含所有 notebook 的 PDF、两张 meme 图片及演示幻灯片；截止时间和格式以 Canvas 为准。

## 当前状态

这份文件是研究方案，**不是已经完成的实验报告**。模型、图片、归因结果、遮挡实验和 meme 尚待实施。建议先用一张图片跑通四种方法，检查 XRAI 接入、Score-CAM 的运行时间和类别标签，再批量分析。
