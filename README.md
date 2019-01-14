# Signage-object-detection

---
##  Updata Time:2019_01_14
**Name:** 自动驾驶场景下小且密集的交通标志检测<br>
**Publication category:** 中文核心<br>
**Publication Name:** 智能系统学报 <br>
**Issuing Time:** 2018_4_11 <br>
**Contribution:** 提出用浅层VGG16网络作为物体检测框架R-FCN的主体网络，并改进VGG16网络以检测小的交通标志。 <br>
**Difficulty:** 减小特征图缩放倍数,去掉VGG16网络卷积conv43后面的特征图,使用RPN网络在浅层卷积conv43上提取候选框;特征拼层,将尺度相同的卷积conv41、conv42、conv43层的特征拼接起来形成组合特征。 <br>
**Result:** 改进后的物体检测框架能够检测到更多的小物体,在驭势科技提供的交通标志数据集上检测的准确率mAP达到了65%。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_Chinese/2018_%E8%87%AA%E5%8A%A8%E9%A9%BE%E9%A9%B6%E5%9C%BA%E6%99%AF%E4%B8%8B%E5%B0%8F%E4%B8%94%E5%AF%86%E9%9B%86%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B_%E8%91%9B%E5%9B%AD%E5%9B%AD.pdf<br>
**Name:** 基于感兴趣区域提取与双过滤器的交通标志检测算法 <br>
**Publication category:** 中文核心 <br>
**Publication Name:** 电子测量与仪器学报 <br>
**Issuing Time:** 2018_5_15 <br>
**Contribution:** 设计了一种ROI提取与双过滤器的交通标志检测方案，通过2种具有互补的兴趣区域MSER与WE测量。 <br>
**Difficulty:** 利用MSER与WE对交通标志的ROI提取得到了标志的候选区域，再联合HOG与SVM方法提取候选区域的特征并进行分类，嵌入了上下文感知过滤器与交通灯过滤器，剔除伪标志区域与交通灯。 <br>
**Result:** 本文算法平均消耗时间为0.68 s，与其他算法效率相当，能够满足实时性要求。本文算法的Precision-Recall曲线表现良好，稳定性优异。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_Chinese/%E5%9F%BA%E4%BA%8E%E6%84%9F%E5%85%B4%E8%B6%A3%E5%8C%BA%E5%9F%9F%E6%8F%90%E5%8F%96%E4%B8%8E%E5%8F%8C%E8%BF%87%E6%BB%A4%E5%99%A8%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95_%E6%9B%B9%E5%86%8D%E8%BE%89.pdf<br>
**Name:** 一种基于局部特征的交通标志检测算法的研究 <br>
**Publication category:** 中文核心 <br>
**Publication Name:** 现代电子技术 <br>
**Issuing Time:** 2015_7_1 <br>
**Contribution:** 以圆形标志牌为例,提出一种统一对称局部特征检测模板提取自然场景下获得的目标区域的特征,设计一组模糊规则判定形状,形成一种基于局部特征的交通标志检测算法。 <br>
**Difficulty:** 自适应中值滤波器设计；局部特征的提取，包括统一的对称局部特征检测模板和子模板内含有特征颜色像素个数的隶属度函数。 <br>
**Result:** 该交通标志检测算法检测率基本达到90%左右;对表面有污损的但特征颜色轮廓完整的标志也能正确检测。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_Chinese/%E4%B8%80%E7%A7%8D%E5%9F%BA%E4%BA%8E%E5%B1%80%E9%83%A8%E7%89%B9%E5%BE%81%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95%E7%9A%84%E7%A0%94%E7%A9%B6_%E5%AE%8B%E5%A9%80%E5%A8%9C.pdf<br>
**Name:** 一种基于局部特征的交通标志检测算法的研究 <br>
**Publication category:** 中文核心 <br>
**Publication Name:** 现代电子技术 <br>
**Issuing Time:** 2015_7_1 <br>
**Contribution:** 以圆形标志牌为例,提出一种统一对称局部特征检测模板提取自然场景下获得的目标区域的特征,设计一组模糊规则判定形状,形成一种基于局部特征的交通标志检测算法。 <br>
**Difficulty:** 自适应中值滤波器设计；局部特征的提取，包括统一的对称局部特征检测模板和子模板内含有特征颜色像素个数的隶属度函数。 <br>
**Result:** 该交通标志检测算法检测率基本达到90%左右;对表面有污损的但特征颜色轮廓完整的标志也能正确检测。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_Chinese/%E4%B8%80%E7%A7%8D%E5%9F%BA%E4%BA%8E%E5%B1%80%E9%83%A8%E7%89%B9%E5%BE%81%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95%E7%9A%84%E7%A0%94%E7%A9%B6_%E5%AE%8B%E5%A9%80%E5%A8%9C.pdf<br>

---
##  Updata Time:2019_01_13
**Name:** 用于车载网络中的真实交通标志的深度检测网络<br>
**Publication category:** ScienceDirect <br>
**Publication Name:** Computer Networks <br>
**Issuing Time:** 2018_4_2 <br>
**Contribution:** 提出了一种新颖的端到端深度网络，通过两阶段调整策略提取区域提案。添加新的AT（注意网络），根据颜色特征定位所有潜在的RoI。 <br>
**Difficulty:** 采用两阶段调整策略定位RoI，以粗略到精细的方式加速对象检测过程。 <br>
**Result:** 仅生成Faster-RCNN的1/14锚，使用ZF-Net检测速度提高了约2fps，在两个基准测试中的平均mAP分别为80.31％和94.95％，分别比使用VGG16的Faster-RCNN高9.69％和7.88％。实验结果表明该网络在速度和mAP方面都优于以前的方法,在小尺寸物体上表现更好。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_English/%E8%AE%BA%E6%96%87_2018_%E7%94%A8%E4%BA%8E%E8%BD%A6%E8%BD%BD%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E7%9C%9F%E5%AE%9E%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E7%9A%84%E6%B7%B1%E5%BA%A6%E6%A3%80%E6%B5%8B%E7%BD%91%E7%BB%9C.pdf<br>
**Thumbnail:** <br> 
<div align=center><img width="439" height="271" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/14-1.PNG"/></div>  <br>
<div align=center><img width="543" height="184" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/14-2.PNG"/></div>  <br>
**Name:** 通过兴趣区域提取检测交通标志 <br>
**Publication category:** ScienceDirect <br>
**Publication Name:** Pattern Recognition <br>
**Issuing Time:** 2015_6_9 <br>
**Contribution:** 使用实体图像分析和模式识别技术的组合来解决移动地图数据中的兴趣区域提取交通标志检测问题。 <br>
**Difficulty:** 兴趣区域提取通过两种互补算法最大稳定极值区域（MSER）检测器和基于波的检测器（WaDe）实现。 <br>
**Result:** 提出的交通标志检测系统在诸如变化的照明、部分遮挡、大规模变化的挑战条件下表现良好。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_English/%E8%AE%BA%E6%96%87_2015_%E9%80%9A%E8%BF%87%E5%85%B4%E8%B6%A3%E5%8C%BA%E5%9F%9F%E6%8F%90%E5%8F%96%E6%A3%80%E6%B5%8B%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97.pdf<br>
<br>

---
##  Updata Time:2019_01_12
**Name:** 基于卷积神经网络的同时交通标志检测与边界估计<br>
**Publication category:** IEEE <br>
**Publication Name:** TITS <br>
**Issuing Time:** 2018_3_8 <br>
**Contribution:** 提出估计交通标志的位置与其精确边界的方法。推广了物体边界框检测问题并制定了物体姿态估计问题，使用CNN建模。为实现检测速度，考虑交通标志的特征，探索了性能最佳的基础网络并修剪了不必要的网络层。优化网络输入的分辨率，在速度和准确度之间实现折衷. <br>
**Difficulty:** 基于SSD结构构建CNN块，网络执行姿态估计，转换成相应交通标志的边界估计。从特征图中，通过两个分离的卷积层，即姿势回归层和形状分类层结合将卷积输出分别转换为2D姿势值和类概率的连续操作来估计2D姿势和形状类概率。使用获得的2D姿势和形状类概率计算边界角。 <br>
**Result:** 如下图所示，两种型号在0.5 IoU时均达到0.8 mAP以上，甚至在0.7 IoU时也达到0.8 mAP。表明该方法能够准确地检测交通标志。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_English/%E5%9F%BA%E4%BA%8E%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%90%8C%E6%97%B6%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%BE%B9%E7%95%8C%E4%BC%B0%E8%AE%A1.pdf<br>
**Thumbnail:** <br> 
<div align=center><img width="738" height="490" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/8-1.PNG"/></div>  <br>
<br>

**Name:** 基于深度学习的交通标志检测与识别研究与应用 <br>
**Publication category:** IEEE <br>
**Publication Name:** ICRIS <br>
**Issuing Time:** 2018_7_12 <br>
**Contribution:** 提出了一种基于SSD算法的交通标志检测与识别新框架；将SSD扩展到新的应用程序交通标志检测和识别问题；缩短训练时间，提高Softmax分类器的准确性。 <br>
**Difficulty:** 向数据集添加更多数据，一些交通标志具有水平或垂直镜像的不变性，并且训练样本的图像可以随机地旋转小角度以处理真实场景中的符号倾斜问题；防止模型过度拟合使用丢失算法。 <br>
**Result:** 最大迭代次数为20000次，准确率可达到约96％。网络在图像严重倾斜、质量差、最小边界框不够准确等极端条件下有改进的余地。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Conference%20abstracts_English/%E4%BC%9A%E8%AE%AEICRIS_2018_%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB%E7%A0%94%E7%A9%B6%E4%B8%8E%E5%BA%94%E7%94%A8.pdf <br>
<br>

**Name:** 实时交通标志检测与分类 <br>
**Publication category:** IEEE <br>
**Publication Name:** TITS <br>
**Issuing Time:** 2015_10_12 <br>
**Contribution:** 提出了一种极快的检测模块，比现有的最佳检测模块快20倍。提出了一种颜色概率模型来处理交通标志的颜色信息，以增强交通标志的特定颜色（如红色，蓝色和黄色），抑制背景颜色，减少算法的搜索空间,缩短检测时间；提取交通标志提案而不是滑动窗口检测，结合SVM和CNN检测和分类交通标志；构建了一个中国交通标志数据集（CTSD）. <br>
**Difficulty:** 检测模块利用颜色概率模型和MSER区域检测器提取交通标志提案，使用SVM分类器过滤误报并基于新颖的颜色HOG特征将剩余的提议分类。 <br>
**Result:** 准确性略差，但速度快了20倍。每个图像的平均时间仅为0.162秒，而其他方法通常需要几秒钟。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_English/%E8%AE%BA%E6%96%87_2016_%E5%AE%9E%E6%97%B6%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E7%B1%BB.pdf<br>
**Thumbnail:** <br> 
<div align=center><img width="748" height="302" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/10-1.PNG"/></div>  <br>
<br>

**Name:** 基于图模型与卷积神经网络的交通标志识别方法 <br>
**Publication category:** EI <br>
**Publication Name:** 交通运输工程学报 <br>
**Issuing Time:** 2016_10_15 <br>
**Contribution:** 提出基于图模型的层次显著性检测方法HSDBGM，有效地融合了局部区域的细节信息与图像的结构信息。建立面向应用的 R-CNN交通标志识别系统，通过显著性检测方法提取ROI，并与CNN结合。 <br>
**Difficulty:** 提出一种基于先验位置约束与局部特征（颜色与边界）的层次显著性模型。 <br>
**Result:** 针对限速标志，基于UCM超像素区域的图模型比基于简单线性迭代聚类（SLIC）超像素的图模型更有利于获取上层显著度图的大尺度结构信息；基于先验位置约束与局部特征（颜色与边界）的层次显著性模型查准率为0.65，查全率为0.8，Ｆ指数为0.73，均高于其他同类基于超像素的显著性检测算法；基于具体检测任务的CNN预训练策略扩展了德国交通标志识别库（GTSRB）的样本集，更好地学习目标内部的局部精细特征，提高了学习与识别能力，总识别率为98.85%，高于SVM分类器的96.73%。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_Chinese/%E5%9F%BA%E4%BA%8E%E5%9B%BE%E6%A8%A1%E5%9E%8B%E4%B8%8E%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E8%AF%86%E5%88%AB%E6%96%B9%E6%B3%95_%E5%88%98%E5%8D%A0%E6%96%87.pdf<br>
**Thumbnail:** <br> 
<div align=center><img width="441" height="748" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/11-1.PNG"/></div>  <br>
<br>

**Name:** 基于图模型与卷积神经网络的交通标志识别方法 <br>
**Publication category:** ScienceDirect <br>
**Publication Name:** Expert Systems With Applications <br>
**Issuing Time:** 2016_4_15 <br>
**Contribution:** 提出一种无参数的圆检测算法EDCircles，即算法在运行测试图像之前不需要训练任何图像来调整任何参数。 <br>
**Difficulty:** EDCircles适用于灰度图像，可以利用颜色信息来提高性能。 <br>
**Result:** 使用提出的RGB阈值技术，即RGB + EDCircles + RGBNDiff，可以获得最佳结果。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_English/%E8%AE%BA%E6%96%87_2016_%E8%AE%BA%E5%9C%86%E5%BD%A2%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E7%9A%84%E6%A3%80%E6%B5%8B%E4%B8%8E%E8%AF%86%E5%88%AB.pdf<br>
**Thumbnail:** <br> 
<div align=center><img width="889" height="487" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/12-1.PNG"/></div>  <br>
<div align=center><img width="887" height="414" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/12-2.PNG"/></div>  <br>
<br>

**Name:** 使用卷积神经网络检测和分类交通标志的实用方法 <br>
**Publication category:** ScienceDirect <br>
**Publication Name:** Robotics and Autonomous Systems <br>
**Issuing Time:** 2016_7_19 <br>
**Contribution:** 提出一种轻量级和精确的ConvNet检测交通标志，进一步优化ConvNet使交通标志分类更快、更准确。 <br>
**Difficulty:** 如何在ConvNet中实现实时滑动窗口检测器。检测模块消耗更多时间，尤其是当其应用于高分辨率图像时。 <br>
**Result:** ConvNet检测交通标志的平均精度等于99.89%。滑动窗口可以实现每秒处理37.72个高分辨率图像并定位交通标志。ConvNet能够分类99.55%测试样本，稳定性分析表明ConvNet能够容忍高斯噪声σ<10。 <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_English/%E8%AE%BA%E6%96%87_2016_%E4%BD%BF%E7%94%A8%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%A3%80%E6%B5%8B%E5%92%8C%E5%88%86%E7%B1%BB%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E7%9A%84%E5%AE%9E%E7%94%A8%E6%96%B9%E6%B3%95.pdf<br>
<br>

---
##  Updata Time:2019_01_11
**Name:** 基于语义分割的交通标志检测与深度学习技术识别 <br>
**Publication category:** IEEE<br>
**Publication Name:** ICCP <br>
**Issuing Time:** 2018_11_7 <br>
**Contribution:** 提出了改进的完全卷积网络（FCN）的语义分割模型FCN8以提取交通标志的ROI.通过用扩张卷积（速率4）替换最后的卷积层而不是传统的卷积（速率1）来改进FCN8。最终精度增加了0.7％，提高了不太明确的对象类别（例如交通标志和信号量）的分割质量。 <br>
**Shortcoming:** 非常靠近（在杆子上）的交通标志很可能被视为一个整体；分段模块检测了多个未包含在GTSRB数据集中的交通标志。<br>
**Difficulty:** 训练CNN进行交通标志检测。修改了完全卷积网络（FCN）架构，并从生成的语义映射中提取交通标志区域。<br>
**Result:** <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Conference%20abstracts_English/%E4%BC%9A%E8%AE%AE_2018_%E5%9F%BA%E4%BA%8E%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8A%80%E6%9C%AF%E8%AF%86%E5%88%AB.pdf <br>
**Thumbnail:** <br> 
<div align=center><img width="550" height="107" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/1-1.png"/></div>  <br>
<div align=center><img width="550" height="427" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/1-2.png"/></div>  <br>
<br>

**Name:** 一种基于HDR技术的交通标志牌检测和识别方法 <br>
**Publication category:** 中文核心期刊 <br>
**Publication Name:** 激光与光电子学进展 <br>
**Issuing Time:** 2018_4_27 <br>
**Contribution:** 提出的基于HDR技术的交通标志牌检测和识别方法在光线强烈和光线昏暗的极端环境下比其他传统方法更优越。 <br>
**Difficulty:** 提出了一种基于高动态范围（HDR）技术的识别方法。利用改进的逆色调映射算法，在不同曝光条件下捕获的LDR图像在亮度范围内自适应拉伸，分别生成两个子图像，然后生成HDR图像。采用多重曝光融合算法代替原始LDR图像进行识别。<br>
**Shortcoming:**  逆色调映射和多曝光融合处理时间较长，实时性不高。<br>
**Result:**  <br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_Chinese/%E4%B8%80%E7%A7%8D%E5%9F%BA%E4%BA%8EHDR%E6%8A%80%E6%9C%AF%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E7%89%8C%E6%A3%80%E6%B5%8B%E5%92%8C%E8%AF%86%E5%88%AB%E6%96%B9%E6%B3%95_%E5%BC%A0%E6%B7%91%E8%8A%B3.pdf <br>
**Thumbnail:** <br> 
<div align=center><img width="1267" height="628" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/2-1.jpg"/></div>  <br>
<div align=center><img width="1267" height="626" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/2-2.jpg"/></div>  <br>
<div align=center><img width="1268" height="161" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/2-3.jpg"/></div>  <br>
<div align=center><img width="1268" height="163" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/2-4.jpg"/></div>  <br>
<br>

**Name:** 一种基于深度卷积神经网络的交通标志检测算法 <br>
**Publication category:** IEEE <br>
**Publication Name:** ICSIP <br>
**Issuing Time:** 2017_3_30 <br>
**Contribution:** 提出了一种基于DCNN的中国交通标志检测算法。该方法可以检测中国所有7大类交通标志，实时性和精度都高。<br>
**Difficulty:** 交通标志图像的多样化背景；中国的交通标志文字较多，更复杂。<br>
**Result:** 该方法具有实时检测速度和99％以上的检测精度，视频序列的检测时间是实时的。 <br>
**Link**https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Conference%20abstracts_English/%E4%BC%9A%E8%AE%AE_2016_%E4%B8%80%E7%A7%8D%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95.pdf <br>

**Name:** 一种基于颜色分割和鲁棒形状匹配的交通标志检测方法 <br>
**Publication category:** Elsevier BV<br>
**Publication Name:** Neurocomputing <br>
**Issuing Time:** 2015_12_2 <br>
**Contribution:** 提出了一种基于颜色不变量和改进的PHOG来检测交通标志的新方法。<br>
**Difficulty:** 通过聚类颜色不变量特征将图像分割成不同的区域以获得感兴趣的候选区域，采用PHOG特征来表示ROI的形状特征。<br>
**Result:** 所提出的基于颜色不变的聚类在交通标志分割中比HSI分割有效，对诸如阴影，遮挡，天气，复杂背景等各种因素非常稳健。<br>
**Link:**https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_English/%E4%B8%80%E7%A7%8D%E5%9F%BA%E4%BA%8E%E9%A2%9C%E8%89%B2%E5%88%86%E5%89%B2%E5%92%8C%E9%B2%81%E6%A3%92%E5%BD%A2%E7%8A%B6%E5%8C%B9%E9%85%8D%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E6%96%B9%E6%B3%95.pdf<br>
**Thumbnail:** <br> 
<div align=center><img width="858" height="475" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/4-1.PNG"/></div>  <br>
<br>

**Name:** 自然环境下圆形禁令交通标志检测 <br>
**Publication category:** EI<br>
**Publication Name:** 武汉大学学报(信息科学版) <br>
**Issuing Time:** 2016_12_5<br>
**Contribution:** 针对中国圆形禁令交通标志提出一种颜色分割与形状分析的标志检测算法。<br>
**Difficulty:** 在RGB颜色空间选择一个基础通道，计算相对通道差值，通过实验数据拟合阈值曲线进行自适应颜色分割;对分割图像提取边缘，提出最小二乘椭圆拟合后验偏差估计法对边缘进行筛选。<br>
**Shortcoming:** 存在错检漏检；在普适性及自动化上有所欠缺。<br>
**Result:** 有效检测率高达97.92％，平均耗时只有65.42ms，对亮度、视角变化、褪色、模糊等情况具有较好的鲁棒性。<br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_Chinese/%E8%87%AA%E7%84%B6%E7%8E%AF%E5%A2%83%E4%B8%8B%E5%9C%86%E5%BD%A2%E7%A6%81%E4%BB%A4%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B_%E6%9D%8E%E8%BF%8E%E6%9D%BE.pdf<br>
<br>

**Name:** 自然环境下圆形禁令交通标志检测 <br>
**Publication category:** EI<br>
**Publication Name:** 吉林大学学报(工学版) <br>
**Issuing Time:** 2017_3_9<br>
**Contribution:** 提出了一种基于深度属性学习的交通标志检测方法。引入了形状，颜色和模式三个视觉属性，在CNN中加入属性学习约束。<br>
**Difficulty:** 在HSV颜色空间上提取最大稳定极值区域（MSER）以提取交通标志候选区域。<br>
**Shortcoming:** 交通标志表面图案较为复杂，元素较多时，候选区域提取的性能会有一定的损失。<br>
**Result:** 有效提高交通标志检测精度和召回率<br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Research%20articles_Chinese/%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%B1%9E%E6%80%A7%E5%AD%A6%E4%B9%A0%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B_%E7%8E%8B%E6%96%B9%E7%9F%B3.pdf<br>
**Thumbnail:** <br> 
<div align=center><img width="889" height="345" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/6-1.PNG"/></div>  <br>
<br>

**Name:** 基于颜色特征和神经网络的交通标志检测与分类 <br>
**Publication category:** IEEE<br>
**Publication Name:** ICICPI<br>
**Issuing Time:** 2017_2_23<br>
**Contribution:** 提出了一个框架，可以从图像中检测和分类不同类型的交通标志。<br>
**Difficulty:** 将RGB转换为HSV，颜色阈值技术用于检测交通标志，图像转为仅由黑色和白色组成的二进制图像，再根据每个像素的8连通性标记。<br>
**Result:** 检出率在90％以上，识别准确率在88％以上。<br>
**Link:** https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Conference%20abstracts_English/%E4%BC%9A%E8%AE%AE_2016_%E5%9F%BA%E4%BA%8E%E9%A2%9C%E8%89%B2%E7%89%B9%E5%BE%81%E5%92%8C%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BA%A4%E9%80%9A%E6%A0%87%E5%BF%97%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E7%B1%BB.pdf<br>
**Thumbnail:** <br> 
<div align=center><img width="736" height="351" src="https://github.com/ZGX010/Signage-object-detection/blob/master/articles/Screenshot/7-1.PNG"/></div>  <br>
<br>
