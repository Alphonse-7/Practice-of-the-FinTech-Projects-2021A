# Practice-of-the-FinTech-Projects-2021A
Practice of the FinTech Projects, 2021 Autumn.



## 教师与助教

Instructor: Jianwu Lin

TA: Dingjun Wu



## 课程回顾



| 时间                 | 安排                                      |
| -------------------- | ----------------------------------------- |
| 2021/9/16（第1周）   | 企业参观 - 私募排排网                     |
| 2021/9/23（第2周）   | 企业讲座 - 开放获取与期刊投稿的关系及策略 |
| 2021/9/30（第3周）   | 企业讲座 - 天云大数据                     |
| 2021/10/7（第4周）   | 国庆放假                                  |
| 2021/10/14（第5周）  | 第1次论文分享；各小组研究汇报             |
| 2021/10/21（第6周）  | 第2次论文分享；各小组研究汇报             |
| 2021/10/28（第7周）  | 第3次论文分享；各小组研究汇报             |
| 2021/11/4（第8周）   | 第4次论文分享；各小组研究汇报             |
| 2021/11/11（第9周）  | 第5次论文分享；各小组研究汇报             |
| 2021/11/18（第10周） | 第6次论文分享；各小组研究汇报             |
| 2021/11/25（第11周） | 第7次论文分享；各小组研究汇报             |
| 2021/11/2（第12周）  | 第8次论文分享；各小组研究汇报             |
| 2021/12/9（第13周）  | 第9次论文分享；各小组研究汇报             |
| 2021/12/16（第14周） | 调课                                      |
| 2021/12/23（第15周） | 第10次论文分享；各小组研究汇报            |



## 论文分享环节



| 小组成员               | 分享次序 | 论文分享时间         | 论文题目                                                     |
| ---------------------- | -------- | :------------------- | ------------------------------------------------------------ |
| 郭泓、苏昭帆           | 1        | 2021/10/14（第5周）  | Marketing Making via Reinforcement Learning                  |
| 徐有珩、李晨霄、徐嘉营 | 2        | 2021/10/21（第6周）  | A picture is worth a thousand words: Measuring investor sentiment by combining machine learning and photos from news |
| 周梅、陈悦、周雨慧     | 3        | 2021/10/28（第7周）  | Pre-trained Models for Natural Language Processing: A Survey |
| 陈丽行、许越玥、周俊池 | 4        | 2021/11/4（第8周）   | DisenKGAT: Knowledge Graph Embedding with Disentangled Graph Attention Network |
| 孙佳琪、刘睿、孙继丰   | 5        | 2021/11/11（第9周）  | Financial time series forecasting with multi-modality graph neural network |
| 张胜楠、刘长           | 6        | 2021/11/18（第10周） | Enhancing a Pairs Trading strategy                           |
| 李代猛、罗跃           | 7        | 2021/11/25（第11周） | Machine learning in the Chinese stock market                 |
| 赵越，吴定俊，蔡紫宴   | 8        | 2021/10/14（第12周） | TabNet: Attentive Interpretable Tabular Learning             |
| 蒋子函、文家伟         | 9        | 2021/12/2（第13周）  | Empirical Asset Pricing via Machine Learning                 |
| /                      | /        | 2021/12/9（第14周）  | 调课一周                                                     |
| 吴冠陞 史可为          | 10-1     | 2021/12/23（第15周） | A Unified Model for Opinion Target Extraction and Target Sentiment Prediction |
| 苗佳、王禹珂           | 10-2     | 2021/12/23（第15周） | Direct Estimation of Equity Market Impact                    |



### 1

论文题目：Market Making via Reinforcement Learning

论文分享者：郭泓、苏昭帆

来源：AAMAS 2018

简介：做市是一个基本的交易问题。在做市过程中，经纪人通过不断地提出买卖证券来提供流动性。在本文中，作者构建了一个模拟的限价订单簿，并利用它来设计了一个时序差分的强化学习智能体。作者使用瓷砖编码的线性组合作为价值函数逼近器，并设计一个自定义奖励函数来控制库存风险。作者提出的智能体表现优于Baseline和在线学习方法，因此证明了方法的有效性。

### 2

论文题目：A picture is worth a thousand words: Measuring investor sentiment by combining machine learning and photos from news

论文分享者：徐有珩、李晨霄、徐嘉营 

来源：Journal of Financial Economics

简介：本文通过将机器学习应用于照片构建情绪指标。从新闻照片样本中获得的每日市场的新闻构建新闻文本情绪和新闻照片情绪。首先，悲观的照片情绪会预测一种市场反转，可利用合适的时机获取收益。其次，文本中的情绪和照片中的情绪在预测收益上呈替代关系。最后，本文解释了照片中的情绪是如何和金融市场背景相关的。

### 3

论文题目：Pre-trained Models for Natural Language Processing: A Survey

论文分享者：周梅、周雨慧、陈悦 

来源：Science China. Technological Sciences, 2020, 63(10): 1872–1897. DOI:10.1007/s11431-020-1647-3.

简介：随着预训练模型(pre- training model, PTMs)的出现，自然语言处理(NLP)进入了一个新的时代， 该综述提供了一个对NLP的PTMs的全面回顾： 1、简要介绍语言表征学习及其研究进展。 2、从四个不同的角度系统地对现有的PTMs进行分类。3、描述如何使PTMs适应下游任务。 4、提出了PTMs未来研究的一些潜在方向。 

### 4

论文题目：DisenKGAT: Knowledge Graph Embedding with Disentangled Graph Attention Network 

论文分享者：陈立行，许越玥，周俊池

来源：CIKM 2021

简介：该文提出了一个知识图谱解耦表征方法，将每个实体表征为多个独立的向量，从而提升模型的表达能力。一方面，在知识图谱卷积中引入关系感知信息聚合机制，促使表征的每个成分聚合到不同的信息，此部分实现了“微观解耦”；另一方面，通过添加互信息正则项来增强表征中每个成分之间的独立性，从而实现了“宏观解耦”。最后，在常用的两个基准数据集上验证了解耦表征可以有效的提升性能。

### 5

论文分享者：孙继丰、孙佳琪、刘睿

论文题目：使用多模态图神经网络进行金融时间序列预测

来源：Pattern Recognition ( IF 7.740 ) Pub Date : 2021-08-02 , DOI: 10.1016/j.patcog.2021.108218
Dawei Cheng, Fangzhou Yang, Sheng Xiang, Jin Liu

简介：金融时间序列分析在对冲市场风险和优化投资决策方面发挥着核心作用。这是一项具有挑战性的任务，因为问题总是伴随着多模态流和超前滞后效应。例如，股票的价格走势是复杂市场状态在不同扩散速度下的反映，包括历史价格序列、媒体新闻、相关事件等。此外，金融行业要求预测模型具有可解释性和合规性。因此，在本文中，我们提出了一种多模态图神经网络 (MAGNN)，以从这些多模态输入中学习以进行金融时间序列预测。异构图网络由我们的金融知识图中的源作为节点和关系作为边构建。为了确保模型的可解释性，我们利用两阶段注意力机制进行联合优化，允许最终用户调查内部模态和跨模态源的重要性。对真实世界数据集的大量实验证明了 MAGNN 在金融市场预测中的卓越性能。我们的方法为投资者提供了一个有利可图且可解释的选择，并使他们能够做出明智的投资决策。

### 6

论文题目：基于机器学习方法的增强型配对交易策略

论文分享者：刘长、张胜楠

来源：Expert Systems with Applications(IF:6.95) .Pub Date : 2020-05-04 

简介：配对交易是对冲基金所采用的最有价值的市场中性策略之一。该方法以相对估价为核心克服了困难的证券估值过程，因而特别具有吸引力。通过买入相对低估的证券，卖出相对高估的证券，从而可以在配对资产价格趋同时获利。然而，随着数据可利用程度的提高，寻找有价值的配对资产变得越来越困难。在本文工作中，我们解决了两个问题：(i) 如何在受限的搜索空间中寻找可获利的配对资产以及 (ii) 如何避免配对资产价值长时间的发散而导致收益的长期下降。为了解决这些困难，本文详细研究了富有前途的机器学习方法在该领域的应用。我们建议采用一种集成的无监督学习算法OPTICS来处理问题 (i)。结果表明，与标准的配对搜索方法相比，用本文建议的方法所获得投资组合的平均夏普比率为3.79，而标准配对方法所得的夏普比率为3.58和2.59。对于问题 (ii)，我们引入了一个基于预测的交易模型，能够将投资组合的下跌周期缩短75%。当然，这是以整体盈利能力下降为代价的。最后，本文使用ARMA模型、LSTM和LSTM编解码器对所提出的策略进行了测试。该工作以2009年1月至2018年12月的一组由208支与大宗商品挂钩的ETF的5分钟价格数据作为模拟测算对象并考虑了交易成本。

### 7

论文题目：Machine learning in the Chinese stock market

论文分享者：李代猛、罗跃

来源：Journal of Financial Economics. Pub Date : 2021-08

简介：来自苏黎世大学的Markus Leippold，Qian Wang以及来自浙江大学的Wenyu Zhou在国际金融学顶级期刊《Journal of Financial Economics》发表文章“Machine learning in the Chinese stock market”，文章利用机器学习技术，从资本市场特征入手对我国股票收益的可预测性、无做空的投资组合收益等进行研究，扩充了我国资本市场领域的相关文献。

### 8

论文题目：TabNet: Attentive Interpretable Tabular Learning

论文分享者：赵越、蔡紫宴、吴定俊

来源：AAAI 2021（2019年提出）

简介：这是来自 Google Cloud AI 团队的研究工作。虽然深度神经网络（DNN）已经在图像、文本、音频等领域方面取得显著成功，但是各大以表格数据为主的数据挖掘比赛，仍然是基于决策树的模型（LightGBM、Xgboost等）的主场。作者对其中的原因进行了分析，并尝试将决策树模型的优点融合进DNN架构，提出了一种新的高性能和可解释的表格数据学习架构——TabNet。TabNet吸收了决策树模型和DNN架构的优点，在表格数据的预测和分类任务上取得了良好效果，目前已经成为kaggle等竞赛平台最热的模型之一。代码实现（Pytorch）：https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tabnet.py
API调用： https://github.com/dreamquark-ai/tabnet

附基于线下商店销量预测数据集的比较实验：https://github.com/Alphonse-7/Fintech-Project-Practice-2021/tree/main/TabNet

### 9

论文题目：Empirical Asset Pricing via Machine Learning

论文分享者：文家伟、蒋子函

来源：The Review of Financial Studies, Volume 33, Issue 5, May 2020, Pages 2223–2273

简介：该文对使用机器学习做实证资产定价的经典问题（即测度资产的风险溢价）进行了可比较的分析，总结了几类机器学习模型对比OLS的结果分析，强调了GBDT及神经网络模型ANN比传统线性回归获得了更好的结果。最终表明使用机器学习的投资者可获得巨大的经济收益，甚至可比现有文献中基于回归的策略表现高出一倍。
代码地址：https://github.com/xiubooth/ML_Codes

### 10-1

论文题目：A Unified Model for Opinion Target Extraction and Target Sentiment Prediction

论文分享者：史可为、吴冠陞

论文来源：AAAI Technical Track: Natural Language Processing

简介：近年来NLP技术在金融领域的应用与研究产生了重要的影响。该文主要提出一种端到端的方案，通过一个应用统一标记方案的统一模型来完成TBSA。模型包括两个LSTM，上层预测统一标签进行初步TBSA，下层通过辅助目标边界预测来指导上层网络提升性能。

### 10-2

论文题目：Direct Estimation of Equity Market Impact

论文分享者：苗佳、王禹珂

论文来源：Risk

简介：大宗交易对市场价格的影响是一个被广泛讨论但很少衡量的现象，对卖方和买方参与者至关重要。文章使用一个简单但现实的理论框架，分析了花旗集团(Citigroup)美国股票交易部门的大量数据集。文章将模型适用于各种股票，确定这些系数对波动率、平均日成交量和成交量等参数的依赖性。文章拒绝使用普通的平方根模型，将临时影响作为贸易率的函数，支持在考虑的订单规模范围内采用3/5幂律。结果可以直接结合到最优贸易调度算法和交易前和交易后的成本估计。

