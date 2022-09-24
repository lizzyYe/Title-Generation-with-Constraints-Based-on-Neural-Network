# Title-Generation-with-Constraints-Based-on-Neural-Network
Neural Headline Generation, Lexical Constraints, Transformer, Dynamic Beam Allocation
# 基于神经网络的带有约束的标题生成 

摘要

自动标题生成任务在金融和新闻等领域都有广泛的应用。本文在标题生成任务中引入词汇约束，基于神经网络在解码端构建带有约束的标题生 成。具体做法是使用 Transformer 作为基线模型，利用动态波束分配算法 在测试阶段进行带有约束的标题解码，并使用 Trie 结构进行约束表示，矢 量化的动态波束分配算法来并行解码操作。实验比较不同约束个数的影响， 同时与不带有约束的贪心算法和波束搜索算法进行时间开销，标题质量对 比，生成标题的质量使用 ROUGE 分数进行评估。实验结果显示矢量化动 态波束分配算法可以在较短时间内完成带有约束的标题生成任务，并在添 加少量约束后标题质量就有了很高的提升。

关键词:神经标题生成，词汇约束，Transformer，动态波束分配，ROUGE

# Title Generation with Constraints Based on Neural Network 
Abstract

Automatic headline generation task is widely used in finance, news and other fields. This paper introduces lexical constraints in the task of headline generation, which is constructed on the decoding side based on neural network. The specific method is to use Transformer as baseline, use Dynamic Beam Allocation algorithm to decode the headline with constraints in the test phase, with Trie structure for constraint expression and the Vectorized Dynamic Beam Allocation algorithm for paralleling decoding operations. The experiment compares the influence of different number of constraints. At the same time, the time cost and headline quality are compared with the greedy algorithm and beam search algorithm without constraints. The quality of generated headlines is evaluated by ROUGE score. The experimental results show that the Vectorized Dynamic Beam Allocation algorithm can complete the task of generating the headline with constraints in a short time, and the quality of generated headlines is greatly improved after adding a few constraints.

Key Words: Neural Headline Generation, Lexical Constraints, Transformer, Dynamic Beam Allocation
