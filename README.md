# Title-Generation-with-Constraints-Based-on-Neural-Network
Neural Headline Generation, Lexical Constraints, Transformer, Dynamic Beam Allocation
## 基于神经网络的带有约束的标题生成 

自动标题生成任务在金融和新闻等领域都有广泛的应用。本文在标题生成任务中引入词汇约束，基于神经网络在解码端构建带有约束的标题生 成。具体做法是使用 Transformer 作为基线模型，利用动态波束分配算法 在测试阶段进行带有约束的标题解码，并使用 Trie 结构进行约束表示，矢 量化的动态波束分配算法来并行解码操作。实验比较不同约束个数的影响， 同时与不带有约束的贪心算法和波束搜索算法进行时间开销，标题质量对 比，生成标题的质量使用 ROUGE 分数进行评估。实验结果显示矢量化动 态波束分配算法可以在较短时间内完成带有约束的标题生成任务，并在添 加少量约束后标题质量就有了很高的提升。

关键词:神经标题生成，词汇约束，Transformer，动态波束分配，ROUGE

# Title Generation with Constraints Based on Neural Network 

Abstract

Automatic headline generation task is widely used in finance, news and other fields. This paper introduces lexical constraints in the task of headline generation, which is constructed on the decoding side based on neural network. The specific method is to use Transformer as baseline, use Dynamic Beam Allocation algorithm to decode the headline with constraints in the test phase, with Trie structure for constraint expression and the Vectorized Dynamic Beam Allocation algorithm for paralleling decoding operations. The experiment compares the influence of different number of constraints. At the same time, the time cost and headline quality are compared with the greedy algorithm and beam search algorithm without constraints. The quality of generated headlines is evaluated by ROUGE score. The experimental results show that the Vectorized Dynamic Beam Allocation algorithm can complete the task of generating the headline with constraints in a short time, and the quality of generated headlines is greatly improved after adding a few constraints.

Key Words: Neural Headline Generation, Lexical Constraints, Transformer, Dynamic Beam Allocation

### 参考文献 Reference
[1] 池军奇. 基于深度语义挖掘的标题生成技术研究与实现[D].北京邮电大学,2019.

[2] 冯浩. 基于 Attention 机制的双向 LSTM 在文本标题生成中的研究与应用[D].华北理工大学,2020.

[3] 姜志祥,叶青,傅晗,张帆.基于自注意力与指针网络的自动摘要模型[J].计算机工程与设计,2021,42(03):711-718.

[4] 李慧,陈红倩,马丽仪,祁梅.结合注意力机制的新闻标题生成模型[J].山西大学学报(自然科学版),2017,40(04):670-675.

[5] 李晨斌,詹国华,李志华.基于改进 Encoder-Decoder 模型的新闻摘要生成方法[J].计算机应用,2019,39(S2):20-23.

[6] 李金鹏,张闯,陈小军,胡玥,廖鹏程.自动文本摘要研究综述[J].计算机研究与发展,2021,58(01):1-21.

[7] 刘一闻,李泽魁,秦玉芳. 基于深度神经网络的稿件标题生成[A]. 中国新闻技术工作者联合会.中国新闻技术工作者联合会 2020 年学术年会论文集[C].中国新闻技术工作者联合会:中国新闻技术工作者联合会,2020:7.

[8] 马雪雯. 英文文本标题自动生成方法研究[D].山西大学,2020.

[9] 庞超. 神经网络在新闻标题生成中的研究[D].北京交通大学,2018.

[10] 周志华.机器学习[M].清华大学出版社,2016:97-115.

[11] 张仕森. 基于聚类和神经网络的文章标题生成系统研究[D].上海工程技术大学,2020.

[12] Ayana,Shi-Qi Shen,Yan-Kai Lin,Cun-Chao Tu,Yu Zhao,Zhi-Yuan Liu,Mao-Song Sun. Recent Advances on Neural Headline Generation[J]. Journal of Computer Science and Technology,2017,32(4).

[13] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.

[14] Chen G, Chen Y, Wang Y, et al. Lexical-constraint-aware neural machine translation via data augmentation[C]//Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20. 2020: 3587-3593.

[15] Dinu G, Mathur P, Federico M, et al. Training Neural Machine Translation to Apply Terminology Constraints[C]//Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019: 3063-3068.

[16] Gong Y, Liu X. Generic text summarization using relevance measure and latent semantic analysis[C]//Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval. 2001: 19-25.

[17] Hu B, Chen Q, Zhu F. LCSTS: A Large Scale Chinese Short Text Summarization Dataset[C]//Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. 2015: 1967-1972.

[18] Hokamp C,Liu Q.Lexically constrained decoding for sequence generation using grid beam search[C].Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics.2017,1,:1535-1546.

[19] Hu J E, Khayrallah H, Culkin R, et al. Improved lexically constrained decoding for translation and monolingual rewriting[C]//Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019: 839-850.

[20] Jin D, Jin Z, Zhou J T, et al. Hooks in the Headline: Learning to Generate Headlines with Controlled Styles[C]//Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020: 5082-5093.

[21] Lin C Y. Rouge: A package for automatic evaluation of summaries[C]//Text summarization branches out. 2004: 74-81.

[22] Mihalcea R, Tarau P. Textrank: Bringing order into text[C]//Proceedings of the 2004 conference on empirical methods in natural language processing. 2004: 404-411.

[23] Post M, Vilar D. Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation[C]//Proceedings of NAACL-HLT. 2018: 1314-1324.

[24] Rush A M, Chopra S, Weston J. A Neural Attention Model for Abstractive Sentence Summarization[C]//Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. 2015: 379-389.

[25] Salton G , Yu C T . On the construction of effective vocabularies for information retrieval[J]. Acm Sigplan Notices, 1975, 10(1):48-60.

[26] Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[J]. arXiv preprint arXiv:1409.3215, 2014.

[27] See A Liu P J, Manning C D. Get To The Point: Summarization with Pointer-Generator Networks[C]//Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017: 1073-1083.

[28] Song K, Zhang Y, Yu H, et al. Code-Switching for Enhancing NMT with Pre-Specified Translation[C]//Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019: 449-459.

[29] Vaswani A, Shazeer N, Parmar N, et al. Attention is All you Need[C]//NIPS. 2017.

[30] Wang T, Kuang S, Xiong D, et al. Merging external bilingual pairs into neural machine translation[J]. arXiv preprint arXiv:1912.00567, 2019.
