from torchtext.data.metrics import _compute_ngram_counter
import numpy as np
import math

class ROUGE_N:
    def __init__(self,src,tgt,file='TRUE'):
        self.file=file
        if file:
            src=open(src,'rb')
            tgt=open(tgt,'rb')
            self.gene_headlines=src.readlines()
            self.true_headlines=tgt.readlines()
        elif (isinstance(src,list) and isinstance(tgt,list) and len(src)==len(tgt)) or (isinstance(src,str) and isinstance(tgt,str)):
            self.gene_headlines = src
            self.true_headlines = tgt
        else:
            Exception('wrong')

    def RougeN(self,N,src,tgt):
        counter_src = _compute_ngram_counter(src.split(' '), 2)
        counter_tgt = _compute_ngram_counter(tgt.split(' '), 2)
        deno = [counter_tgt[word] for word in counter_tgt.keys() if len(word)==N]
        nume = [counter_src[word] for word in counter_tgt.keys() if counter_src[word] and len(word)==N]

        recall=sum(nume)/(sum(deno)+ 1e-8)
        precision=sum(nume)/len(src.split(' '))
        f_measure=2*recall*precision/(recall+precision+ 1e-8)
        return {'f':f_measure,'p':precision,'r':recall}

    def rouge12l(self,avg=False):
        accs_1=[]
        accs_2=[]
        accs_l=[]
        if isinstance(self.true_headlines,list):
            for true, pred in zip(self.true_headlines, self.gene_headlines):
                try:
                    true=true.decode()
                    pred=pred.decode()
                except:
                    pass
                accs_1.append(self.RougeN(1, pred.replace('\n', ''), true.replace('\n', '')))
                accs_2.append(self.RougeN(2, pred.replace('\n', ''), true.replace('\n', '')))
                accs_l.append(self.RougeL(pred.replace('\n', ''), true.replace('\n', '')))
            if avg:
                a=np.array([[x['f'], x['p'],x['r']] for x in accs_1])
                b=np.array([[x['f'], x['p'],x['r']] for x in accs_2])
                c=np.array([[x['f'], x['p'],x['r']] for x in accs_l])
                return {'ROUGE-1':{'f':np.mean(a[:,0]),'p':np.mean(a[:,1]),'r':np.mean(a[:,2])},'ROUGE-2':{'f':np.mean(b[:,0]),'p':np.mean(b[:,1]),'r':np.mean(b[:,2])},'ROUGE-L':{'f':np.mean(c[:,0]),'p':np.mean(c[:,1]),'r':np.mean(c[:,2])},}
            else:
                return {'ROUGE-1':accs_1,'ROUGE-2':accs_2,'ROUGE-L':accs_l}
        else:
            pred=self.gene_headlines
            true=self.true_headlines
            accs_1.append(self.RougeN(1, pred.replace('\n', ''), true.replace('\n', '')))
            accs_2.append(self.RougeN(2, pred.replace('\n', ''), true.replace('\n', '')))
            accs_l.append(self.RougeL(pred.replace('\n', ''), true.replace('\n', '')))
            return {'ROUGE-1': accs_1, 'ROUGE-2': accs_2, 'ROUGE-L': accs_l}

    def RougeL(self,src,tgt):
        src=src.split(' ')
        tgt=tgt.split(' ')
        lstr1 = len(src)
        lstr2 = len(tgt)
        # record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
        maxNum = 0  # 最长匹配长度
        # s = 0  # 匹配的起始位
        # for i in range(lstr1):
        #     s = 0
        #     for j in range(lstr2):
        #         if src[i] == tgt[j]:
        #             record[i + 1][j + 1] = record[i][j] + 1
        #             if record[i + 1][j + 1] > maxNum:
        #                 maxNum = record[i + 1][j + 1]
        #                 s = i + 1
        while len(src) > 0:
            thiskey = src[0]
            if len(tgt) <= 0:
                break
            if thiskey in tgt:
                tgt = tgt[tgt.index(thiskey) + 1:]
                maxNum += 1
            src = src[1:]
        recall = maxNum / lstr2
        precision=maxNum / lstr1
        f_measure=2*recall*precision/(recall+precision+ 1e-8)
        return {'f':f_measure,'p':precision,'r':recall}
# t0="u.s. soldier 's family : ' i ' m good , in afghan ' "
# t1="soldier 's family : ' i ' m a good bergdahl ' "
# t2="bergdahl in afghanistan program aims to help ' honor ' "
# t3="bergdahl in afghanistan program aims to help ' safe landing ' "
# truehdl="army program aims to bring bergdahl in for a ' safe landing '"

t0='report : victims of abuse victims in congo'
t1='report : panel told victims of abuse victims in abuse'
t2='report : abuse victims of boys victims in congo '
t3='panel told of boys victims in salvation army '
truehdl='australian panel told of sexual abuse of boys at salvation army homes'
for srcn in [t0,t1,t2,t3]:
    assess=ROUGE_N(srcn,truehdl,False)
    print(assess.rouge12l())

import datetime
# file_items=['4.42614_4.62997_27']
# file_mode='./result/best_baseline_withpunc_{}.pt'
# file_items=['256_6__min1_5.72525_5.22809_49_','256_6__min1_5.37337_5.18472_61_','256_6__min1_5.35166_5.37699_50_']

# ---assess these files
file_items=['_min1_5.67872_4.82787_35','_min1_5.71646_4.71497_49','4.96901_35','5.14739_62','4.2126_4.7437_56',
            '4.42614_4.62997_27','4.32073_5.55574_25','_min1_5.16003_5.02806_36_']#'4.2126_4.7437_56','4.42614_4.62997_27','4.32073_5.55574_25','4.96901_35','5.14739_62',,'_min1_5.67872_4.82787_35','_min1_5.71646_4.71497_49'
beam_width=3
true_file='./result/true_headlines.txt'

for file_name in file_items:
    for i in [1]:
        if i==0:
            result_file = './result/test/test_{}_{}.txt'.format(beam_width, file_name)
            score = ROUGE_N(result_file, true_file, file=True)
            print(result_file,':\n',score.rouge12l(avg=True))
        else:
            for cons_num in range(1,4):
                result_file = './result/test/test_DBA1_cons{}_{}_{}.txt'.format(cons_num, beam_width, file_name)
                score = ROUGE_N(result_file, true_file, file=True)
                print(result_file, ':\n', score.rouge12l(avg=True))


# time=datetime.datetime.now()
# print('time:',(datetime.datetime.now()-time).seconds)


#
# accs_1=[]
# accs_2=[]
# gene_file='./result/test4.62997_27.txt'
# true_file='./result/true_headlines.txt'
#
# f1=open(true_file,'r')
# true_headlines=f1.readlines()
# # f2=open('./result/headlines_0330.txt','r')
# f2=open(gene_file,'r')
# gene_headlines=f2.readlines()
#
#
# # print(gene)
# print('ROUGE-1:',np.mean(accs_1),'ROUGE-2',np.mean(accs_2))
# print(sum(accs_1)/3000,sum(accs_2)/3000)
#
# # import files2rouge
# from rouge import FilesRouge,Rouge
# files_rouge = FilesRouge()
# # scores = files_rouge.get_scores(hyp_path, ref_path)
# # or
# scores = files_rouge.get_scores(gene_file, true_file,avg=True)
# print(gene_headlines[0],true_headlines[0])
# print(scores)

