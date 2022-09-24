import torch
import logging
from logging import handlers
import numpy as np
import math
import datetime
from statistics import mean

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger

# def _generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask
#
# # print(_generate_square_subsequent_mask(3))
#
#
# def get_attn_subsequence_mask(seq):
#     '''
#     seq: [batch_size, tgt_len]
#     '''
#     attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
#     subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
#     subsequence_mask = torch.from_numpy(subsequence_mask).byte()
#     return subsequence_mask
# # print(get_attn_subsequence_mask(torch.randn((3,5))))
#
#
# def get_attn_pad_mask(seq_q, seq_k):
#     '''
#     seq_q: [batch_size, seq_len]
#     seq_k: [batch_size, seq_len]
#     seq_len could be src_len or it could be tgt_len
#     seq_len in seq_q and seq_len in seq_k maybe not equal
#     '''
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     # eq(zero) is PAD token
#     pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
#     return pad_attn_mask.expand(batch_size, len_q, len_k)
# a=torch.tensor([[1,1,0],[1,0,0]])
# b=torch.tensor([[1,0,0,0],[1,1,0,0]])
# # print((get_attn_pad_mask(a,a)))
#
#
# def padding_mask( seq_k, seq_q):
#     '''
#     :param seq_k: batch_size*key_seq_len
#     :param seq_q: batch_size*query_seq_len
#     :return:batch_size*query_seq_len*key_seq_len
#     '''
#     # seq_k和seq_q的形状都是[B,L]
#     len_q = seq_q.size(1)
#     # `PAD` is 0
#     pad_mask = seq_k.eq(0)
#     print(pad_mask)
#     pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
#     return pad_mask
# # pad_mask=padding_mask(a,a)
# # print(pad_mask)
# # print(pad_mask.size())
# #
# # pad_mask=padding_mask(a,b)
# # print(pad_mask)
# # print(pad_mask.size())
#
# # def sequence_mask(seq):
# #     batch_size, seq_len = seq.size()
# #     mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
# #                     diagonal=1)
# #     mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
# #     return mask
# # # print(sequence_mask(a))
# # a=[[[1,2,4,0],[2,3,4,0]],[[1,2,4,0],[2,3,4,0]],[[1,2,4,0],[2,3,4,0]],[[1,2,4,0],[2,3,4,0]],[[1,2,4,0],[2,3,4,0]]]
# # a=torch.tensor(a)
# # print(a.shape)
#
# # dic={0:'[PAD]',1:'[STR]',2:'b',3:'c',4:'[EOS]'}
# # a=[[dic[x.item()] for x in sent]for sent in a]
# # a=[''.join(sent).replace('[PAD]','').replace('[STR]','').replace('[EOS]','') for sent in a]
# # print(a)
#
# # aa=map(lambda x:print(x),a)
# # print(a)
# # b=[1,2]
# #
# # print(sum(b)/2)
#
# # result = open('test.txt', 'w', newline='')
#
# # def write_opt(output):
# #     # output = torch.argmax(output, dim=-1)
# #     output = [[dic[x.item()] for x in sent]for sent in output]
# #     output = [' '.join(sent).replace(' [PAD]','').replace('[STR] ','').replace(' [EOS]','')+'\n' for sent in output]
# #     result.writelines(output)
# # write_opt(a)
# # print(a.shape)
# # a=a.view(-1, 4)
# # print(a.shape)
# # url = "i love you forever"
# # aaa = "i do n't love you"
# # print (url[0:url.rfind('.jpg', 1) ])
# # test='...'
# # punct='!.?><}{,:;"/-=+~`^&*#[]|()@\''
# # if test in punct:
# #     print('...ok delete')
import torch.nn as nn
# criterion = nn.NLLLoss(ignore_index=0)
from model_tranformer import TransformerModel
from torch.utils.data import DataLoader
from decode_test import BeamSearch_decode
from decode_test import DBA
from rouge import FilesRouge

logger = init_logger(filename='rouge_assess.log')
def write_opt(output,result):
    idx2word={v:k for k,v in dataset.word2idx.items()}
    # output = torch.argmax(output, dim=-1)
    # output = [[idx2word[x.item()] for x in sent] for sent in output]
    output=[idx2word[i] for i in output]
    # output = [' '.join(sent) for sent in output]
    output=' '.join(output)
    if output.find('[EOS]', 1)>0:
        output = output[0:output.find('[EOS]', 1)].replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n'
    else:
        output =output.replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n'
        # output = [sent[0:sent.find('[EOS]', 1)].replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n' for sent in output]
    # print(output)
    result.writelines(output)

def find_all_nodes(item,idx2word):
    cons = ''
    for constraint_ in list(item.generated.elements()):
        cons+=idx2word[constraint_.id]
        if constraint_.terminal==1:
            cons+='; '
        else:
            cons+=' '
    return cons

def write_cons_dba(constraints,fullcons,constraint):
    idx2word = {v: k for k, v in dataset.word2idx.items()}
    constraints_w=[]
    for i in range(len(constraints)):
        cons=find_all_nodes(constraints[i],idx2word)
        num=constraints[i].num_completed
        num_full=constraints[i].root.num_constraints
        full=''
        from fairseq.token_generation_constraints import unpack_constraints
        fullcon = unpack_constraints(fullcons[i])
        for fulcon_item in fullcon:
            constraint_full=[idx2word[x.item()] for x in fulcon_item]
            constraint_full=' '.join(constraint_full)+'; '
            full+=constraint_full
        constraints_w.append('all constraint:/'+str(num_full)+':'+full+'\t'+'|completed '+str(num)+':'+cons+'\n')
    # print('------'+str(datetime.datetime.now())+'------')
    # print(constraints_w)
    constraint.writelines(constraints_w)
#
def write_opt_dba(output,fullcons,constraints,result,constraint):
    idx2word = {v: k for k, v in dataset.word2idx.items()}
    # output = torch.argmax(output, dim=-1)
    output = [[idx2word[x.item()] for x in sent] for sent in output]
    output = [' '.join(sent) for sent in output]
    output = [sent[0:sent.find('[EOS]', 1)].replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n' for sent in output]
    # fullcons = unpack_constraints(fullcons)
    write_cons_dba(constraints,fullcons,constraint)
    # print(output)
    result.writelines(output)

    # if output.find('[EOS]', 1)>0:
    #     output = output[0:output.find('[EOS]', 1)].replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n'
    # else:
    #     output =output.replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n'



files_rouge = FilesRouge()
true_file='./result/true_headlines.txt'
beam_width=3

# file_mode='./result/best_baseline_withpunc_{}.pt'
# file_items=['256_6__min1_5.72525_5.22809_49_','256_6__min1_5.37337_5.18472_61_','256_6__min1_5.35166_5.37699_50_']
# 512
file_mode='./result/best_baseline_withpunc_512_6_{}.pt'
file_items=['_min1_5.67872_4.82787_35']#'_min1_5.16003_5.02806_36_','_min1_5.71646_4.71497_49','4.96901_35','5.14739_62','4.2126_4.7437_56','4.42614_4.62997_27','4.32073_5.55574_25',
# ,'_min1_5.67872_4.82787_35','_min1_5.71646_4.71497_49''4.96901_35','5.14739_62','4.2126_4.7437_56','4.42614_4.62997_27','4.32073_5.55574_25','4.96901_35','5.14739_62','_min1_5.16003_5.02806_36_','_min1_5.67872_4.82787_35','_min1_5.71646_4.71497_49'
for file_name in file_items:
    if file_name.find('min1',1)==-1:
        dataset = torch.load('./CNN_NYT/train.pt')
        test_dataset_ori = torch.load('./CNN_NYT/test.pt')
    else:
        dataset = torch.load('./CNN_NYT/train_min1.pt')
        test_dataset_ori = torch.load('./CNN_NYT/test_min1.pt')

    logger.info(file_mode.format(file_name))
    f = open(file_mode.format(file_name), 'rb')
    model = TransformerModel(len(dataset.word2idx), 512, 8, 6, 6, 512)
    DEVICE = "cuda:0"
    model.to(DEVICE)
    model.load_state_dict(torch.load(f))


    # -------------beam search-------------------
    # result_file='./result/test/test_{}_{}.txt'.format(beam_width, file_name)
    # result_bs = open(result_file, 'w', newline='')
    # test_dataset = DataLoader(test_dataset_ori, batch_size=1)
    #
    # time = datetime.datetime.now()
    # logger.info('----beamsearch-'+str(datetime.datetime.now())+ '-----')
    # beam = BeamSearch_decode(beam_width, model, DEVICE=DEVICE)
    # with torch.no_grad():
    #     for batch in test_dataset:
    #         model.eval()
    #         idx, sent, hdl_ipt, hdl_opt = map(lambda x: x.to(DEVICE), batch)
    #         beam_results = beam.BeamSearch(sent)
    #         # src_embed = model.embedding(sent)*math.sqrt(model.d_model)
    #         # src_mark = model._generate_square_subsequent_mask
    #         # enc_opt = model.trans.encoder()
    #         # ipt = hdl_ipt[:, 0].unsqueeze(1)
    #         # root=Node(None,None,1)
    #         # beam_results= {}
    #         # # beam_probs=[]
    #         #
    #         #     # beam_results[i].append(root)
    #         #
    #         # for pos in range(35):
    #         #     if len(beam_results) == 0:
    #         #         for i in range(beam_width):
    #         #             beam_results[i] = root
    #         #         output = model(sent, ipt,"cuda:1")
    #         #         values, indices = output.topk(beam_width, dim=-1, largest=True, sorted=True)
    #         #         for i in range(beam_width):
    #         #             beam_results[i]=Node(beam_results[i],indices[:,-1,i],values[:,-1,i])
    #         #     else:
    #         #         nodes_waiting=[]
    #         #         for i in range(beam_width):
    #         #             if beam_results[i].index==2:
    #         #                 nodes_waiting.append(accum_list(beam_results[i]))
    #         #                 continue
    #         #             input=find_ipt(ipt,beam_results[i])
    #         #             output=model(sent,input,"cuda:1")
    #         #             values, indices = output.topk(beam_width, dim=-1, largest=True, sorted=True)
    #         #             for j in range(beam_width):
    #         #                 nodes_waiting.append(accum_list(Node(beam_results[i],indices[:,-1,j],values[:,-1,j])))
    #         #         # print(nodes_waiting)
    #         #         nodes_waiting.sort(key=lambda x:-x[1])
    #         #         for i in range(beam_width):
    #         #             # if beam_results[i].index == 2:
    #         #             #     continue
    #         #             beam_results[i] = nodes_waiting[i][0]
    #         # 选择beamsearch中概率最大的
    #         indices = beam_results[0]
    #         output = beam.find_opt(indices)
    #         write_opt(output,result_bs)
    # logger.info('----endbeamsearch-'+ str(datetime.datetime.now())+'-----')
    # logger.info(result_file+'time:'+str((datetime.datetime.now()-time).seconds))
    # result_bs.close()
    #
    # # evaluate beam search with ROUGE
    # score=files_rouge.get_scores(true_file, result_file, avg=True)
    # logger.info(str(score))
    # logger.info('')
    #
    # if score['rouge-l']['f']<0.13 or score['rouge-1']['f']<0.13:
    #     print('bad model')
    #     continue
    test_dataset = DataLoader(test_dataset_ori, batch_size=32)

    # -------------DBA-------------------
    for cons_num in range(1,4):
        result_file = './result/test/test_DBA1_cons{}_{}_{}.txt'.format(cons_num,beam_width, file_name)
        result_dba = open(result_file, 'w', newline='')
        constraint_file = './result/test/constraint_DBA1_cons{}_{}_{}.txt'.format(cons_num,beam_width,file_name)
        constraint_dba = open(constraint_file, 'w', newline='')


        time = datetime.datetime.now()
        logger.info('-----DBA-cons_up_num-' + str(cons_num)+ '---'+str(datetime.datetime.now()) + '-----')
        beam = DBA(dataset.word2idx, model, DEVICE)
        genconstime=0
        batch_constraint=[]
        with torch.no_grad():
            for batch in test_dataset:
                model.eval()
                idx, sent, hdl_ipt, hdl_opt = map(lambda x: x.to(DEVICE), batch)
                full_constraint, results,cons_time,batch_constraint_ = beam.dba(sent, hdl_opt, beam_size=beam_width,cons_num=cons_num)
                batch_constraint+=batch_constraint_
                genconstime+=cons_time
                results = results.reshape(sent.size(0), beam_width, -1)
                write_opt_dba(results[:, 0, 1:], full_constraint, [x[0] for x in beam.constraint_DBA.constraint_states],result_dba,constraint_dba)
        logger.info('-----endDBA-cons_up_num-' + str(cons_num)+ '---'+str(datetime.datetime.now()) + '-----')
        logger.info(result_file + 'time:' + str((datetime.datetime.now() - time).seconds-genconstime))
        cons_len=[len(x) for x in batch_constraint]
        cons_word_len=[len(z) for x in batch_constraint for z in x]

        logger.info('constraint statistic:--max:'+str(max(cons_len))+',min:'+str(min(cons_len))+',mean:'+str(mean(cons_len)))
        logger.info('constraint word statistic:--max:' + str(max(cons_word_len)) + ',min:' + str(min(cons_word_len)) + ',mean:' + str(
            mean(cons_word_len)))

        result_dba.close()
        constraint_dba.close()

        # evaluate DBA with ROUGE
        # logger.info(str(files_rouge.get_scores(true_file, result_file, avg=True)))
        logger.info('')