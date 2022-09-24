import torch
import logging
import argparse
import math
import csv

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from logging import handlers
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from CNN_NYT.DataLoad import BaseDataset
from model_tranformer import TransformerModel

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

parser = argparse.ArgumentParser(description='PyTorch Transformer for Neural Headline Generation')
parser.add_argument('--atc_data', type=str, default='./CNN_NYT/raw/data.abstracts',
                    help='location of the abstracts/articles data corpus')
parser.add_argument('--hdl_data', type=str, default='./CNN_NYT/raw/data.headlines',
                    help='location of the headlines data corpus')
# parser.add_argument('--model', type=str, default='transformer',
#                     help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--atc_testdata', type=str, default='./CNN_NYT/raw/test.abstracts',
                    help='location of the abstracts/articles test corpus)')
parser.add_argument('--hdl_testdata', type=str, default='./CNN_NYT/raw/test.headlines',
                    help='location of the headlines test corpus)')
parser.add_argument('--log_file', type=str, default='./result/log_file_512_6_min1.log',
                    help='path to save the log')
parser.add_argument('--max_len', type=int, default=500,#500
                    help='maximum length of atc sequence')
parser.add_argument('--min_occurance', type=int, default=1,
                    help='minimum time for the corpus`s words occurance')
parser.add_argument('--max_headline_len', type=int, default=35,#35
                    help='maximum length of hdl sequence')
parser.add_argument('--train_size',type=float, default=0.8,
                    help='the size of train_data, a ratio to divide data into train_data and eval_data')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')

parser.add_argument('--embedding_dim', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--n_head', type=int, default=8,#4
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--num_encoder', type=int, default=6,#
                    help='number of encoder layers')
parser.add_argument('--num_decoder', type=int, default=6,
                    help='number of decoder layers')
parser.add_argument('--dim_feed', type=int, default=512,#2048,512
                    help='size of feedforward layer')

parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout value')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--beam_width', type=int, default=3,
                    help='the width of beam search')
# parser.add_argument('--tied', action='store_true',
#                     help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
# parser.add_argument('--log-interval', type=int, default=200, metavar='N',
#                     help='report interval')

parser.add_argument('--save', type=str, default='./result/best_baseline_withpunc_{}.pt',
                    help='path to save the final model')
parser.add_argument('--save_result', type=str, default='./result/test_headlines.txt',
                    help='path to save the final model')
# parser.add_argument('--onnx-export', type=str, default='',
#                     help='path to export the final model in onnx format')

parser.add_argument('--if_train', type=int, default=1,
                    help='if train? input 1 to train and evaluate the model, input 0 to evaluate the best model saved directly')
# parser.add_argument('--dry-run', action='store_true',
#                     help='verify the code and the model')

args = parser.parse_args()
import random
random.seed(args.seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = init_logger(filename=args.log_file)

try:
    dataset = torch.load('./CNN_NYT/train_min1.pt')
except:
    dataset = BaseDataset(args.atc_data, args.hdl_data, args.max_len, args.min_occurance, args.max_headline_len)
    torch.save(dataset, './CNN_NYT/train_min1.pt')

train_size = int(args.train_size * len(dataset))
train_data, dev_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_data, batch_size =args.batch_size, shuffle=True)

global model
model = TransformerModel(len(dataset.word2idx), args.embedding_dim, args.n_head, args.num_encoder, args.num_decoder, args.dim_feed,args.dropout)
# model=nn.DataParallel(model,device_ids=[0,1,2,3,4,5,6,7])
model.to(DEVICE)

criterion = nn.NLLLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.8)

try:
    test_dataset = torch.load('./CNN_NYT/test_min1.pt')
except:
    test_dataset = BaseDataset(args.atc_testdata, args.hdl_testdata, args.max_len, args.min_occurance, args.max_headline_len,word2idx=dataset.word2idx)
    torch.save(test_dataset, './CNN_NYT/test_min1.pt')
test_dataset = DataLoader(test_dataset, batch_size=1)
# def loss_f(opt,tgt):
#     '''
#     :param opt: batch*seq_len*len(word2idx)
#     :param tgt: batch*seq_len
#     :return:
#     '''
#     nll=[]
#     for b_n in range(opt.size(0)):
#         loss=criterion(opt[b_n,:,:],tgt[b_n,:])
#         nll.append(loss)
#     return sum(nll) / opt.size(0)

def evaluate(dataloader):
    model.eval()
    losses= []
    ii=10
    with torch.no_grad():
        for batch in dataloader:
            idx, sent, hdl_ipt, hdl_opt = map(lambda x: x.to(DEVICE), batch)

            output = model(sent, hdl_ipt)
            output = torch.log(output)

            output = output.contiguous().view(-1, len(dataset.word2idx))
            hdl_opt = hdl_opt.contiguous().view(-1)
            loss = criterion(output, hdl_opt)
            if ii>0:
                output = torch.argmax(output, dim=-1)
                output = [idx2word[x.item()] for x in output]
                output = ' '.join(output).replace('[EOS]','')
                logger.info(output)
                ii-=1
            losses.append(loss.item())

    return losses

def load_network(network,save_path):
    state_dict=torch.load(save_path)

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        namekey = k[7:]
        new_state_dict[namekey]=v

    #load params
    network.load_state_dict(new_state_dict)
    return network

global best_val_loss
best_val_loss = None
idx2word={v:k for k,v in dataset.word2idx.items()}
def train():
    global best_val_loss
    global model

    for epoch in range(args.epochs):
        model.train()
        logger.info('=' * 100)
        losses, accs = [], []
        pbar = tqdm(total=len(train_dataloader))
        ii = 10
        for batch in train_dataloader:
            idx, sent, hdl_ipt, hdl_opt = map(lambda x: x.to(DEVICE), batch)
            optimizer.zero_grad()
            # if torch.cuda.device_count()>1:
            #     output = nn.parallel.data_parallel(model,(sent, hdl_ipt))
            # else:
            output = model(sent,hdl_ipt)
            output = output.contiguous().view(-1, len(dataset.word2idx))
            hdl_opt=hdl_opt.contiguous().view(-1)

            loss = criterion(output, hdl_opt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            losses.append(loss.item())

            pbar.set_description('Epoch: %2d | Loss: %.3f ' % (epoch, loss.item()))
            if ii>0:
                output = torch.argmax(output, dim=-1)
                output = [idx2word[x.item()] for x in output]
                output = ' '.join(output).replace('[EOS]','')
                logger.info(output)
                ii-=1
            # output = torch.argmax(output, dim=-1)
            # output = [idx2word[x.item()] for x in  output]
            # output = ' '.join(output).replace('[EOS] ','')
            # logger.info(output)
            pbar.update(1)

        pbar.close()

        # logger.info log
        dev_loss = evaluate(dev_dataloader)
        logger.info('Training:\t  Loss: %.3f' % (np.mean(losses)))
        logger.info('Evaluating:\t  Loss: %.3f' % (np.mean(dev_loss)))
        logger.info('')
        scheduler.step()

        # if not best_val_loss or np.mean(dev_loss) < best_val_loss:
        file_saved=args.save+'tt_min1_'+str(round(np.mean(dev_loss),5))+'_'+str(round(np.mean(losses),5))+'_'+str(36+epoch)+'_.pt'
        with open(file_saved, 'wb') as f:
            # if torch.cuda.device_count() > 1:
            #     torch.save(model.module.state_dict(), f)
            # else:
            torch.save(model.state_dict(), f)
        # best_val_loss = np.mean(dev_loss)
    # f = open(file_saved, 'rb')
    # # if torch.cuda.device_count()>1:
    # #     model=load_network(model,f)
    # # else:
    # model.load_state_dict(torch.load(f))


def write_opt(output, dataset):
    idx2word = {v: k for k, v in dataset.word2idx.items()}
    output = torch.argmax(output, dim=-1)
    output = [[idx2word[x.item()] for x in sent] for sent in output]
    output = [' '.join(sent) for sent in output]
    output = [
        sent[0:sent.find('[EOS]', 1)].replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]',
                                                                                                               '') + '\n'
        for sent in output]



#     print(output)
#     result.writelines(output)
#
# model.eval()
# losses= []
def test(e, dataset, test_dataset):
    test_dataset = DataLoader(test_dataset, batch_size=32)
    import datetime
    time = datetime.datetime.now()
    print('----greedy-', datetime.datetime.now(), '-----')
    with torch.no_grad():
        for batch in test_dataset:
            idx, sent, hdl_ipt, hdl_opt = map(lambda x: x.to(DEVICE), batch)
            ipt = hdl_ipt[:, 0].unsqueeze(1)

            for pos in range(35):
                output = model(sent, ipt)
                input=(torch.argmax(output,dim=-1))[:,-1].unsqueeze(1)
                ipt=torch.cat((ipt,input),dim=-1)

            # output = torch.argmax(output, dim=-1)

            # ipt = hdl_ipt[:, 0].unsqueeze(1)
            #
            #         for _ in range(args.max_headline_len):
            #             output = model(sent, ipt)
            #             ipt = torch.cat((ipt, torch.argmax(output, dim=-1)[:, -1].unsqueeze(1)), dim=-1)
            #
            #         # output = model(sent, hdl_ipt)
            write_opt(output, dataset)
    print(e, 'time:', (datetime.datetime.now() - time).seconds)
if args.if_train==0:
    # try:
        for e in ['256_6__min1_5.72525_5.22809_49_','256_6__min1_5.37337_5.18472_61_','256_6__min1_5.35166_5.37699_50_']:#'4.2126_4.7437_56','4.42614_4.62997_27','4.32073_5.55574_25','4.96901_35','5.14739_62','_min1_5.16003_5.02806_36_','_min1_5.67872_4.82787_35','_min1_5.71646_4.71497_49'
            f=open(args.save.format(e), 'rb')
            if e.find('min1',1)>0:
                dataset = torch.load('./CNN_NYT/train_min1.pt')
                test_dataset=torch.load('./CNN_NYT/test_min1.pt')
            else:
                dataset = torch.load('./CNN_NYT/train.pt')
                test_dataset = torch.load('./CNN_NYT/test.pt')
            model = TransformerModel(len(dataset.word2idx), args.embedding_dim, args.n_head, args.num_encoder,
                                     args.num_decoder, args.dim_feed, args.dropout)
            model.to(DEVICE)
            model.load_state_dict(torch.load(f))
            test(e,dataset,test_dataset)
    #
    #
    # except:
    #     train()
else:
    model.load_state_dict(torch.load(open('./result/best_baseline_withpunc_512_6__min1_5.67872_4.82787_35.pt','rb')))
    # model=nn.DataParallel(model,device_ids=[0,1,2,3])
    # model.to(DEVICE)
    # model.cuda()
    train()




# generate headline of test_dataset save result in save_result dir
# result = open(args.save_result, 'w', newline='')
# result = open('./result/headlines_withpunc.txt', 'w', newline='')

# def write_opt(output,dataset):
#     idx2word={v:k for k,v in dataset.word2idx.items()}
#     output = torch.argmax(output, dim=-1)
#     output = [[idx2word[x.item()] for x in sent] for sent in output]
#     output = [' '.join(sent) for sent in output]
#     output = [sent[0:sent.find('[EOS]', 1)].replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n' for sent in output]
# #     print(output)
#     result.writelines(output)
#
# model.eval()
# losses= []
# def test(e,dataset,test_dataset):
#     import datetime
#     time = datetime.datetime.now()
#     print('----greedy-', datetime.datetime.now(), '-----')
#     with torch.no_grad():
#         for batch in test_dataset:
#             idx, sent, hdl_ipt, hdl_opt = map(lambda x: x.to(DEVICE), batch)
#             output = model(sent, hdl_ipt)
#             output=torch.argmax(output,dim=-1)
#
            # ipt = hdl_ipt[:, 0].unsqueeze(1)
    #
    #         for _ in range(args.max_headline_len):
    #             output = model(sent, ipt)
    #             ipt = torch.cat((ipt, torch.argmax(output, dim=-1)[:, -1].unsqueeze(1)), dim=-1)
    #
    #         # output = model(sent, hdl_ipt)
    #         write_opt(output,dataset)
    # print(e,'time:',(datetime.datetime.now()-time).seconds)
#         output = output.contiguous().view(-1, len(dataset.word2idx))
#         hdl_opt = hdl_opt.contiguous().view(-1)
#         loss =criterion(output , hdl_opt)
#         losses.append(loss.item())

# def write_opt(output):
#     idx2word={v:k for k,v in dataset.word2idx.items()}
#     # output = torch.argmax(output, dim=-1)
#     # output = [[idx2word[x.item()] for x in sent] for sent in output]
#     output=[idx2word[i] for i in output]
#     # output = [' '.join(sent) for sent in output]
#     output=' '.join(output)
#     output = output[0:output.find('[EOS]', 1)].replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n'
#     # output = [sent[0:sent.find('[EOS]', 1)].replace(' [PAD]', '').replace('[STR] ', '').replace('[STR]', '').replace(' [EOS]', '')+'\n' for sent in output]
#     print(output)
#     result.writelines(output)
#
# beam_width=args.beam_width
# class Node:
#     def __init__(self,previous_node,index,prob):
#         '''
#         size:batch*1
#         :param previous_node:
#         :param index:
#         :param prob:
#         '''
#         self.pre=previous_node
#         self.index=index
#         self.prob=prob
#
# def accum_list(node):
#     # node_list=[node]
#     pre=node.pre
#     prob=node.prob
#     while pre:
#         # node_list.append(pre)
#         prob *= pre.prob
#         pre=pre.pre
#     return node,prob
#
# def find_opt(node):
#     node_list=[node.index.item()]
#     pre=node.pre
#     while pre and pre.index:
#         node_list.append(pre.index.item())
#         pre=pre.pre
#     node_list.reverse()
#     return node_list
#
# def find_ipt(ipt,node):
#     input=node.index
#     pre=node.pre
#     while pre and pre.index:
#         input=torch.cat((pre.index,input),dim=-1)
#         pre=pre.pre
#     return torch.cat((ipt,input.reshape(1,len(input))),dim=-1)
#
# with torch.no_grad():
#     for batch in test_dataset:
#         model.eval()
#         idx, sent, hdl_ipt, hdl_opt = map(lambda x: x.to('cuda'), batch)
#         # src_embed = model.embedding(sent)*math.sqrt(model.d_model)
#         # src_mark = model._generate_square_subsequent_mask
#         # enc_opt = model.trans.encoder()
#         ipt = hdl_ipt[:, 0].unsqueeze(1)
#         root=Node(None,None,1)
#         beam_results= {}
#         # beam_probs=[]
#
#             # beam_results[i].append(root)
#
#         for pos in range(35):
#             if len(beam_results) == 0:
#                 for i in range(beam_width):
#                     beam_results[i] = root
#                 output = model(sent, ipt)
#                 values, indices = output.topk(beam_width, dim=-1, largest=True, sorted=True)
#                 for i in range(beam_width):
#                     beam_results[i]=Node(beam_results[i],indices[:,-1,i],values[:,-1,i])
#             else:
#                 nodes_waiting=[]
#                 for i in range(beam_width):
#                     if beam_results[i].index==2:
#                         nodes_waiting.append(accum_list(beam_results[i]))
#                         continue
#                     input=find_ipt(ipt,beam_results[i])
#                     output=model(sent,input)
#                     values, indices = output.topk(beam_width, dim=-1, largest=True, sorted=True)
#                     for j in range(beam_width):
#                         nodes_waiting.append(accum_list(Node(beam_results[i],indices[:,-1,j],values[:,-1,j])))
#                 # print(nodes_waiting)
#                 nodes_waiting.sort(key=lambda x:-x[1])
#                 for i in range(beam_width):
#                     # if beam_results[i].index == 2:
#                     #     continue
#                     beam_results[i] = nodes_waiting[i][0]
#         #选择beamsearch中概率最大的
#         indices=beam_results[i]
#         output=find_opt(indices)
#         write_opt(output)
#
# logger.info('headlines for test_data saved.{}'.format(args.save_result))
# logger.info('testing_loss:\t  Loss: %.3f' % (np.mean(losses)))
logger.info('')

