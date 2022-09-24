import torch
# from assess_baseline import RougeN
import random
from fairseq import token_generation_constraints
from search import LexicallyConstrainedBeamSearch
import math
import datetime
from statistics import mean
import time

class BeamSearch_decode:
    def __init__(self,beam_width,model,DEVICE="cuda",MAX_LEN=35):
        self.MAX_LEN=MAX_LEN
        self.beam_width=beam_width
        self.model=model
        self.DEVICE=DEVICE

    class Node:
        def __init__(self,previous_node,index,prob):
            '''
            size:batch*1
            :param previous_node:
            :param index:
            :param prob:
            '''
            self.pre=previous_node
            self.index=index
            self.prob=prob

    def accum_list(self,node):
        # node_list=[node]
        pre=node.pre
        prob=node.prob
        while pre:
            # node_list.append(pre)
            prob *= pre.prob
            pre=pre.pre
        return node,prob

    def find_opt(self,node):
        node_list=[node.index.item()]
        pre=node.pre
        while pre and pre.index:
            node_list.append(pre.index.item())
            pre=pre.pre
        node_list.reverse()
        return node_list

    def find_ipt(self,ipt,node):
        input=node.index
        pre=node.pre
        while pre and pre.index:
            input=torch.cat((pre.index,input),dim=-1)
            pre=pre.pre
        return torch.cat((ipt,input.reshape(1,len(input))),dim=-1)

    def BeamSearch(self,sent):
        beam_width=self.beam_width
        model=self.model

        ipt = torch.full((sent.size(0),1),3).to(self.DEVICE)
        root = self.Node(None, None, 1)
        beam_results = {}

        for pos in range(self.MAX_LEN):
            if len(beam_results) == 0:
                for i in range(beam_width):
                    beam_results[i] = root
                output = model(sent, ipt, self.DEVICE)
                values, indices = output.topk(beam_width, dim=-1, largest=True, sorted=True)
                for i in range(beam_width):
                    beam_results[i] = self.Node(beam_results[i], indices[:, -1, i], values[:, -1, i])
            else:
                nodes_waiting = []
                for i in range(beam_width):
                    if beam_results[i].index == 2:#2:[EOS]
                        nodes_waiting.append(self.accum_list(beam_results[i]))
                        continue
                    input = self.find_ipt(ipt, beam_results[i])
                    output = model(sent, input,self.DEVICE)
                    values, indices = output.topk(beam_width, dim=-1, largest=True, sorted=True)
                    for j in range(beam_width):
                        nodes_waiting.append(self.accum_list(self.Node(beam_results[i], indices[:, -1, j], values[:, -1, j])))
                # print(nodes_waiting)
                nodes_waiting.sort(key=lambda x: -x[1])
                for i in range(beam_width):
                    # if beam_results[i].index == 2:
                    #     continue
                    beam_results[i] = nodes_waiting[i][0]


        return beam_results

# class constraint_gen:
#     '''
#     #!/usr/bin/env python3
#     #
#     # Copyright (c) Facebook, Inc. and its affiliates.
#     #
#     # This source code is licensed under the MIT license found in the
#     # LICENSE file in the root directory of this source tree.
#     '''
#     def __init__(self):
#         pass

def constraint_gen(tgt,stopwords,constraint_number=3,phrase_maxlen=2,seed=1111,pad_id=0,eos_id=2,unk_id=1):
    '''
    generate up to three constraints per headline
    min(3,int(hdl_len*0.25))

    each constraint contains up to 2 words

    :param tgt: batch_size x hdl_maxlen:35
    :param seed:
    :return: packed constraints
    unpacked constraint like:
    [ [ [3 1 2], [3], [4 5 6 7], ]
          [],
          [ [1 8 9 10 1 4 11 12], ]
        ]
    '''
    if seed:
        random.seed(seed)
    time=datetime.datetime.now()
    batch_constraints_=[]
    for line in tgt:
        constraints = []

        def add_constraint(constraint):
            constraints.append(constraint)

        # source = line.rstrip()
        # if "\t" in line:
        #     source, target = line.split("\t")
        #     if args.add_sos:
        #         target = f"<s> {target}"
        #     if args.add_eos:
        #         target = f"{target} </s>"

        if len(line) >= phrase_maxlen:
            words = [x.item() for x in line if x!=pad_id and x!=eos_id and x!=unk_id]#remove pad and EOS

            num = min(constraint_number,int(len(words)*0.25))

            choices = {}
            for i in range(num):
                phrase_len=random.choice(range(phrase_maxlen))+1
                # if len(words) == 0:
                #     break

                range_list=list(range(len(words)))
                if i>0:
                    for index_chosen in choices.keys():
                        for x in range(index_chosen,index_chosen+len(choices[index_chosen])):
                            try:
                                range_list.remove(x)
                            except:
                                pass
                        if phrase_len>1:
                            for y in range(index_chosen-1,max(index_chosen-phrase_len,-1),-1):
                                try:
                                    range_list.remove(y)
                                except:
                                    pass
                if phrase_len==1:
                    # when we generate constraint and the constraint is a word, remove stop word
                    for z in words:
                        if z in stopwords:
                            try:
                                range_list.remove(words.index(z))
                            except:
                                pass

                if len(range_list)==0:
                    # sta_len[i]+=1
                    break
                phrase_index = random.choice(range_list)
                choice = torch.tensor(words[phrase_index: min(len(words), phrase_index + phrase_len)])

                # for j in range(min(len(words), phrase_index + phrase_len)-1,phrase_index-1,-1):
                #     words.pop(j)
                # if phrase_index > 0:
                #     words.append(" ".join(tokens[0:phrase_index]))
                # if phrase_index + 1 < len(tokens):
                #     words.append(" ".join(tokens[phrase_index:]))
                choices[phrase_index] = choice


                # # mask out with spaces
                # target = line.replace(choice, " " * len(choice), 1)
            # sta_len[len(choices)]+=1
            for key in sorted(choices.keys()):
                add_constraint(choices[key])
        batch_constraints_.append(constraints)
    batch_constraints=token_generation_constraints.pack_constraints(batch_constraints_)
    constime=(datetime.datetime.now()-time).seconds
    return batch_constraints,constime,batch_constraints_

        # print( *constraints, sep="\t")
# tgt=torch.tensor([[1,15,70,29,15,3,0,0,0,0],[15,70,29,15,1,15,70,29,15,23]])
# constraint_gen(tgt)

class dict_obj:
    def __init__(self,dic):
        # self.eos=dic['[EOS]']
        # self.unk=dic['[UNK]']
        # self.pad=dic['[PAD]']
        self.dic=dic

    def pad(self):
        return self.dic['[PAD]']

    def eos(self):
        return self.dic['[EOS]']

    def unk(self):
        return self.dic['[UNK]']

    def __len__(self):
        return len(self.dic)

class DBA:
    def __init__(self,word2idx,model,DEVICE="cuda",MAX_LEN=35,constraint_represent="unordered"):
        self.tgt_dic=dict_obj(word2idx)
        self.constraint_DBA=LexicallyConstrainedBeamSearch(self.tgt_dic,constraint_represent)
        self.model=model
        self.MAX_LEN=MAX_LEN
        self.DEVICE=DEVICE
        from nltk.corpus import stopwords
        stoplist = stopwords.words('english')
        for x in ["'s", "n't", ",", ".", "?", "!", "-", "&", ":", ";", "<", ">", "{", "}", "+", "^", "[", "]", "@", "#",
                  "(", ")", "|", "~","''",'``']:
            stoplist.append(x)
        self.stopword = [word2idx[x] for x in stoplist if x in word2idx.keys()]




    def dba(self,sent,hdl_opt,beam_size,cons_num=3,forcons=False):
        model=self.model
        batch_constraints,constime,batch_constraints_=constraint_gen(hdl_opt,self.stopword,constraint_number=cons_num)
        ipt = torch.full((sent.size(0) * beam_size, 1), 3).to(self.DEVICE)
        if not forcons:
            self.constraint_DBA.init_constraints(batch_constraints,beam_size)

            data=[]

            finished_beam=torch.zeros(sent.size(0),beam_size).to(self.DEVICE)
            # finished_hdl=finished_beam.contiguous().view(-1,1)
            ones_=torch.ones(1,len(self.tgt_dic)).to(self.DEVICE)
            for sentence in sent:
                s=[x.item() for x in sentence]
                for _ in range(beam_size):
                    data.append(s)
            sent_= torch.tensor(data).to(self.DEVICE)
            scores=None

            for pos in range(self.MAX_LEN):
                output=torch.log(model(sent_,ipt,self.DEVICE))
                # output: bts*beam_size,step+1
                # if pos==0:
                #     lprobs=output
                #     for _ in range(beam_size-1):
                #         lprobs = torch.cat((lprobs,output),dim=1)
                # else:
                #     pass
                lprobs=output[:, -1,:].clone()
                lprobs[:,self.tgt_dic.pad()] = -math.inf# never select pad
                # lprobs[:, self.tgt_dic.unk] -= self.unk_penalty  # apply unk penalty
                if pos ==0:
                    lprobs[:, self.tgt_dic.eos()] = -math.inf
                # when the hypo is finished, the following tokens will be anyone, which log-probs would be 0
                if_finish = finished_beam.contiguous().view(-1, 1).mm(ones_)
                lprobs = torch.where(if_finish > 0, torch.zeros_like(lprobs).to(self.DEVICE), lprobs)

                lprobs=lprobs.contiguous().view(sent.size(0),-1,len(self.tgt_dic))

                if pos==0:
                    score,indices,beam=self.constraint_DBA.step(pos,lprobs,None)
                else:
                    score,indices,beam=self.constraint_DBA.step(pos,lprobs,scores)

                # and find unreasonable hypo:
                # other step: phrase constraint is not consecutive ---deleted
                # eos_mask=torch.full([indices.size(0),indices.size(1)],-math.inf)

                # activate hypo to update constraint state and truncate the beam from 2*beam_size to beam_size
                active_hypo=torch.tensor([range(beam_size) for _ in range(sent.size(0))])
                self.constraint_DBA.update_constraints(active_hypo)

                # handle finished hypos
                index_=indices[:,:beam_size]
                finished_beam = torch.gather(finished_beam, dim=1, index=beam[:, :beam_size])
                if pos>1 and torch.sum(torch.where(index_ == self.tgt_dic.eos(), 1, 0).to(self.DEVICE))>1:
                    finished = torch.where(index_ == self.tgt_dic.eos(), 1, 0).to(self.DEVICE)
                    finished_beam = finished_beam+finished

                # handle finished headline: all items in beam are finished

                # generate ipt (the decode hypo) and scores (cumulative scores for hypos)
                hypo_id_pre=beam[:,:beam_size].reshape(-1,1)
                hypo_id=torch.tensor([int(x/beam_size)*beam_size for x in range(hypo_id_pre.size(0))]).to(self.DEVICE).reshape(-1,1)
                hypo_id=hypo_id_pre+hypo_id
                hypo_id=hypo_id.repeat(1,pos+1)

                # hypo_id=hypo_ids.clone()
                # for _ in range(pos):
                #     hypo_id=torch.cat((hypo_id,hypo_ids),dim=-1)
                ipt=torch.gather(ipt,dim=0,index=hypo_id)

                input=indices[:,:beam_size].reshape(-1,1)

                ipt=torch.cat((ipt,input),dim=-1)
                score=score[:,:beam_size].reshape(score.size(0),beam_size,1)
                if pos>0:
                    scores = torch.gather(scores, dim=1, index=beam[:, :beam_size].reshape(score.size(0),beam_size,1).repeat(1,1,pos))
                    scores=torch.cat((scores,score),dim=-1)
                else:
                    scores=score

        return batch_constraints,ipt,constime,batch_constraints_






