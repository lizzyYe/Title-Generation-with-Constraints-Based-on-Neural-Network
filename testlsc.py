def RougeL(src, tgt):
    lstr1 = len(src)
    lstr2 = len(tgt)
    # record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    # s = 0  # 匹配的起始位
    # i=0
    # j=0
    while len(src)>0:
        thiskey=src[0]
        if len(tgt)<=0:
            break
        if thiskey in tgt:
            tgt=tgt[tgt.index(thiskey)+1:]
            maxNum+=1
        src=src[1:]

    # for i in range(lstr1):
    #     for j in range(lstr2):
    #         if src[i] == tgt[j]:
    #             record[i + 1][j + 1]  = (record[i][j] + 1)
    #             if record[i + 1][j + 1] > maxNum:
    #                 maxNum = record[i + 1][j + 1]
    #                 # if s>=lstr1:
    #                 #     break
    #                 # else:
    #                 #     i=s
    recall = maxNum / lstr2
    precision = maxNum / lstr1
    f_measure = 2 * recall * precision / (recall + precision + 1e-8)
    return {'f': f_measure, 'p': precision, 'r': recall}

tgt='police killed the gunman'
src='police ended the gunman'
src2='the gunman murdered police'
print(RougeL(src.split(' '),tgt.split(' ')))
print(RougeL(src2.split(' '),tgt.split(' ')))