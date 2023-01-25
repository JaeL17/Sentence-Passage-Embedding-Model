import json
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics.pairwise import cosine_similarity
import random


def create_pos_neg(data, model):                        
    pos, neg = [],[]
    for cnt, i in enumerate(data):
        sents = i['sentences']
        psg_pos = i['passage']
        for s in sents:
            s_vec = model.encode(s)
            ng_cand_lst = []
            rand_n = random.sample(range(0, len(data)),20)
            if cnt in rand_n:
                rand_n.remove(cnt)
                
            for rn in rand_n:
                max_cos_sim = 0
                psg = data[rn]['passage']
                psg_lst = psg.split(' [SEP] ')
                for p in psg_lst:
                    p_vec = model.encode(p)
                    cos_sim = cosine_similarity([s_vec, p_vec])[0][1]
                    if cos_sim > max_cos_sim:
                        max_cos_sim = cos_sim
                ng_cand_lst.append((max_cos_sim, psg))
            
            ng_sorted = sorted(ng_cand_lst)
            
            neg_dic = {'sentence':s,
                      'passage':ng_sorted[0][1],
                      'score': float(0.0),
                      'max_sbert_score':float(ng_sorted[0][0])}
            neg.append(neg_dic)
            
            pos_dic = {'sentence':s,
                      'passage':psg_pos,
                      'score': float(0.9)}
            pos.append(pos_dic)
        if cnt % 100 ==0:
            print('progress: ', cnt, ' / ', len(data))
    return pos,neg

if __name__ == "__main__":
    print('start!')
    sbert_iq = SentenceTransformer('../../imagef/model_results/klue_sts_general', device='cuda:0')
    print('model done!')
    with open('data_form1.json', 'r', encoding='utf8') as fp:
        data = json.load(fp)
    x_train, x_dev, y_train, y_dev = train_test_split(data, data, test_size =0.1, random_state=62)
    
#     print('create train............')
#     pos_train, neg_train  = create_pos_neg(x_train, sbert_iq)
#     data_f =[]
#     data_f.extend(pos_train)
#     data_f.extend(neg_train)
    
#     with open('train_final.json', 'w', encoding='utf8') as fp:
#         fp.write(json.dumps(data_f, ensure_ascii=False, indent=4))
        
    print('create dev............')    
    pos_dev, neg_dev  = create_pos_neg(x_dev, sbert_iq)
    data_f_dev =[]
    data_f_dev.extend(pos_dev)
    data_f_dev.extend(neg_dev)
    
    with open('dev_final2.json', 'w', encoding='utf8') as fp:
        fp.write(json.dumps(data_f_dev, ensure_ascii=False, indent=4))
    
    