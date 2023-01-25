from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import argparse
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from pororo import Pororo
from konlpy.tag import Okt
from krwordrank.word import KRWordRank
from random import sample
from sentence_transformers import InputExample
import numpy as np

def load_stsdataset():
        
    kor_sts_train = open('data/sts-train.tsv.txt', "r")
    kor_sts_train = kor_sts_train.readlines()
    
    kor_sts_dev = open('data/sts-dev.tsv.txt', "r")
    kor_sts_dev = kor_sts_dev.readlines()
    return kor_sts_train, kor_sts_dev

def kor_sts_reformat(data):
    data_lst = []
    for count, i in enumerate(data):
        if count != 0:
            try:
                example = i.strip('\n')
                example2 = example.split('\t')
                score = float(example2[-3])/5
                dic = {'s1': example2[-1],
                      's2': example2[-2],
                      'score': score}
                data_lst.append(dic)
            except:
                pass
            
    return data_lst
            
def keywordRank(data):
    sent_list = list()
    
    okt = Okt()
    tap = '	'
    for line in data:
        token = line.split(tap)
        senta = token[5]
        sentb = token[6].replace('\n', '')
        sent_list.append(senta)
        sent_list.append(sentb)
        
    sentences = list(set(sent_list))
    min_count = 2   # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length = 10 # 단어의 최대 길이
    wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)
    
    beta = 0.85    # PageRank의 decaying factor beta
    max_iter = 30
    keywords, rank, graph = wordrank_extractor.extract(sentences, beta, max_iter)
    
    imp_words = []
    for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:]:
            if len(okt.nouns(word)) != 0:
                imp_words.append(word)
                
                    #print('%8s:\t%.4f' % (word, r))
    return imp_words

def dp_scan(dp):
    np_lst = [i for i in dp if 'NP' in i[3]]
    if len(np_lst) == 0:
        return 0
    return np_lst

def np_lst_scan(sent,dp, imp_keys, okt):
    np_listss= []  
    
    
    for i in dp:
        if i[3] =='NP_OBJ' and i[1] in imp_keys:
            phrase = i[1] 
            st =i[2]
            in_dic_noun = okt.nouns(phrase)
            in_dic = {'s1':in_dic_noun[0], 's2': sent,'score':0.3}

            while st!=-1:
                next_word = ' '+dp[st-1][1]
                phrase +=next_word
                st = dp[st-1][2]
             
            np_listss.append((phrase, 'NP_OBJ', sent, in_dic))
    
    return np_listss

def create_n_gram_in(data, imp_key, dp_model):
    save_lst = []
    dic_lst = []
    okt = Okt()

    for i in data:
        sent1, sent2 = i['s1'], i['s2']
        dp1 = dp_model(sent1)
        dp2 = dp_model(sent2)

        np_lst1 = dp_scan(dp1)
        np_lst2 = dp_scan(dp2)


        if np_lst1 ==0:
            pass
        else:
            result1 = np_lst_scan(sent1, dp1,imp_key,okt)
            if len(result1) !=0:
                save_lst.extend(result1)
        if np_lst2 ==0:
            pass
        else:
            result2 = np_lst_scan(sent2, dp2,imp_key,okt)
            if len(result2) !=0:
                save_lst.extend(result2)
    
    sentences = []
    for i in data:
        sentences.append(i['s1'])
        sentences.append(i['s2'])
    sentences = list(set(sentences))
    
    for i in save_lst:
        dic={'s1':i[0], 's2':i[2], 'score':0.8} # n-gram pos
        rsent = sample(sentences,1)
        while i[1] in rsent:
            rsent = sample(sentences,1)
        
        dic2 = {'s1':i[0], 's2':rsent[0], 'score':0} # n-gram neg
        
        rsent = sample(sentences,1)
        while i[1] in rsent:
            rsent = sample(sentences,1)
        dic3 = {'s1':i[-1]['s1'], 's2':rsent[0], 'score':0} # not in
        
        dic_lst.append(dic2)
        dic_lst.append(dic3)
        dic_lst.append(dic) 
        dic_lst.append(i[-1]) # in
        
    print('pos n-gram. neg n-gram, in, not in   each has: ', len(dic_lst)/4)
    return dic_lst

def data_final(aug_data, kor_train, dev_data):
    train_samples, dev_samples = [],[]

    for example in kor_train:
        score = float(example["score"])
        inp_example = InputExample(
            texts=[example["s1"], example["s2"]], 
            label=score,)
        train_samples.append(inp_example)
        
    for example in aug_data:
        score = float(example["score"])
        inp_example = InputExample(
            texts=[example["s1"], example["s2"]], 
            label=score,)
        train_samples.append(inp_example)
        
    for example in dev_data:
        score = float(example["score"])
        inp_example = InputExample(
            texts=[example["s1"], example["s2"]], 
            label=score,)
        dev_samples.append(inp_example)
            
    return train_samples, dev_samples

def add_spt(data):
    dicf_lst = []
    for i in data:
        dict_n = {
        's1': '[SENT] '+ i['s1'],
        's2': '[SENT] '+ i['s2'],
        'score': i['score']}
        dicf_lst.append(dict_n)
    return dicf_lst

def main():
    dp_model = Pororo(task='dep_parse', lang='ko')
    sts_train, sts_dev = load_stsdataset() 
    imp_key = keywordRank(sts_train)
    train_data=kor_sts_reformat(sts_train)
    dev_data=kor_sts_reformat(sts_dev)
    
    print('start creating ngram and in data')
    tdf = create_n_gram_in(train_data, imp_key, dp_model)
    
    print(len(tdf))
    print(tdf[:8])
    if args.stokens:
        train_data = add_spt(train_data)
        dev_data = add_spt(dev_data)
        tdf = add_spt(tdf)
        
        
    train_samples, dev_samples=data_final(tdf, train_data, dev_data)
    
    print('data done')
    print('')
    print('sts_train: ', train_data[95])
    print('sts_dev: ', dev_data[16])
    print('tdf: ', tdf[0])
    
    print(len(train_samples))
    print(len(dev_samples))
    word_embedding_model = models.Transformer('klue/roberta-base', max_seq_length=128)
    
    if args.stokens:
        tokens = ["[SENT]", "[PSG]"]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        print('special tokens added !!')
        
    
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    model_save_path = 'model_results/' + args.output_model_name
    
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples,
    name="finetuning_dev",)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    #warmup_steps = math.ceil(len(train_dataloader) * 10  * 0.1)
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs / args.train_batch_size * 0.1) #10% of train data for warm-up

    
    
    print('Start model training')
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=args.epochs, 
          warmup_steps=warmup_steps,
          output_path =model_save_path)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--output_model_name", type=str, required=True)
    parser.add_argument("--stokens", type=bool, default=False)
    args = parser.parse_args()
    
    main()
