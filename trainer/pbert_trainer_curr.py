from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import json
import torch
import torch.nn as nn
import math
import argparse
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import InputExample

def read_data(data):
    passage, query, score = [],[],[]
    for i in data:
        passage.append(i['passage'])
        query.append(i['query'])
        score.append(i['score'])
    return passage, query, score

def input_data(t_passage, t_title, t_score, d_passage, d_title, d_score,
              sbert_train, sbert_dev):
    train_samples, dev_samples =[], []
    
    for example in sbert_train:
        score = float(example["score"])
        inp_example = InputExample(
            texts=[example["s1"], example["s2"]], 
            label=score,)
        train_samples.append(inp_example)
        
    for example in sbert_dev:
        score = float(example["score"])
        inp_example = InputExample(
            texts=[example["s1"], example["s2"]], 
            label=score,)
        dev_samples.append(inp_example)
    
    for p, t, s in zip(t_passage, t_title, t_score):
        score = float(s)
        inp_example = InputExample(
            texts = [p,t],
            label=score,
        )
        train_samples.append(inp_example)

    for p, t, s in zip(d_passage, d_title, d_score):
        score = float(s)
        inp_example = InputExample(
            texts = [p,t],
            label=score,
        )
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
    with open (args.train_dataset, 'r', encoding='utf8') as fp:
        train_data = json.load(fp)
    with open (args.dev_dataset, 'r', encoding='utf8') as fp:
        dev_data = json.load(fp)
    with open ('sbert_train.json', 'r', encoding='utf8') as fp:
        sbert_train_data = json.load(fp)
    with open ('sbert_dev.json', 'r', encoding='utf8') as fp:
        sbert_dev_data = json.load(fp)
    
        
    train_passage, train_title, train_score = read_data(train_data)
    dev_passage, dev_title, dev_score = read_data(dev_data)
    
    if args.stokens:
        train_passage = ['[PSG] '+ i for i in train_passage]
        dev_passage = ['[PSG] '+ i for i in dev_passage]
        train_title = ['[SENT] '+ i for i in train_title]
        dev_title = ['[SENT] '+ i for i in dev_title]
        
        sbert_train_data = add_spt(sbert_train_data)
        sbert_dev_data = add_spt(sbert_dev_data)
        
    print('train_title: ', train_title[0])
    print('train_passage: ', train_passage[0])
    print('dev_title: ', dev_title[0])
    print('dev_passage: ', dev_passage[0])
    print('sbert_train: ', sbert_train_data[0])
    print('sbert_dev: ', sbert_dev_data[0])
    
    train_samples, dev_samples = input_data(train_passage, train_title, train_score,
                                                dev_passage, dev_title, dev_score,
                                           sbert_train_data, sbert_dev_data)

    word_embedding_model = models.Transformer(args.model_path, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model_save_path ='model_results_ft/' + args.output_model_name

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    warmup_steps = math.ceil(len(train_dataloader)*args.epochs / args.train_batch_size *0.1)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="finetuning_dev")
    print('Start model training')
        
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=args.epochs, 
            warmup_steps=warmup_steps,
            output_path =model_save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output_model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--stokens", type=bool, default=False)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--dev_dataset", type=str, required=True)
    
    args = parser.parse_args()
    
    main()
