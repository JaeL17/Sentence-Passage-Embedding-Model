# special token X
CUDA_VISIBLE_DEVICES=2 nohup python -u pbert_trainer_adap.py --output_model_name ours_ad --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all/ >> ours_ad.out  &

CUDA_VISIBLE_DEVICES=2 nohup python -u pbert_trainer_adap.py --output_model_name ours_ad_2 --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all/ >> ours_ad_2.out  &

CUDA_VISIBLE_DEVICES=1 nohup python -u pbert_trainer_adap.py --output_model_name ours_ad_3 --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all/ >> ours_ad_3.out  &

# exc_neu_data
CUDA_VISIBLE_DEVICES=1 nohup python -u pbert_trainer_adap.py --output_model_name ours_ad_exc_neu --train_dataset data/tq_train_exc_neu.json --dev_dataset data/tq_dev_exc_neu.json --model_path model_results/sbert_iq_all/ >> ours_ad_exc_neu.out  &

# special token O
CUDA_VISIBLE_DEVICES=2 nohup python -u pbert_trainer_adap.py --output_model_name ours_ad_st --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_ad_st.out  &

CUDA_VISIBLE_DEVICES=1 nohup python -u pbert_trainer_adap.py --output_model_name ours_ad_st_2 --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_ad_st_2.out  &

CUDA_VISIBLE_DEVICES=2 nohup python -u pbert_trainer_adap.py --output_model_name ours_ad_st_3 --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_ad_st_3.out  &

# exc_neu_data
CUDA_VISIBLE_DEVICES=1 nohup python -u pbert_trainer_adap.py --output_model_name ours_ad_st_exc_neu --train_dataset data/tq_train_exc_neu.json --dev_dataset data/tq_dev_exc_neu.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_ad_st_exc_neu.out  &

# curriculum learning
CUDA_VISIBLE_DEVICES=2 nohup python -u pbert_trainer_curr.py --output_model_name ours_curr --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_curr.out  &

CUDA_VISIBLE_DEVICES=2 nohup python -u pbert_trainer_curr.py --output_model_name ours_curr_2 --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_curr_2.out  &

CUDA_VISIBLE_DEVICES=1 nohup python -u pbert_trainer_curr.py --output_model_name ours_curr_3 --train_dataset data/tq_train_v2.json --dev_dataset data/tq_dev_v2.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_curr_3.out  &

# exc_neu_data
CUDA_VISIBLE_DEVICES=2 nohup python -u pbert_trainer_curr.py --output_model_name ours_curr_exc_neu --train_dataset data/tq_train_exc_neu.json --dev_dataset data/tq_dev_exc_neu.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_curr_exc_neu.out  &

CUDA_VISIBLE_DEVICES=2 nohup python -u pbert_trainer_curr.py --output_model_name ours_curr_exc_neu_2 --train_dataset data/tq_train_exc_neu.json --dev_dataset data/tq_dev_exc_neu.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_curr_exc_neu2.out  &

CUDA_VISIBLE_DEVICES=1 nohup python -u pbert_trainer_curr.py --output_model_name ours_curr_exc_neu_3 --train_dataset data/tq_train_exc_neu.json --dev_dataset data/tq_dev_exc_neu.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_curr_exc_neu3.out  &

CUDA_VISIBLE_DEVICES=1 nohup python -u pbert_trainer_curr.py --output_model_name ours_curr_exc_neu_4 --train_dataset data/tq_train_exc_neu.json --dev_dataset data/tq_dev_exc_neu.json --model_path model_results/sbert_iq_all_st --stokens True >> ours_curr_exc_neu4.out  &