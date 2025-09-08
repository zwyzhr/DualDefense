
#  图2 子图1
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion trimmed_mean --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion cos_defense --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion krum --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion average --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion median --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion clipping_median --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion fedavg --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda


#  图2 子图2
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion trimmed_mean --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion cos_defense --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion krum --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion average --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion median --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion clipping_median --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
#python main.py --num_clients 100 --dataset mnist --fusion fedavg --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda



#  图2 子图3
python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset mnist --fusion trimmed_mean --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset mnist --fusion cos_defense --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset mnist --fusion krum --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset mnist --fusion average --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset mnist --fusion median --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset mnist --fusion clipping_median --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset mnist --fusion fedavg --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda


#  图2 子图4
python main.py --num_clients 100 --dataset fmnist --fusion dual_defense --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion trimmed_mean --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion cos_defense --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion krum --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion average --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion median --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion clipping_median --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion fedavg --training_round 100 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_ipm --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda


#  图2 子图5
python main.py --num_clients 100 --dataset fmnist --fusion dual_defense --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion trimmed_mean --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion cos_defense --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion krum --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion average --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion median --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion clipping_median --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion fedavg --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_alie --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda



#  图2 子图6
python main.py --num_clients 100 --dataset fmnist --fusion dual_defense --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion trimmed_mean --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion cos_defense --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion krum --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion average --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion median --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion clipping_median --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda
python main.py --num_clients 100 --dataset fmnist --fusion fedavg --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda

python main.py --num_clients 100 --dataset fdmnist --fusion fedavg --training_round 150 --local_epochs 3 --optimizer sgd  --batch_size 64  --learning_rate 0.01 --attacker_strategy model_poisoning_scaling --attack_start_round 50  --attacker_ratio 0.3 --epsilon 0.01 --device cuda




#python main.py --num_clients 100 --dataset mnist --fusion fedavg --training_round 10 --local_epochs 3 --optimizer sgd --batch_size 64 --regularization 1e-5
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 10 --local_epochs 3 --batch_size 64 --attacker_strategy model_poisoning_ipm --attack_start_round 5 --attacker_ratio 0.2 --epsilon 0.01



#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 20 --local_epochs 3 --batch_size 64 \
#  --attacker_strategy model_poisoning_ipm --attack_start_round 5 --attacker_ratio 0.1 --epsilon 0.01
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 20 --local_epochs 3 --batch_size 64 \
#  --attacker_strategy model_poisoning_ipm --attack_start_round 5 --attacker_ratio 0.2 --epsilon 0.01
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 20 --local_epochs 3 --batch_size 64 \
#  --attacker_strategy model_poisoning_ipm --attack_start_round 5 --attacker_ratio 0.3 --epsilon 0.01
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 20 --local_epochs 3 --batch_size 64 \
#  --attacker_strategy model_poisoning_ipm --attack_start_round 5 --attacker_ratio 0.4 --epsilon 0.01
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 20 --local_epochs 3 --batch_size 64 \
#  --attacker_strategy model_poisoning_ipm --attack_start_round 5 --attacker_ratio 0.5 --epsilon 0.01





#  python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 100 --local_epochs 10 --batch_size 64 --attacker_strategy model_poisoning_ipm --attack_start_round 5 --attacker_ratio 0.5 --epsilon 0.01