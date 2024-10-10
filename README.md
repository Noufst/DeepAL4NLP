# EASE: Enhanced Active learning based on Sample diversity and data augmEntation

This code is based on DeepAL [1].

In EASE, we implement Active Learning with Data Augmentation and Sample Diversity for textual classfication.

It includes Python implementations of the following active learning algorithms:

- Least Confidence [2]
- Margin Sampling [3]

We use the pre-trained BERT [4] as basic classifier and fine-tune it. One can replace it to other basic classifiers and add them to nets.py. 



## Prerequisites 

- Python                3.10.12
- numpy                 1.26.4
- pandas                2.1.4
- torch                 2.3.1
- sklearn               1.3.2
- nlpaug                1.1.11
- transformers          4.42.4
- sentence_transformers 3.0.1


## Demo 

```
  python demo.py \
      --n_query 20 \
      --n_init_labeled 20 \
      --patient 5 \
      --dataset_name SemEval_Restaurants \
      --strategy_name LeastConfidence \
      --active_learning True \
      --enhanced_active_learning True \
      --with_augmentation True \

```


## Reference

[1] DeepAL: Deep Active Learning in Python, arXiv preprint, 2021

[2] A Sequential Algorithm for Training Text Classifiers, SIGIR, 1994

[3] Active Hidden Markov Models for Information Extraction, IDA, 2001

[4] Bert: Pre-training of deep bidirectional transformers for language understanding, arXiv preprint, 2018





