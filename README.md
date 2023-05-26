# DeepAL4NLP: Deep Active Learning for Natural Language Processing in Python 

This code is based on DeepAL [1].

In DeepAL4NLP, we implement Active Learning with Data Augmentation for textual classfication. 

It includes Python implementations of the following active learning algorithms:

- Least Confidence [2]
- Margin Sampling [3]

We use the pre-trained BERT [4] as basic classifier and fine-tune it. One can replace it to other basic classifiers and add them to nets.py. 



## Prerequisites 

- Python                3.8.16
- numpy                 1.21.2
- pandas                1.5.3
- scipy                 1.7.1
- torch                 1.11.0
- scikit-learn          1.0.1
- nlpaug                1.1.11
- transformers          4.27.4
- sentence_transformers 2.2.2

## Demo 

```
  python demo.py \
      --n_query 20 \
      --n_init_labeled 20 \
      --dataset_name SemEval_Laptops \
      --strategy_name LeastConfidence \
      --active_learning True \
      --with_augmentation True \
      --enhanced_augmentation True \
      --seed 1
```


## Reference

[1] DeepAL: Deep Active Learning in Python, arXiv preprint, 2021

[2] A Sequential Algorithm for Training Text Classifiers, SIGIR, 1994

[3] Active Hidden Markov Models for Information Extraction, IDA, 2001

[4] Bert: Pre-training of deep bidirectional transformers for language understanding, arXiv preprint, 2018





