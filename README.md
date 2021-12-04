# chaii - hindi & tamil question answering

This is the solution for rank 5th in Kaggle competition: chaii - Hindi and Tamil Question Answering. The competition can be found here: https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering

## Datasets required

Download squadv2 data from https://rajpurkar.github.io/SQuAD-explorer/

    $ mkdir input && cd input
    $ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
    $ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json


Download tydiqa data in the `input` folder:

    $ wget https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json
    $ wget https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json


Download data from https://www.kaggle.com/tkm2261/google-translated-squad20-to-hindi-and-tamil to `input` folder

Download original competition dataset to `input` folder: https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/data

Download outputs of this kernel: https://www.kaggle.com/rhtsingh/external-data-mlqa-xquad-preprocessing/ to `input` folder


Now, you have all the data needed to train the model. We will first create folds and munge the data a bit.

To create folds, please use the following command:

    $ cd src
    $ python create_folds.py

To munge the datasets and prepare for training, please run the following command:

    $ cd src
    $ python munge_data.py

## Training

There are two GPU models and one model needs TPUs.

GPU models: XLM-Roberta & Rembert
TPU model: Muril-Large

### XLM-Roberta:

    $ cd src
    $ TOKENIZERS_PARALLELISM=false python xlm_roberta.py --fold 0
    $ TOKENIZERS_PARALLELISM=false python xlm_roberta.py --fold 1
    $ TOKENIZERS_PARALLELISM=false python xlm_roberta.py --fold 2
    $ TOKENIZERS_PARALLELISM=false python xlm_roberta.py --fold 3
    $ TOKENIZERS_PARALLELISM=false python xlm_roberta.py --fold 4

### Rembert:

    $ cd src
    $ TOKENIZERS_PARALLELISM=false python rembert.py --fold 0
    $ TOKENIZERS_PARALLELISM=false python rembert.py --fold 1
    $ TOKENIZERS_PARALLELISM=false python rembert.py --fold 2
    $ TOKENIZERS_PARALLELISM=false python rembert.py --fold 3
    $ TOKENIZERS_PARALLELISM=false python rembert.py --fold 4

## Muril-Large

** please note that training this model needs TPUs **

    $ cd src
    $ TOKENIZERS_PARALLELISM=false python muril_large.py --fold 0
    $ TOKENIZERS_PARALLELISM=false python muril_large.py --fold 1
    $ TOKENIZERS_PARALLELISM=false python muril_large.py --fold 2
    $ TOKENIZERS_PARALLELISM=false python muril_large.py --fold 3
    $ TOKENIZERS_PARALLELISM=false python muril_large.py --fold 4


## Inference

After training all the models, the outputs were pushed to Kaggle Datasets.

The final model datasets can be found here:

    - https://www.kaggle.com/abhishek/xlmrobertalargewithsquadv2tydiqasqdtrans384f
    - https://www.kaggle.com/ubamba98/modelsrembertwithsquadv2tydiqa384
    - https://www.kaggle.com/ubamba98/murillargecasedchaii

And the final inference kernel can be found here: https://www.kaggle.com/abhishek/chaii-xlm-roberta-x-muril-x-rembert-score-based


Solution writeup: https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/288049