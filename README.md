# vieCap4H Challenge 2021: A transformer-based method for Healthcare Image Captioning in Vietnamese


This repo GitHub contains our solution for vieCap4H Challenge 2021. In detail, we use grid features as visual presentation and pre-training a BERT-based language model from PhoBERT-based pre-trained model to obtain language presentation. Besides, we indicate a suitable schedule with the self-critical training sequence (SCST) technique to achieve the best results. Through experiments, we achieve an average of BLEU 30.3% on the public-test round and 28.9% on the private-test round, which ranks 3rd and 4th, respectively.

![](https://i.imgur.com/LuJHJ83.png)

**Figure 1.** An overview of our solution based on RSTNet

## 1. Data preparation

The grid features of vieCap4H can be downloaded via links below:

- X101:
    - [Train](https://drive.google.com/file/d/1lbuDA6gcL5HPcMqicyRD2kQ--ehY6cIV/view?usp=sharing)
    - [Public-test](https://drive.google.com/file/d/1-8j_gu8aS8rEDmsasaQ2FePUrIV9aF2r/view?usp=sharing)
- X152:
    - [Train](https://drive.google.com/file/d/1WAf8n0_9GxSwPle4OycyEJ0jIiN_E7Bu/view?usp=sharing)
    - [Public-test](https://drive.google.com/file/d/1024QBniesjPSI1KUR6-2JPPXPKIBM1wQ/view?usp=sharing)
- X152++:
    - [Train](https://drive.google.com/file/d/12cndu-64vryHPPnQbhV7IGdElphseF2z/view?usp=sharing)
    - [Public-test](https://drive.google.com/file/d/15SRL6MF9lBnbXJ3m99ole8t3WZSHjClQ/view?usp=sharing)
    - [Private-test](https://drive.google.com/file/d/166U72LCCcstJE41XXiKh7iSFxkFRU6CN/view?usp=sharing)

Dataset can be downloaded at https://aihub.vn/competitions/40
Annotations must be converted to COCO format. We have already converted and it is available at:
- [viecap4h-public-train.json](https://drive.google.com/file/d/11xsDl3ZTm84uz6BEgzIYFnPLc3SykOJO/view?usp=sharing).

## 2. Training

Pre-training BERT-based model with `PhoBERT-based`
```
python train_language.py \
--img_path <images path> \
--features_path <features path> \
--annotation_folder <annotations folder> \
--batch_size 40
```

Weights of BERT-based model should be appeared in folder `saved_language_models`

Then, continue to train Transformer model via command below::

```
python train_transformer.py \
--img_path <images path> \
--features_path <features path> \
--annotation_folder <annotations folder> \
--batch_size 40
```

Weights of Transformr-based model should be appeared in folder `saved_transformer_rstnet_models`

Where `<images path>` is data folder, `<features path>` is the path of grid features folder, `<annotations folder>` is the path of folder that contains file `viecap4h-public-train.json`.

## 3. Inference

The results can be obtained via command below:

```
python test_viecap.py
```

## 4. Reproduction

To implement our results on leaderboard, two pretrained models for BERT-based model, Transformer model can be downloaded via links below:

- [BERT-based language model](https://drive.google.com/file/d/1NlpAHVLGyX_SelHseNxjSiAmKj5942OR/view?usp=sharing), should be put in `saved_language_models` folder
- [Transformer model](https://drive.google.com/file/d/11UsfZReuMU90FtY4aH7vwzLTaozcvmud/view?usp=sharing).

Besides, we also prepared our vocabulary file used for training and sample submission to arrange the predicted captions like the organizer.
- [Vocabulary](https://drive.google.com/file/d/1IYHSpwJMOg11IkhR5ALCRXiJHGI84oaL/view?usp=sharing)
- [Sample submission](https://drive.google.com/file/d/1noKu57koburNq9u2nofRkq4Mof0jpyNG/view?usp=sharing)

Then, run the command line below for result reproduction:

```
python test_viecap.py
```
