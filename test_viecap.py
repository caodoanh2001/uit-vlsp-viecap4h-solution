#RSTNet
import random
import os
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.rstnet import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
#from models.m2_transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import time
import json

import h5py
import glob
import itertools
from tqdm import tqdm
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

features_path = './X152++_VieCap_feature_test.hdf5'
img_path = './images_public_test'
annotation_folder = './annotations_VieCap'
vocab_path = './backup_12_10_2021/vocab_2946.pkl'
model_path = './backup_12_10_2021/rstnet_2946.pth'
path_sample_submission = './sample_submission.json'
device = 'cuda'
batch_size = 40
workers = 4

# Pipeline for image regions
image_field = ImageDetectionsField(detections_path=features_path, max_detections=49, load_in_tmp=False)

# Pipeline for text
text_field = TextField(init_token='<s>', eos_token='</s>', lower=True, tokenize='spacy',
                        remove_punctuation=True, nopoints=False)

# Create the dataset
dataset = COCO(image_field, text_field, img_path, annotation_folder, annotation_folder)
_, _, test_dataset = dataset.splits

text_field.vocab = pickle.load(open(vocab_path, 'rb'))

# Model and dataloaders
encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
model = Transformer(text_field.vocab.stoi['<s>'], encoder, decoder).to(device)

#encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': 40})
#decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
#model = Transformer(text_field.vocab.stoi['<s>'], encoder, decoder).to(device)

data = torch.load(model_path)
model.load_state_dict(data['state_dict'])

dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=batch_size, num_workers=workers)

### PREDICTING ... ###

max_detections = 40
image_ids = [i.split('/')[-1] for i in \
            glob.glob(os.path.join(img_path, '*'))]
f = h5py.File(features_path, 'r')

# Trainval
#json_trainval = json.load(open('./annotations_VieCap_train/train.json', 'r'))
#image_ids = [item['file_name'].split('.')[0] for item in json_trainval['images']]

results = []
for image_name in tqdm(image_ids):
    image = f['%s_grids' % image_name.split('.')[0]][()]
    torch_image = torch.from_numpy(np.array([image])).to(device)
    with torch.no_grad():
        out, _ = model.beam_search(torch_image, 23, text_field.vocab.stoi['</s>'], 3, out_size=1)
    caps_gen = text_field.decode(out, join_words=False)
    gen_i = ' '.join([k for k, g in itertools.groupby(caps_gen[0])])
    gen_i = gen_i.strip().replace('_',' ')
    results.append({"id": image_name, "captions": gen_i})


with open(path_sample_submission, 'r') as f:
  sample_submission = json.load(f)

for item in sample_submission:
    for result in results:
      if item['id'] == result['id']:
          item['captions'] = result['captions'].split(' ')[0].capitalize() + ' ' + ' '.join(result['captions'].split(' ')[1:]) + '.'
          continue

with open('results.json', 'w') as f:
  json.dump(sample_submission, f, indent=4, ensure_ascii=False)
