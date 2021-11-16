import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from models.rstnet.grid_aug import PositionEmbeddingSine
from data import ImageDetectionsField, TextField, RawField
from transformers import BertModel, RobertaModel, PhobertTokenizer
import pickle

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.grid_embedding = PositionEmbeddingSine(self.decoder.d_model // 2, normalize=True)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()
        self.text_field = TextField(init_token='<s>', eos_token='</s>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
        self.text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
        self.bert_tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')
        self.dict_stoi_bert = {value: [self.bert_tokenizer.convert_tokens_to_ids(key)] for key, value in dict(self.text_field.vocab.stoi).items()}
        #self.dict_stoi_bert = {value: self.bert_tokenizer.encode(key)[1:-1] for key, value in dict(self.text_field.vocab.stoi).items()}

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_pos_embedding(self, grids):
        bs = grids.shape[0]
        grid_embed = self.grid_embedding(grids.view(bs, 7, 7, -1))
        return grid_embed

    def forward(self, images, seq, input_ids_bert, masks, lens):
        grid_embed = self.get_pos_embedding(images)
        enc_output, mask_enc = self.encoder(images, pos=grid_embed)
        dec_output = self.decoder(seq, input_ids_bert, masks, lens, enc_output, mask_enc, pos=grid_embed)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        lens, masks = [], []
        
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.grid_embed = self.get_pos_embedding(visual)
                self.enc_output, self.mask_enc = self.encoder(visual, pos=self.grid_embed)

                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        # input_ids_bert, masks, lens

        sequences = it.tolist()
        ori_max_len = it.shape[1]

        input_ids_bert_subword = [[self.dict_stoi_bert[s] for s in sequence] for sequence in sequences]
        input_ids_bert = [sum([self.dict_stoi_bert[s] for s in sequence], []) for sequence in sequences]
        max_len = max([len(i) for i in input_ids_bert])
        input_ids_bert = [i + [1]*(max_len-len(i)) for i in input_ids_bert]
        input_ids_bert = torch.tensor(input_ids_bert).to('cuda')

        masks = [[1]*max_len for i in range(input_ids_bert.shape[0])]
        lens = [[len(i[0])] for i in input_ids_bert_subword]

        masks = torch.tensor(masks).to('cuda')
        masks = masks.gt(0)
        lens = torch.tensor(lens).to('cuda')

        # input_ids_bert = {'sub' : [], 'flatten': []}

        # for sequence in sequences:
        #   ids = [self.dict_stoi_bert[s] for s in sequence]
        #   flatten_ids = sum(ids, [])
        #   input_ids_bert['sub'].append(ids)
        #   input_ids_bert['flatten'].append(flatten_ids)
        
        # max_len = max([len(x) for x in input_ids_bert['flatten']])

        # print('-----------------------------------------')
        # print(input_ids_bert['flatten'])
        # print('max_len', max_len)
        
        # for i, ids in enumerate(input_ids_bert['flatten']):
        #   masks.append([1]*len(ids) + [1]*(max_len-len(ids)))
        #   input_ids_bert['flatten'][i] = ids + [1]*(max_len-len(ids))
        #   lens.append([len(piece) for piece in input_ids_bert['sub'][i]] + \
        #    [1]*(max_len-len(input_ids_bert['flatten'][i])))

        
        # input_ids_bert = torch.tensor(input_ids_bert['flatten']).to('cuda')
        # masks = torch.tensor(masks).to('cuda')
        # masks = masks.gt(0)
        # lens = torch.tensor(lens).to('cuda')

        #import pdb; pdb.set_trace()

        return self.decoder(it, input_ids_bert, masks, lens,  self.enc_output, self.mask_enc, pos=self.grid_embed)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
