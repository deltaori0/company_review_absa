import torch
import torch.nn as nn
from transformers import BertModel

class KoBERT(nn.Module):
    def __init__(self, opt, embed_dim=768, fc_hid_dim=128, top_k=3, att_head='all', att_pooling='mean'):
        assert type(att_head)==list or att_head=='all', "att_head should be 'all' or list type [3, 5]"
        super(KoBERT, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.fc_hid_dim = fc_hid_dim
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.top_k = top_k
        self.att_head = att_head
        self.att_pooling = att_pooling
        if self.att_pooling == 'concat':
            self.fc1 = nn.Linear((self.embed_dim) * top_k, self.num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim, self.num_classes)
        self.device = opt.device

    def forward(self, input_ids, att_mask, token_ids):
        '''

        '''
        # [SEP] ids
        sep_ids = [(i==3).nonzero(as_tuple=True)[0] for i in input_ids]
        
        # aspect words ids
        asp_ids = [list(range(s[0]+1, s[-1])) for s in sep_ids]
        
        output_dict = self.bert(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_ids,
            output_attentions=True, encoder_hidden_states=True, return_dict=True)

        # get top-k att idx in final att layer
        last_att = output_dict.attentions[-1]
        target_heads = last_att if self.att_head == 'all' else last_att[:, self.att_head, :, :]
        atts = [torch.mean(i, dim=0) for i in target_heads] # len(atts): batch_size, atts[0].shape: (200, 200)
        
        # get top-k score words ids
        top_k_words = list() # len 16 list, [0]: len top-k list [14, 3, 17]
        for att, asp, sep in zip(atts, asp_ids, sep_ids):
            att_score = sum(att[asp, :]) # (200), sum att scores of asp words
            top_k_idx = torch.sort(att_score[1:sep[0]], descending=True).indices[:self.top_k] # (3), exclude [CLS], [SEP], aspect words
            top_k_words.append(top_k_idx+1) # consider [CLS] token for idx
        
        # get top-k hidden states
        hids = output_dict.last_hidden_state
        output = get_hiddens(last_hids=hids, top_k_list=top_k_words, pooling=self.att_pooling) # only top-k att score words
        output = self.fc1(output)
        return output, top_k_words

def get_hiddens(last_hids, top_k_list, pooling='mean'):
    '''
    @args
    last_hids: bert last hidden states
    top_k_list: idxs to get hids, top-k words or top-k words + aspect words
    pooling: how get final rep. vetors, 'mean', 'sum', 'concat'
    '''
    final = list()
    for idx, hid in zip(top_k_list, last_hids):
        if pooling=='sum':
            final.append(torch.sum(hid[idx, :], dim=0).unsqueeze(0)) # (1, 768)
        elif pooling=='mean' or pooling=='average':
            final.append(torch.mean(hid[idx, :], dim=0).unsqueeze(0)) # (1, 768)
        elif pooling=='concat':
            final.append(hid[idx, :].view(1, -1)) # (1, 768*k)
        elif pooling in ['rnn', 'lstm', 'gru']:
            final.append(hid[idx, :].unsqueeze(0))
    final = torch.cat(final, dim=0) # to tensor (batch_size, 768) or (batch_size, 768*k) (concat)
    return final