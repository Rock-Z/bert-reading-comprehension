import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare


# use https://analyticsindiamag.com/converting-a-model-from-pytorch-to-tensorflow-guide-to-onnx/
# to convert Pytorch to Tensorflow

class SSingleMatchNet(nn.Module):
    def __init__(self, config):
        super(SSingleMatchNet, self).__init__()
        self.map_linear = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop_module = nn.Dropout(2*config.hidden_dropout_prob)
        self.rank_module = nn.Linear(config.hidden_size * 2, 1)

class BertForMultipleChoiceWithMatch(PreTrainedBertModel):

    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoiceWithMatch, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(2*config.hidden_size, 1)
        self.classifier3 = nn.Linear(3*config.hidden_size, 1)
        self.classifier4 = nn.Linear(4*config.hidden_size, 1)
        self.classifier6 = nn.Linear(6*config.hidden_size, 1)
        self.ssmatch = SSingleMatchNet(config)
        self.pooler = BertPooler(config)
        self.fuse = FuseNet(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None, option_len=None, labels=None, is_3 = False):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()

        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        sequence_output, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)

        doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(sequence_output, doc_len, ques_len, option_len)
       
        pa_output = self.ssmatch([doc_seq_output, option_seq_output, option_len+1])
        ap_output = self.ssmatch([option_seq_output, doc_seq_output, doc_len+1])
        pq_output = self.ssmatch([doc_seq_output, ques_seq_output, ques_len+1])
        qp_output = self.ssmatch([ques_seq_output, doc_seq_output, doc_len+1])
        qa_output = self.ssmatch([ques_seq_output, option_seq_output, option_len+1])
        aq_output = self.ssmatch([option_seq_output, ques_seq_output, ques_len+1])

        pa_output_pool, _ = pa_output.max(1)
        ap_output_pool, _ = ap_output.max(1)
        pq_output_pool, _ = pq_output.max(1)
        qp_output_pool, _ = qp_output.max(1)
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)

        pa_fuse= self.fuse([pa_output_pool, ap_output_pool])
        pq_fuse= self.fuse([pq_output_pool, qp_output_pool])
        qa_fuse= self.fuse([qa_output_pool, aq_output_pool])

        cat_pool = torch.cat([pa_fuse, pq_fuse, qa_fuse], 1)
        output_pool = self.dropout(cat_pool)
        match_logits = self.classifier3(output_pool)
        match_reshaped_logits = match_logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss
        else:
            return match_reshaped_logits

