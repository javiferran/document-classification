
"""
Code adapted from https://github.com/uzaymacar/comparatively-finetuning-bert
"""


import logging
import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch_transformers import BertConfig, BertTokenizer, BertModel
from pytorch_transformers import BertConfig, DistilBertTokenizer, DistilBertModel
from data_utils import tokenize_and_encode, get_features

# Set logging level to INFO to print number of learnable parameters
logging.getLogger().setLevel(logging.INFO)


class FineTunedBert(nn.Module):
    """
    Finetuning model that utilizes BERT tokenizer, pretrained BERT embedding, pretrained BERT
    encoders, an optional recurrent neural network  choice of LSTM, dropout, and finally a dense
    layer for classification.

    @param (str) pretrained_model_name: name of the pretrained BERT model for tokenizing input
           sequences, extracting vector representations for each token, [...]
    @param (int) num_pretrained_bert_layers: number of BERT Encoder layers to be utilized
    @param (int) max_tokenization_length: maximum number of positional embeddings, or the sequence
           length of an example that will be fed to BERT model (default: 512)
    @param (int) num_classes: number of classes to distinct between for classification; specify
           2 for binary classification (default: 1)
    @param (bool) top_down: whether to assign parameters (weights and biases) in order or
           backwards (default: True)
    @param (int) num_recurrent_layers: number of LSTM layers to utilize (default: 1)
    @param (bool) use_bidirectional: whether to use a bidirectional LSTM or not (default: False)
    @param (int) hidden_size: number of recurrent units in each LSTM cell (default: 128)
    @param (bool) reinitialize_pooler_parameters: whether to use the pretrained pooler parameters
           or initialize weights as ones and biases zeros and train for scratch (default: False)
    @param (float) dropout_rate: possibility of each neuron to be discarded (default: 0.10)
    @param (bool) aggregate_on_cls_token: whether to pool on only the hidden states of the [CLS]
           token for classification or on the hidden states of all (512) tokens (default: True)
    @param (bool) concatenate_hidden_states: whether to concatenate all the available hidden states
           outputted by the embedding and encoder layers (K+1) or only use the latest hidden state
           (default: False)
    @param (bool) use_gpu: whether to utilize GPU (CUDA) or not (default: False)
    """
    def __init__(self, pretrained_model_name, num_pretrained_bert_layers, max_tokenization_length,
                 num_classes=1, top_down=True, num_recurrent_layers=1, use_bidirectional=False,
                 hidden_size=128, reinitialize_pooler_parameters=False, dropout_rate=0.10,
                 aggregate_on_cls_token=True, concatenate_hidden_states=False, use_gpu=False):
        super(FineTunedBert, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.use_bidirectional = use_bidirectional
        self.hidden_size = hidden_size
        self.aggregate_on_cls_token = aggregate_on_cls_token
        self.concatenate_hidden_states = concatenate_hidden_states
        self.use_gpu = use_gpu

        print('loading bert tokenizer')

        type_model = 'bert'

        # Configure tokenizer
        if type_model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('./.cache/torch/pytorch_transformers/bert-base-uncased-vocab.txt')
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained('./.cache/torch/pytorch_transformers/bert-base-uncased-vocab.txt')
        # self.tokenizer = BertTokenizer.from_pretrained('./.cache/torch/pytorch_transformers/vocab_sci.txt')

        print('loaded')
        self.tokenizer.max_len = max_tokenization_length

        print('loading bert config')

        # Get global BERT config
        if type_model == 'bert':
            self.config = BertConfig.from_pretrained('./.cache/torch/pytorch_transformers/bert-base-uncased-config.json')
        else:
            self.config = BertConfig.from_pretrained('./.cache/torch/pytorch_transformers/distilbert-base-uncased-config.json')
        # self.config = BertConfig.from_pretrained('./.cache/torch/pytorch_transformers/config_sci.json')
        print('loaded')

        print('loading bert model')
        # Extract all parameters (weights and bias matrices) for the 12 layers
        if type_model == 'bert':
            all_states_dict = BertModel.from_pretrained('./.cache/torch/pytorch_transformers/bert-base-uncased-pytorch_model.bin',#tobacco_pretrained.bin
                                                    config=self.config).state_dict()
        else:
            all_states_dict = DistilBertModel.from_pretrained('./.cache/torch/pytorch_transformers/distilbert-base-uncased-pytorch_model.bin',
                                                        config=self.config).state_dict()
        # all_states_dict = BertModel.from_pretrained('./.cache/torch/pytorch_transformers/pytorch_model_sci.bin',#tobacco_pretrained.bin
        #                                             config=self.config).state_dict()
        print('loaded')

        # Get customized BERT config
        self.config.max_position_embeddings = max_tokenization_length
        self.config.num_hidden_layers = num_pretrained_bert_layers
        self.config.output_hidden_states = True
        self.config.output_attentions = True

        # Get pretrained BERT model & all its learnable parameters
        if type_model == 'bert':
            self.bert = BertModel.from_pretrained('./.cache/torch/pytorch_transformers/bert-base-uncased-pytorch_model.bin',#bert-base-uncased-pytorch_model.bin
                                                  config=self.config)
        else:
            self.bert = DistilBertModel.from_pretrained('./.cache/torch/pytorch_transformers/distilbert-base-uncased-pytorch_model.bin',#bert-base-uncased-pytorch_model.bin
                                                  config=self.config)
        # self.bert = BertModel.from_pretrained('./.cache/torch/pytorch_transformers/pytorch_model_sci.bin',#bert-base-uncased-pytorch_model.bin
        #                                    config=self.config)

        current_states_dict = self.bert.state_dict()

        # Assign matching parameters (weights and biases of all kinds of layers)
        # i)  Top-Down Approach: 1st layer takes weights of 1st pretrained BERT layer
        if top_down:
            for param in current_states_dict.keys():
                if 'pooler' not in param or not reinitialize_pooler_parameters:
                    current_states_dict[param] = all_states_dict[param]
                else:
                    if 'weight' in param:
                        current_states_dict[param] = torch.ones(self.config.hidden_size,
                                                                self.config.hidden_size)
                    elif 'bias' in param:
                        current_states_dict[param] = torch.zeros(self.config.hidden_size)

        # ii) Bottom-Up Approach: 1st layer takes weights of 12th (last) pretrained BERT layer
        else:
            align = 5 + ((12 - num_pretrained_bert_layers) * 16)
            for index, param in enumerate(current_states_dict.keys()):
                # There are 5 initial (shared) parameters from embeddings in each BERT model
                if index < 5 and 'embeddings' in param:
                    current_states_dict[param] = all_states_dict[param]
                # There are 16 parameters for each of the K pretrained BERT layers (16 x K params)
                elif index >= 5 and 'pooler' not in param:
                    current_states_dict[param] = list(all_states_dict.values())[align:][index-5]
                # There are 2 parameters for the pooling layer at the end in each BERT model
                else:
                    if not reinitialize_pooler_parameters:
                        current_states_dict[param] = all_states_dict[param]
                    else:
                        if 'weight' in param:
                            current_states_dict[param] = torch.ones(self.config.hidden_size,
                                                                    self.config.hidden_size)
                        elif 'bias' in param:
                            current_states_dict[param] = torch.zeros(self.config.hidden_size)

        del all_states_dict


        # Update parameters in extracted BERT model
        self.bert.load_state_dict(current_states_dict)

        logging.info('Loaded %d learnable parameters from pretrained BERT model with %d layer(s)' %
                     (len(list(self.bert.parameters())), num_pretrained_bert_layers))

        # Number of input hidden dimensions from the final BERT layer, as input to other layers
        input_hidden_dimension = None
        if concatenate_hidden_states:
            input_hidden_dimension = (num_pretrained_bert_layers + 1) * self.config.hidden_size
        else:
            input_hidden_dimension = self.config.hidden_size

        # Define additional layers & utilities specific to the finetuned task
        # Flatten tensors to (B, P*(H or H')) -> converts tensors to 2D for classification
        self.flatten_sequence_length = lambda t: t.view(-1,
                                                        self.config.max_position_embeddings *
                                                        input_hidden_dimension)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)
        if self.num_recurrent_layers > 0:
            # Recurrent Layer
            self.lstm = nn.LSTM(input_size=input_hidden_dimension,
                                hidden_size=hidden_size,
                                num_layers=num_recurrent_layers,
                                bidirectional=use_bidirectional,
                                batch_first=True)
            # Dense Layer for Classification
            self.clf = nn.Linear(in_features=hidden_size*2 if use_bidirectional else hidden_size,
                                 out_features=num_classes)
        else:
            # Dense Layer for Classification
            if aggregate_on_cls_token:
                self.clf = nn.Linear(in_features=input_hidden_dimension,
                                     out_features=num_classes)
            else:
                self.clf = nn.Linear(in_features=max_tokenization_length * input_hidden_dimension,
                                     out_features=num_classes)

    def get_tokenizer(self):
        """Function to easily access the BERT tokenizer"""
        return self.tokenizer

    def get_bert_attention(self, raw_sentence, device):
        """Function for getting the multi-head self-attention output from pretrained BERT"""
        # Tokenize & encode raw sentence
        x = tokenize_and_encode(text=raw_sentence,                             # (P)
                                tokenizer=self.get_tokenizer(),
                                max_tokenization_length=self.config.max_position_embeddings,
                                truncation_method='head-only')
        # Convert the tokenized list to a Tensor
        x = torch.tensor(data=x, device=device)
        # Reshape input for BERT output
        x = x.unsqueeze(dim=1).view(1, -1)                                     # (B=1, P)
        # Get features
        token_type_ids, attention_mask = get_features(input_ids=x,
                                                      tokenizer=self.get_tokenizer(),
                                                      device=device)
        # Pass tokenized sequence through pretrained BERT model
        bert_outputs = self.bert(input_ids=x,                                  # (...) SEE forward()
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 position_ids=None,
                                 head_mask=None)
        attention_outputs = bert_outputs[3]                                    # ([K] x (1, N, P, P))
        return attention_outputs

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,     # input_ids: (B, P)
                position_ids=None, head_mask=None):
        """Function implementing a forward pass of the model"""
        # Pass tokenized sequence through pretrained BERT model
        bert_outputs = self.bert(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 head_mask=head_mask)
        sequence_output = bert_outputs[0]                                      # (B, P, H)
        pooled_output = bert_outputs[1]                                        # (B, H)
        hidden_outputs = bert_outputs[2]                                       # ([K+1] x (B, P, H))
        attention_outputs = bert_outputs[3]                                    # ([K] x (B, N, P, P))

        if self.concatenate_hidden_states:
            sequence_output = torch.cat(hidden_outputs, dim=-1)                # (B, P, H' = (K+1) x H)

        if self.num_recurrent_layers > 0:
            # Set initial states
            if self.use_gpu:
                h0 = Variable(torch.zeros(self.num_recurrent_layers * 2        # (L * 2 OR L, B, H)
                                          if self.use_bidirectional else self.num_recurrent_layers,
                                          input_ids.shape[0],
                                          self.hidden_size)).cuda()
                c0 = Variable(torch.zeros(self.num_recurrent_layers * 2        # (L * 2 OR L, B, H)
                                          if self.use_bidirectional else self.num_recurrent_layers,
                                          input_ids.shape[0],
                                          self.hidden_size)).cuda()
            else:
                h0 = Variable(torch.zeros(self.num_recurrent_layers * 2        # (L * 2 OR L, B, H)
                                          if self.use_bidirectional else self.num_recurrent_layers,
                                          input_ids.shape[0],
                                          self.hidden_size))
                c0 = Variable(torch.zeros(self.num_recurrent_layers * 2        # (L * 2 OR L, B, H)
                                          if self.use_bidirectional else self.num_recurrent_layers,
                                          input_ids.shape[0],
                                          self.hidden_size))

            lstm_output = self.lstm(sequence_output, (h0, c0))                 # (B, P, H*), (2 x (B, B, H*))
            sequence_output, _ = lstm_output

            # Get last timesteps for each example in the batch; we do this to counteract padding
            last_timesteps = []
            for i in range(len(attention_mask)):
                last_timesteps.append(
                    attention_mask[i].tolist().index(0)
                    if 0 in attention_mask[i].tolist() else self.tokenizer.max_len - 1
                )

            if self.use_gpu:
                last_timesteps = torch.tensor(data=last_timesteps).cuda()      # (B)
            else:
                last_timesteps = torch.tensor(data=last_timesteps)             # (B)
            relative_hidden_size = self.hidden_size*2 if self.use_bidirectional else self.hidden_size
            last_timesteps = last_timesteps.repeat(1, relative_hidden_size)    # (1, B x H*)
            last_timesteps = last_timesteps.view(-1, 1, relative_hidden_size)  # (B, 1, H*)
            pooled_sequence_output = sequence_output.gather(                   # (B, H*)
                dim=1,
                index=last_timesteps
            ).squeeze()

            pooled_sequence_output = self.dropout(pooled_sequence_output)      # (B, H*)
            logits = self.clf(pooled_sequence_output)                          # (B, num_classes)
        else:
            if not self.aggregate_on_cls_token:
                pooled_output = self.flatten_sequence_length(sequence_output)  # (B, P x H)

            pooled_output = self.dropout(pooled_output)                        # (B, P x H OR H)
            logits = self.clf(pooled_output)                                   # (B, num_classes)

        return logits                                                          # (B, num_classes)
