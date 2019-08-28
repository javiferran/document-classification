import torch.nn as nn
import torch.nn.functional as F
import torch

import flair.nn
import flair.embeddings
from typing import List, Union
from flair.embeddings import Sentence

class Fully_Connected(nn.Module):

    def __init__(self,embedding_dim):
        super(Fully_Connected, self).__init__()

        input_linear = embedding_dim
        C = 10

        self.fc1 = nn.Linear(input_linear, C)

    def forward(self, x):

        logit = self.fc1(x)

        return logit

class Concatenation(nn.Module):

    def __init__(self, text_model, image_model, input_dim):
        super(Concatenation, self).__init__()

        self.text_model = text_model
        self.image_model = image_model

        input_linear = input_dim
        C = 10

        self.fc1 = nn.Linear(input_linear, C)

    def forward(self, x, x2):

        x = self.text_model(x)

        x2 = self.image_model(x2)

        x = torch.cat((x,x2),1)


        logit = self.fc1(x)

        return logit

class CNN_Text_Image(nn.Module):

    def __init__(self, image_model, embedding_dim):
        super(CNN_Text_Image, self).__init__()
        #self.args = args

        #V = args.embed_num
        D = embedding_dim #embed_dim, 4196 for doc_embeddings
        C = 10 #class_num
        Ci = 1
        Co = 100 #kernel_num -> number of kernel with the same size
        Ks = [3,4,5] #kernel_sizes -> size = number of words

        #self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])


        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc1 = nn.Linear(600, C)

        self.image_model = image_model

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        #Output will be size (1,Ks*Co) -> Maxpool will get one ĉ value =  max(c_1,c_2...), where c_i is
        #the result of the convolution operation of the kernel over the input

        return x



    def forward(self, x, x2):
        #x = self.embed(x)  # (N, W, D)

        #if self.args.static:
            #x = Variable(x)
        #print('CNN Text entry',x.shape)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        #print('unsqueeze',x.shape)


        #print(x.shape)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)


        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)


        x = torch.cat(x, 1) #[1,100] + [1,100] + [1,100] = [1,300]

        #print('After cat', x.shape)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)

        x2 = self.image_model(x2)

        x2 = torch.cat((x,x2),1)


        logit = self.fc1(x2)

        return logit

class CNN_Text(nn.Module):

    def __init__(self, embedding_dim):
        super(CNN_Text, self).__init__()
        #self.args = args

        #V = args.embed_num
        D = embedding_dim #embed_dim, 4196 for doc_embeddings
        C = 10 #class_num
        Ci = 1
        Co = 100 #kernel_num -> number of kernel with the same size
        Ks = [10,11,12] #kernel_sizes -> size = number of words

        #self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])


        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(len(Ks)*Co, C)
        convs_out = len(Ks)*Co
        self.fc1 = nn.Linear(convs_out, C)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        #Output will be size (1,Ks*Co) -> Maxpool will get one ĉ value =  max(c_1,c_2...), where c_i is
        #the result of the convolution operation of the kernel over the input

        return x



    def forward(self, x):
        #x = self.embed(x)  # (N, W, D)

        #if self.args.static:
            #x = Variable(x)
        #print('CNN Text entry',x.shape)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        #print('unsqueeze',x.shape)


        #print(x.shape)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)


        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)


        x = torch.cat(x, 1) #[1,100] + [1,100] + [1,100] = [1,300]

        #print('After cat', x.shape)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)


        logit = self.fc1(x)

        return logit

class TextClassifier(nn.Module):
    """
    Text Classification Model
    The model takes word embeddings, puts them into an RNN to obtain a text representation, and puts the
    text representation in the end into a linear layer to get the actual class label.
    The model can handle single and multi class data sets.
    """

    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        # label_dictionary: Dictionary,
        # multi_label: bool = None,
        # multi_label_threshold: float = 0.5,
    ):
        """
        Initializes a TextClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        """

        super(TextClassifier, self).__init__()

        self.document_embeddings: flair.embeddings.DocumentRNNEmbeddings = document_embeddings
        # self.label_dictionary: Dictionary = label_dictionary

        # if multi_label is not None:
        #     self.multi_label = multi_label
        # else:
        #     self.multi_label = self.label_dictionary.multi_label
        #
        # self.multi_label_threshold = multi_label_threshold

        self.decoder = nn.Linear(
            self.document_embeddings.embedding_length, 10
        )
        print(self.document_embeddings.embedding_length)
        self._init_weights()

        # auto-spawn on GPU if available
        self.to(flair.device)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, sentences) -> List[List[float]]:

        self.document_embeddings.embed(sentences)

        text_embedding_list = [
            sentence.get_embedding().unsqueeze(0) for sentence in sentences
        ]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)
        #print(text_embedding_tensor.shape)

        label_scores = self.decoder(text_embedding_tensor)

        #print(label_scores.shape)

        return label_scores
