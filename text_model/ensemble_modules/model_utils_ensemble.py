
"""
Code adapted from https://github.com/uzaymacar/comparatively-finetuning-bert
"""

import logging
import torch
from data_utils import get_features
import torch.nn as nn
import csv
import numpy
import re

def test_ensemble(bert_model, image_model, iterator, device, include_bert_masks=True):
    """
    Function to carry out the testing (or validation) process

    @param (torch.nn.Module) model: model object to be trained
    @param (torch.utils.data.DataLoader) iterator: data loader to iterate over batches
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    @param (bool) include_bert_masks: whether to include token type IDs & attention masks alongside
           input IDs when passing to model or not (default: True)
    """
    bert_model.eval()
    image_model.eval()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_loss, epoch_acc = 0.0, 0.0

    confusion_matrix = torch.zeros(10, 10)
    number_corrected = 0.0
    epoch_corrected = 0.0

    with torch.no_grad():
        counta = 0

        for batch in iterator:
            # Get testing input IDs & labels from the current batch
            image, input_ids, labels = batch
            image, input_ids, labels = image.to(DEVICE), input_ids.to(DEVICE), labels.to(DEVICE)
            # Get corresponding additional features from the current batch
            token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                          tokenizer=bert_model.get_tokenizer(),
                                                          device=device)
            # Pass features through the model w/ or w/o BERT masks for attention & token type
            #if include_bert_masks:
            outputsA = bert_model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        # else:
            #     predictions = image_model(input_ids=input_ids)

            #print(image)


            outputsB = image_model(image)
            #print(outputsB)
            #print(outputsB)

            smax = nn.Softmax(dim=1)


            prob_outA = smax(outputsA)
            valueA, predsA = torch.max(prob_outA.data, 1)


            prob_outB = smax(outputsB)
            valueB, predsB = torch.max(prob_outB.data, 1)


            prob_a_list = prob_outA.tolist()
            prob_b_list = prob_outB.tolist()
            new_row = []


            final_preds = 0.5*prob_outA + 0.5*prob_outB
            values, preds = torch.max(final_preds.data, 1)#values is prob

            # Calculate loss and accuracy
            for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    if t.long() == p.long():
                        number_corrected += 1



            #number_corrected, conf_mat = binary_accuracy(preds, labels, confusion_matrix)

            #epoch_loss += loss.item()
            #epoch_acc += acc.item()

            #epoch_corrected += number_corrected

    return number_corrected, confusion_matrix
