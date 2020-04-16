
"""
Code adapted from https://github.com/uzaymacar/comparatively-finetuning-bert
"""

import logging
import torch
from data_utils import get_features

num_classes = 10

def binary_accuracy(y_pred, y_true, confusion_matrix):
    """Function to calculate binary accuracy per batch"""
    corrected = 0
    y_pred_max = torch.argmax(y_pred, dim=-1)
    #correct_pred = (y_pred_max == y_true).float()
    # print(y_pred_max)
    # print(y_true)
    # print(correct_pred)
    # print(correct_pred.sum())
    # print()


    #acc = correct_pred.sum() / len(correct_pred)

    for t, p in zip(y_true.view(-1), y_pred_max.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
        if t.long() == p.long():
            corrected += 1
    #print(confusion_matrix)

    return corrected, confusion_matrix


def train(model, iterator, criterion, optimizer, device, include_bert_masks=True):
    """
    Function to carry out the training process

    @param (torch.nn.Module) model: model object to be trained
    @param (torch.utils.data.DataLoader) iterator: data loader to iterate over batches
    @param (torch.nn.[...]) criterion: loss function to backpropagate on
    @param (torch.optim.[...]) optimizer: optimization algorithm
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    @param (bool) include_bert_masks: whether to include token type IDs & attention masks alongside
           input IDs when passing to model or not (default: True)
    """
    epoch_loss, epoch_acc = 0.0, 0.0
    counter = 0
    epoch_corrected = 0.0
    number_corrected = 0.0



    confusion_matrix = torch.zeros(num_classes, num_classes)

    for batch in iterator:
        counter += 1
        # Get training input IDs & labels from the current batch
        input_ids, labels = batch
        # Get corresponding additional features from the current batch
        token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                      tokenizer=model.get_tokenizer(),
                                                      device=device)
        #print(token_type_ids)
        if counter%50 == 0:
            print(counter)
        # Reset the gradients from previous processes
        optimizer.zero_grad()
        # Pass features through the model w/ or w/o BERT masks for attention & token type
        if include_bert_masks:
            predictions = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        else:
            predictions = model(input_ids=input_ids)

        # Calculate loss and accuracy
        loss = criterion(predictions, labels)
        # print(predictions)
        # print(labels)
        number_corrected, conf_mat = binary_accuracy(predictions, labels, confusion_matrix)
        #acc = 1.0

        # Backward and optimize
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        #epoch_acc += acc.item()
        #epoch_acc += acc
        epoch_corrected += number_corrected

    print('counter',counter)
    return epoch_loss / len(iterator), epoch_corrected


def test(model, iterator, criterion, device, include_bert_masks=True):
    """
    Function to carry out the testing (or validation) process

    @param (torch.nn.Module) model: model object to be trained
    @param (torch.utils.data.DataLoader) iterator: data loader to iterate over batches
    @param (torch.nn.[...]) criterion: loss function to backpropagate on
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    @param (bool) include_bert_masks: whether to include token type IDs & attention masks alongside
           input IDs when passing to model or not (default: True)
    """
    epoch_loss, epoch_acc = 0.0, 0.0

    confusion_matrix = torch.zeros(num_classes, num_classes)
    number_corrected = 0.0
    epoch_corrected = 0.0

    with torch.no_grad():
        for batch in iterator:
            # Get testing input IDs & labels from the current batch
            input_ids, labels = batch
            # Get corresponding additional features from the current batch
            token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                          tokenizer=model.get_tokenizer(),
                                                          device=device)
            # Pass features through the model w/ or w/o BERT masks for attention & token type
            if include_bert_masks:
                predictions = model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
            else:
                predictions = model(input_ids=input_ids)

            # Calculate loss and accuracy
            loss = criterion(predictions, labels)
            number_corrected, conf_mat = binary_accuracy(predictions, labels, confusion_matrix)

            epoch_loss += loss.item()
            #epoch_acc += acc.item()

            epoch_corrected += number_corrected

    return epoch_loss / len(iterator), epoch_corrected, conf_mat
