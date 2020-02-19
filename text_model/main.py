
"""
Code adapted from https://github.com/uzaymacar/comparatively-finetuning-bert
"""

from bert_utils import *


def main():

    logging.getLogger('pytorch_transformers').setLevel(logging.CRITICAL)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE FOUND: %s" % DEVICE)

    files = ['1','2','3','4','5','6','7','8','9','10']
    #file_bert = open('./bert_smalltobacco_results_' + '6_heads' +  '.csv', "w")
    #writer_bert = csv.writer(file_bert, delimiter=',')


    for file_number in files:
        hdf5_file = scratch_path + 'HDF5_files/hdf5_small_tobacco_papers_audebert_' + file_number + '.hdf5'
        print('File number: ', file_number)
        results_file = []

        NUM_CLASSES = 10
        NUM_EPOCHS = 1
        BATCH_SIZE = 6
        PRETRAINED_MODEL_NAME = 'bert-base-uncased'#'bert-base-cased'
        NUM_PRETRAINED_BERT_LAYERS = 6
        MAX_TOKENIZATION_LENGTH = 512

        TOP_DOWN = True
        NUM_RECURRENT_LAYERS = 0
        HIDDEN_SIZE = 768
        REINITIALIZE_POOLER_PARAMETERS = False
        USE_BIDIRECTIONAL = False
        DROPOUT_RATE = 0.2
        AGGREGATE_ON_CLS_TOKEN = True
        CONCATENATE_HIDDEN_STATES = False

        APPLY_CLEANING = False
        TRUNCATION_METHOD = 'head+tail'#head-only
        NUM_WORKERS = 0

        BERT_LEARNING_RATE = 3e-5
        CUSTOM_LEARNING_RATE = 1e-3
        BETAS = (0.9, 0.999)
        BERT_WEIGHT_DECAY = 0.01
        EPS = 1e-8


        bert_model = FineTunedBert(pretrained_model_name=PRETRAINED_MODEL_NAME,
                              num_pretrained_bert_layers=NUM_PRETRAINED_BERT_LAYERS,
                              max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                              num_classes=NUM_CLASSES,
                              top_down=TOP_DOWN,
                              num_recurrent_layers=NUM_RECURRENT_LAYERS,
                              use_bidirectional=USE_BIDIRECTIONAL,
                              hidden_size=HIDDEN_SIZE,
                              reinitialize_pooler_parameters=REINITIALIZE_POOLER_PARAMETERS,
                              dropout_rate=DROPOUT_RATE,
                              aggregate_on_cls_token=AGGREGATE_ON_CLS_TOKEN,
                              concatenate_hidden_states=CONCATENATE_HIDDEN_STATES,
                              use_gpu=True if torch.cuda.is_available() else False)

        train_dataset = H5Dataset(path=hdf5_file,
                                    tokenizer=bert_model.get_tokenizer(),
                                    apply_cleaning=APPLY_CLEANING,
                                    max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                                    truncation_method=TRUNCATION_METHOD,
                                    device=DEVICE,
                                    phase= 'train')

        test_dataset = H5Dataset(path=hdf5_file,
                                    tokenizer=bert_model.get_tokenizer(),
                                    apply_cleaning=APPLY_CLEANING,
                                    max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                                    truncation_method=TRUNCATION_METHOD,
                                    device=DEVICE,
                                    phase= 'test')

        val_dataset = H5Dataset(path=hdf5_file,
                                    tokenizer=bert_model.get_tokenizer(),
                                    apply_cleaning=APPLY_CLEANING,
                                    max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                                    truncation_method=TRUNCATION_METHOD,
                                    device=DEVICE,
                                    phase= 'val')

        # Acquire iterators through data loaders
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS)

        val_loader = DataLoader(dataset=val_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS)

        criterion = nn.CrossEntropyLoss()

        # Define identifiers & group model parameters accordingly (check README.md for the intuition)
        bert_identifiers = ['embedding', 'encoder', 'pooler']
        no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']
        grouped_model_parameters = [
                {'params': [param for name, param in bert_model.named_parameters()
                            if any(identifier in name for identifier in bert_identifiers) and
                            not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
                 'lr': BERT_LEARNING_RATE,
                 'betas': BETAS,
                 'weight_decay': BERT_WEIGHT_DECAY,
                 'eps': EPS},
                {'params': [param for name, param in bert_model.named_parameters()
                            if any(identifier in name for identifier in bert_identifiers) and
                            any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
                 'lr': BERT_LEARNING_RATE,
                 'betas': BETAS,
                 'weight_decay': 0.0,
                 'eps': EPS},
                {'params': [param for name, param in bert_model.named_parameters()
                            if not any(identifier in name for identifier in bert_identifiers)],
                 'lr': CUSTOM_LEARNING_RATE,
                 'betas': BETAS,
                 'weight_decay': 0.0,
                 'eps': EPS}
        ]

        optimizer = AdamW(grouped_model_parameters)

        bert_model, criterion = bert_model.to(DEVICE), criterion.to(DEVICE)

        best_val_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            print("EPOCH NO: %d" % (epoch + 1))
            since = time.time()

            train_loss, train_corrected = train(model=bert_model,
                                          iterator=train_loader,
                                          criterion=criterion,
                                          optimizer=optimizer,
                                          device=DEVICE,
                                          include_bert_masks=True)

            val_loss, val_corrected, conf_mat_val = test(model=bert_model,
                                       iterator=val_loader,
                                       criterion=criterion,
                                       device=DEVICE,
                                       include_bert_masks=True)



            time_elapsed = time.time()-since
            print('Time', time_elapsed)

            train_acc = train_corrected/len(train_loader.dataset)
            val_acc = val_corrected/len(val_loader.dataset)

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_model_wts = copy.deepcopy(bert_model.state_dict())
            #     #torch.save(bert_model.state_dict(), 'saved_models/bert_300k_best.pt')

            print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
            print(f'\tValidation Loss: {val_loss:.3f} | Validation Accuracy: {val_acc * 100:.2f}%')

        #Test

        test_loss, test_corrected, conf_mat_test = test(model=bert_model,
                                   iterator=test_loader,
                                   criterion=criterion,
                                   device=DEVICE,
                                   include_bert_masks=True)

        test_acc = test_corrected/len(test_loader.dataset)

        print('Number of corrected predictions', test_corrected)
        print('Test length', len(test_loader.dataset))


        print(conf_mat_test)
        print(conf_mat_test.diag()/conf_mat_test.sum(1))

        print(f'\tTest Loss: {test_loss:.3f} | Test Accuracy: {test_acc * 100:.2f}%')

        results_file.append(file_number)
        results_file.append(test_acc * 100)
        #writer_bert.writerow(results_file)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force = 'True')
    main()
