from bert_help import *

import os.path
parent_path = os.path.abspath(os.path.join('./', os.pardir))
model_path = '/resources/taggers/small_tobacco/'

from models.imagenet import mobilenetv2
import nn_models as mod

from efficientnet_pytorch import EfficientNet

scratch_path = '/gpfs/scratch/bsc31/bsc31275/'

logging.getLogger('pytorch_transformers').setLevel(logging.CRITICAL)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE FOUND: %s" % DEVICE)

efficientnet_model = 'b1'

files = ['1','2','3','4','5','6','7','8','9','10']
# file_bert = open('./ensemble_smalltobacco_results_finetuning' + efficientnet_model +  '.csv', "w")
# writer_bert = csv.writer(file_bert, delimiter=',')

for file_number in files:
    hdf5_file = scratch_path + 'HDF5_files/hdf5_small_tobacco_papers_audebert_' + file_number + '.hdf5'
    print('File number: ', file_number)
    results_file = []

    print('File: ', file_number)
    print('Model: ', efficientnet_model)

    # Define hyperparameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 1
    PRETRAINED_MODEL_NAME = 'bert-base-uncased'#'bert-base-cased'
    NUM_PRETRAINED_BERT_LAYERS = 6
    MAX_TOKENIZATION_LENGTH = 512
    NUM_CLASSES = 10
    TOP_DOWN = True
    NUM_RECURRENT_LAYERS = 0
    HIDDEN_SIZE = 768
    REINITIALIZE_POOLER_PARAMETERS = False
    USE_BIDIRECTIONAL = False
    DROPOUT_RATE = 0.10
    AGGREGATE_ON_CLS_TOKEN = True
    CONCATENATE_HIDDEN_STATES = False

    APPLY_CLEANING = False
    TRUNCATION_METHOD = 'head+tail'
    NUM_WORKERS = 0

    BERT_LEARNING_RATE = 2e-5
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

    test_dataset = H5Dataset_ensemble(path=hdf5_file,
                                data_transforms=data_transforms['test'],
                                tokenizer=bert_model.get_tokenizer(),
                                apply_cleaning=APPLY_CLEANING,
                                max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                                truncation_method=TRUNCATION_METHOD,
                                device=DEVICE,
                                phase= 'test')

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)


    # Loading image model
    net = EfficientNet.from_pretrained('efficientnet-' + efficientnet_model, num_classes=1000)
    num_ftrs = net._fc.in_features
    net._fc = nn.Linear(num_ftrs, NUM_CLASSES)
    net = torch.nn.DataParallel(net)#comment if model doesn't come from BigTobacco
    PATH1 = scratch_path + '/image_models/paper_experiments/finetuning_BT/' + '50.025eff' + efficientnet_model + '_aud' + file_number + 'ft.pt'
    checkpoint = torch.load(PATH1)
    net.load_state_dict(checkpoint['model_state_dict'])

    # Loading text model
    bert_model.load_state_dict(torch.load(scratch_path + '/text_models' + '/bert_aud' + file_number + '.pt'))

    criterion = nn.CrossEntropyLoss()

    bert_model, criterion = bert_model.to(DEVICE), criterion.to(DEVICE)
    net = net.to(DEVICE)

    test_corrected, conf_mat_test = test_ensemble(bert_model=bert_model,
                                image_model=net,
                               iterator=test_loader,
                               criterion=criterion,
                               device=DEVICE,
                               include_bert_masks=True)

    test_acc = test_corrected/len(test_loader.dataset)

    print('Number of corrected predictions', test_corrected)
    print('Test length', len(test_loader.dataset))


    print(conf_mat_test)
    print(conf_mat_test.diag()/conf_mat_test.sum(1))

    print(f'\tTest Accuracy:  {test_acc * 100:.2f}%')

    # results_file.append(file_number)
    # results_file.append(test_acc * 100)
    # writer_bert.writerow(results_file)
