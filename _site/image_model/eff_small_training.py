from eff_utils import *

def func():
    number_gpus = torch.cuda.device_count()
    print("Let's use", number_gpus, "GPUs!")

    # Required parameters
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        required=True,
        help="Number of epochs in the training process. Should be an integer number",
    )

    parser.add_argument(
        "--eff_model",
        default='b0',
        type=str,
        required=True,
        help="EfficientNet model used b0, b1, b2, b3 or b4",
    )

    parser.add_argument(
        "--load_path",
        default='/gpfs/scratch/bsc31/bsc31275/',
        type=str,
        required=True,
        help="EfficientNet model used b0, b1, b2, b3 or b4",
    )

    args = parser.parse_args()

    efficientnet_model = args.eff_model

    files = ['1','2','3','4','5','6','7','8','9','10']
    # csv_file = open( scratch_path + '/image_models/paper_experiments/scratch/' +'smalltobacco_results_scratch_' + efficientnet_model +  '.csv', "w")
    # writer = csv.writer(csv_file, delimiter=',')

    for file_number in files:
        results_file = []

        description = 'eff' + efficientnet_model + '_aud' + file_number

        max_epochs = args.epochs
        batch_size = 8*number_gpus
        num_workers = 0*number_gpus #1 GPU used
        from_bigtobacco = False
        learning_rate = (0.8*batch_size)/256
        batches_per_epoch = 800/batch_size
        triangular_lr = True
        input_size = 384#N.Audebert et al ->384
        feature_extracting = False
        save = False
        small_tobacco_classes = 10
        big_tobacco_classes = 16

        print('File number: ',file_number)

        """
        :param batch_size: batch size (16 for B0,B1,B2; 8 for B3,B4)
        :param small_tobacco_classes: number of classes in SmallTobacco dataset (10)
        :param from_bigtobacco: (boolean) model loaded pretrained in BigTobacco
        :param feature_extracting: (boolean) to freeze model weights or not
        """

        scratch_path = args.load_path

        save_path = scratch_path + '/image_models/paper_experiments/scratch/' + str(max_epochs) + str(learning_rate) + description + '.pt'
        hdf5_file = scratch_path + 'HDF5_files/hdf5_small_tobacco_papers_audebert_' + file_number + '.hdf5'

        print('lr = {}, batch_size = {}, image_size = {}, {}'.format(learning_rate, batch_size, input_size, description))

        # dataset creation (train and validation)
        documents_datasets = {x: H5.H5Dataset(path=hdf5_file,
            data_transforms=data_transforms[x], phase = x) for x in ['train', 'val']}
        # dataloader creation (train and validation)
        dataloaders_dict = {x: DataLoader(documents_datasets[x],
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True) for x in ['train', 'val']}

        # loading EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-' + efficientnet_model, num_classes=1000)
        num_ftrs = net._fc.in_features

        PATH_best = scratch_path + '/image_models/' + str(max_epochs) + str(learning_rate) + description + 'best.pt'

        #Fine-tune from BigTobacco or from scratch
        if from_bigtobacco == True:
            net._fc = nn.Linear(num_ftrs, big_tobacco_classes)
            net = torch.nn.DataParallel(net)
            PATH1 = scratch_path + 'image_models/paper_experiments/efficientnets/200.05effb0BT4gpus.pt'
            checkpoint = torch.load(PATH1)
            net.load_state_dict(checkpoint['model_state_dict'])
            set_parameter_requires_grad(net, feature_extracting)
            num_ftrs = net.module._fc.in_features
            net.module._fc = nn.Linear(num_ftrs, small_tobacco_classes)
        else:
            net._fc = nn.Linear(num_ftrs, small_tobacco_classes)

        # move model to gpu
        model = net
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
            momentum=0.9, weight_decay=0.00004)

        # learning rate scheduler
        if triangular_lr == True:
            scheduler = SlantedTriangular(optimizer,num_epochs=max_epochs,
                num_steps_per_epoch=batches_per_epoch,
                discriminative_fine_tuning=False,gradual_unfreezing=False)
        else:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

        loss_values = []
        epoch_values = []
        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # training loop
        for epoch in range(max_epochs):
            since = time.time()
            correct = 0

            for phase in ['train','val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    initial_train_time = time.time()
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                batch_counter = 0

                for local_batch in dataloaders_dict[phase]:
                    batchs_number = len(dataloaders_dict[phase].dataset)/batch_size
                    batch_counter += 1

                    # load image/class, transform them to tensor and load to GPU
                    image, labels = Variable(local_batch['image']), Variable(local_batch['class'])
                    image, labels = image.to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # model output
                    outputs = model(image)
                    # greatest class value output by the model
                    _, preds = torch.max(outputs.data, 1)

                    labels = labels.long()
                    loss = criterion(outputs, labels)

                    # correct predictions
                    running_corrects += torch.sum(preds == labels.data)

                    # backward + optimize if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if triangular_lr == True:
                            scheduler.step_batch()

                    running_loss += loss.item()

                epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

                if phase == 'train':
                    train_loss = epoch_loss
                if phase == 'val':
                    val_loss = epoch_loss

                if phase == 'val':
                    #scheduler.step(epoch_loss) #use this for ReduceLROnPlateau scheduler
                    if triangular_lr == True:
                        scheduler.step()
                    else:
                        scheduler.step(epoch_loss)

                actual_lr = get_lr(optimizer)
                print('[Epoch {}/{}] {} lr: {:.8f}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, max_epochs,phase, actual_lr , epoch_loss, epoch_acc))

                # save best model (lower validation accuracy)
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(model.state_dict())

                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            #saves after epoch
            if save == True:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    }, save_path)

            print()

            time_elapsed = time.time() - since
            print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            print()

        # dataset creation (test)
        h5_dataset_test = H5.H5Dataset(path=hdf5_file, data_transforms=data_transforms['test'], phase = 'test')
        # dataloader creation (test)
        dataloader_test = DataLoader(h5_dataset_test, batch_size=1, shuffle=False, num_workers=0)#=0 in inetractive job


        print('Testing...')
        model.eval()

        correct = 0
        max_size = 0
        confusion_matrix = torch.zeros(small_tobacco_classes, small_tobacco_classes)
        with torch.no_grad():
            for i, local_batch in enumerate(dataloader_test):
                image, labels = Variable(local_batch['image']), Variable(local_batch['class'])
                image, labels = image.to(device), labels.to(device)
                outputs = model(image)
                _, preds = torch.max(outputs.data, 1)

                # confusion_matrix
                for t, p in zip(labels.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                        if t.long() == p.long():
                            correct += 1

        print(confusion_matrix)

        print(confusion_matrix.diag()/confusion_matrix.sum(1))
        accuracy = correct/len(dataloader_test.dataset)
        print('Accuracy: ', accuracy)

        # results_file.append(file_number)
        # results_file.append(accuracy*100)
        # writer.writerow(results_file)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force = 'True')
    print('GPU available', torch.cuda.is_available())
    device = torch.device("cuda:0")
    func()
