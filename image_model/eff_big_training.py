from eff_utils import *

def func():
    number_gpus = torch.cuda.device_count()
    print("GPUs visible to PyTorch", number_gpus, "GPUs")

    parser = argparse.ArgumentParser()

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

    description = 'eff' + efficientnet_model + 'BT' + str(number_gpus) + 'gpus_paper_exp_20_16'

    max_epochs = args.epochs
    batch_size = 16*number_gpus if int(efficientnet_model[1]) < 5 else 8*number_gpus
    big_tobacco_classes = 16
    lr_multiplier = 0.2
    learning_rate = (lr_multiplier*batch_size)/256
    number_workers = 4*number_gpus
    triangular_lr = True
    input_size = 384# not needed to resize since images are already 384x384

    """
    :param batch_size: batch size (16 for B0,B1,B2; 8 for B3,B4)
    :param big_tobacco_classes: number of classes in BigTobacco dataset (16)
    :param triangular_lr: uses triangular learning rate STLR
    """

    scratch_path = args.load_path

    # csv_file = open(scratch_path + '/image_models/paper_experiments/' + description + '.csv', "w")
    # writer = csv.writer(file_bert, delimiter=',')
    save_path = scratch_path + '/image_models/paper_experiments/' + str(max_epochs) + str(learning_rate) + description + '.pt'

    hdf5_file = scratch_path + '/BigTobacco_images_'
    print('batch size: ',batch_size)
    print('number of workers: ',number_workers)
    print('max_epochs', max_epochs)
    print('efficientnet_model', efficientnet_model)

    # dataset creation (train and validation)
    documents_datasets = {x: H5.H5Dataset(path=str(hdf5_file + x + '.hdf5'),
            data_transforms=data_transforms[x], phase = x) for x in ['train', 'val']}
    # dataloader creation (train and validation)
    dataloaders_dict = {x: DataLoader(documents_datasets[x],
            batch_size=batch_size if x == 'train' else int(batch_size/(number_gpus*2)),
            shuffle=False, num_workers=number_workers, pin_memory=True)
            for x in ['train', 'val']}

    feature_extracting = True

    # loading EfficientNet with Linear layer on top
    net = EfficientNet.from_pretrained('efficientnet-' + efficientnet_model, num_classes=1000)
    num_ftrs = net._fc.in_features
    net._fc = nn.Linear(num_ftrs, big_tobacco_classes)
    model = net

    # move model to gpu
    #model = nn.DataParallel(model)
    model = model.to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
        momentum=0.9, weight_decay=0.00004)

    batches_per_epoch = len(dataloaders_dict['train'].dataset)/batch_size

    # learning rate scheduler
    if triangular_lr == True:
        scheduler = SlantedTriangular(optimizer,num_epochs=max_epochs,
            num_steps_per_epoch=batches_per_epoch,gradual_unfreezing=False)
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

    loss_values = []
    epoch_values = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    initial_train_time = 0
    initial_val_time = 0

    # training loop
    for epoch in range(max_epochs):
        results_file = []
        since = time.time()
        correct = 0
        for phase in ['train', 'val']:
            # time measure
            if phase == 'train':
                initial_train_time = time.time()
                if epoch >= 1:
                    end_validation_time = time.time() - initial_val_time
                    print('Validation time: ',end_validation_time)
                model.train()  # Set model to training mode
            else:
                end_train_time = time.time() - initial_train_time
                print('Training time: ',end_train_time)
                initial_val_time = time.time()
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_counter = 0

            for local_batch in dataloaders_dict[phase]:

                batchs_number = len(dataloaders_dict[phase].dataset)/batch_size
                batch_counter += 1

                if batch_counter % 100==0:
                    print(batch_counter)

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
            print('[Epoch {}/{}] {} lr: {:.4f}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, max_epochs,phase, actual_lr , epoch_loss, epoch_acc))

            # save best model (lower validation accuracy)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())

            # #saves after epoch
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': epoch_loss,
            #     }, save_path)

            # # csv writing
            # if phase == 'val':
            #     results_file.append(epoch_loss)
            #     results_file.append(epoch_acc.item() * 100)
            #     results_file.append(end_train_time)
            #     writer.writerow(results_file)

        print()

        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        print()

    # dataset creation (test)
    h5_dataset_test = H5.H5Dataset(path=str(hdf5_file + 'test' + '.hdf5'),
        data_transforms=data_transforms['test'], phase = 'test')

    # dataloader creation (test)
    dataloader_test = DataLoader(h5_dataset_test,
        batch_size=int(batch_size/(number_gpus*2)),
        shuffle=False, num_workers=4)#=0 in inetractive job

    print('Testing...')
    model.eval()

    correct = 0
    max_size = 0
    confusion_matrix = torch.zeros(big_tobacco_classes, big_tobacco_classes)
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
    print('Accuracy: ', correct/len(dataloader_test.dataset))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force = 'True')
    print('GPU available', torch.cuda.is_available())
    device = torch.device("cuda:0")
    func()
