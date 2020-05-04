from eff_utils import *

import torch.multiprocessing as mp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main_worker(gpu, args):

    torch.manual_seed(SEED)

    # rank of the current GPU
    rank = args.nr * args.gpus + gpu

    number_gpus = torch.cuda.device_count()
    args.gpu = gpu

    dist.init_process_group(backend='nccl',init_method='env://',
                            world_size=args.world_size, rank=rank)

    efficientnet_model = args.eff_model



    max_epochs = args.epochs
    #batch_size = 16*number_gpus if int(efficientnet_model[1]) < 3 else 8*number_gpus
    batch_size = 16 if int(efficientnet_model[1]) < 5 else 8
    big_tobacco_classes = 16
    lr_multiplier = 0.2
    learning_rate = (lr_multiplier*batch_size*number_gpus)/256
    number_workers = 4
    triangular_lr = True
    input_size = 384# not needed to resize since images are already 384x384
    save_model = False
    create_csv = False

    description = 'eff' + efficientnet_model + 'BT' + str(number_gpus) + '_seed' + str(SEED) + '_epochs' + str(max_epochs) + 'pruebas_finales'

    """
    :param batch_size: batch size (16 for B0,B1,B2; 8 for B3,B4)
    :param big_tobacco_classes: number of classes in BigTobacco dataset (16)
    :param triangular_lr: uses triangular learning rate STLR
    """

    scratch_path = args.load_path

    if create_csv == True:
        csv_file = open(scratch_path + '/image_models/paper_modifications/' + description + '.csv', "w")
        writer = csv.writer(csv_file, delimiter=',')

    save_path = scratch_path + '/image_models/paper_experiments/' + str(max_epochs) + str(learning_rate) + description + '.pt'

    hdf5_file = scratch_path + '/BigTobacco_images_'
    print('batch size: ',batch_size)
    print('number of workers: ',number_workers)
    print('max_epochs', max_epochs)
    print('efficientnet_model', efficientnet_model)

    #cudnn.benchmark = True

    # dataset creation (train and validation)
    documents_datasets = {x: H5.H5Dataset(path=str(hdf5_file + x + '.hdf5'),
            data_transforms=data_transforms[x], phase = x) for x in ['train', 'val']}

    # non-overlapping dataset partition
    sampler = {x: torch.utils.data.distributed.DistributedSampler(documents_datasets[x],
                                                                num_replicas=args.world_size,
                                                                rank=rank) for x in ['train', 'val']}

    # dataloader creation (train and validation)
    dataloaders_dict = {x: DataLoader(documents_datasets[x],
            batch_size=batch_size if x == 'train' else int(batch_size/(number_gpus*2)),
            shuffle = False, num_workers=number_workers, pin_memory=True,
            sampler = sampler[x])
            for x in ['train', 'val']}

    # loading EfficientNet with Linear layer on top
    net = EfficientNet.from_pretrained('efficientnet-' + efficientnet_model, num_classes=1000)
    num_ftrs = net._fc.in_features
    net._fc = nn.Linear(num_ftrs, big_tobacco_classes)
    model = net


    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # Wrap the model as a DistributedDataParallel model in device_id
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[gpu])

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
        momentum=0.9, weight_decay=0.00004)

    batches_per_epoch = len(dataloaders_dict['train'].dataset)/batch_size
    #batches_per_epoch = (len(dataloaders_dict['train'].dataset)/args.gpus)/batch_size

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

                batchs_number = (len(dataloaders_dict[phase].dataset)/args.gpus)/batch_size
                batch_counter += 1

                if batch_counter % 100==0 and gpu == 0:
                    print(batch_counter)

                # load image/class, transform them to tensor and load to GPU
                image, labels = Variable(local_batch['image']), Variable(local_batch['class'])
                image, labels = image.to(gpu), labels.to(gpu)


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

            epoch_loss = running_loss / (len(dataloaders_dict[phase].dataset)/args.gpus)
            epoch_acc = running_corrects.double() / (len(dataloaders_dict[phase].dataset)/args.gpus)

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
            if gpu == 0:
                print('[Epoch {}/{}] {} lr: {:.4f}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, max_epochs,phase, actual_lr , epoch_loss, epoch_acc))

            # save best model (lower validation accuracy)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())

            #saves after epoch
            if gpu == 0 and save_model == True:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    }, save_path)

            # csv writing
            if phase == 'val' and gpu == 0 and create_csv == True:
                results_file.append(epoch_loss)
                results_file.append(epoch_acc.item() * 100)
                results_file.append(end_train_time)
                results_file.append(actual_lr)
                writer.writerow(results_file)

        if gpu == 0:
            print()

            time_elapsed = time.time() - since
            print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            print()

    if gpu == 0:
        # dataset creation (test)
        h5_dataset_test = H5.H5Dataset(path=str(hdf5_file + 'test' + '.hdf5'),
            data_transforms=data_transforms['test'], phase = 'test')

        # dataloader creation (test)
        dataloader_test = DataLoader(h5_dataset_test,
            batch_size=int(batch_size/(number_gpus*2)),
            shuffle=False, num_workers=4)

        print('Testing...')
        model.eval()

        correct = 0
        max_size = 0
        confusion_matrix = torch.zeros(big_tobacco_classes, big_tobacco_classes)
        with torch.no_grad():
            for i, local_batch in enumerate(dataloader_test):
                image, labels = Variable(local_batch['image']), Variable(local_batch['class'])
                image, labels = image.to(gpu), labels.to(gpu)
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




def main():
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
        help="EfficientNet model to be used",
    )

    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()

    ngpus_per_node = number_gpus

    args.world_size = args.gpus * args.nodes
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=args.gpus, args=(args,))



if __name__ == '__main__':
    # allow multiprocessing
    torch.multiprocessing.set_start_method('spawn',force = 'True')
    print('GPU available', torch.cuda.is_available())
    main()
