import os
import numpy as np
from tqdm import tqdm
import time
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from glob import glob
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import accuracy_score

from early_stop_tool import EarlyStopping
from args_ours import Config

from models.model import SequenceMamba2
from DataProvider.Dataloader import SequenceSleepDataset
from DataProvider.DataGenerator import Fetch_filelist




def set_random_seed(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU


def evaluate(model, test_loader, config):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    pred = []
    label = []

    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.float().to(config.device)
            target = target.to(config.device)

            output = model(data)
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            test_loss += criterion(output, target.long()).item()

            pred.extend(np.argmax(output.data.cpu().numpy(), axis=1))
            label.extend(target.data.cpu().numpy())

        accuracy = accuracy_score(label, pred, normalize=True, sample_weight=None)

    return accuracy, test_loss


def load_data(root, filelist):
    filenames = []
    for sublist in filelist:
        if len(sublist) == 1:
            filenames.append(sublist[0])
        else:
            filenames.append(sublist[0])
            filenames.append(sublist[1])

    files = np.array(filenames)
    print("The Number of the files: ", len(files))

    first = True
    for file in files:
        filepath = os.path.join(root, file)
        data = np.load(filepath)

        if first:
            train_data = data['data']
            labels = data['label']
            first = False
        else:
            train = data['data']
            label = data['label']
            train_data = np.append(train_data, train, axis=0)
            labels = np.append(labels, label, axis=0)

    train_data = train_data.astype(np.float32)
    labels = labels.astype(np.float32)

    return train_data, labels


def train(save_all_checkpoint=False):

    seed = 42
    print('\n', '-' * 15, '>', 'Loading the Dataset', '<', '-' * 15)
    config = Config()

    root = './data/SleepEDF-78/EEG_EOG'
    model_name = 'SleepEDF'

    individual_list = Fetch_filelist(root=root)
    individual_list = individual_list
    filelist = list(individual_list)
    X = filelist

    kf = KFold(n_splits=config.num_fold, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print('\n', '-' * 15, '>', f'Fold {fold}', '<', '-' * 15)

        if not os.path.exists('./Kfold_models_{}/fold{}'.format(model_name, fold)):
            os.makedirs('./Kfold_models_{}/fold{}'.format(model_name, fold))
            print(model_name)

        train_individuals = [X[i] for i in train_idx]
        test_individuals = [X[i] for i in test_idx]

        train_files = [individual_list[ind] for ind in train_individuals]
        test_files = [individual_list[ind] for ind in test_individuals]

        print(train_files)
        print(test_files)

        X_train, train_label = load_data(root, train_files)
        X_test, test_label = load_data(root, test_files)

        print('train shape:', X_train.shape, train_label.shape)
        print('test shape:', X_test.shape, test_label.shape)

        train_set = SequenceSleepDataset(X_train, train_label, window=config.window)
        test_set = SequenceSleepDataset(X_test, test_label, window=config.window)

        train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=3, drop_last=True)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=3, drop_last=True)

        model = SequenceMamba2(config)
        model = model.to(config.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

        early_stopping = EarlyStopping(patience=config.patience, verbose=True, save_all_checkpoint=save_all_checkpoint)

        # evaluating indicator
        train_ACC = []
        train_LOSS = []
        test_ACC = []
        test_LOSS = []


        for epoch in range(config.num_epochs):
            running_loss = 0.0
            correct = 0

            model.train()

            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (data, target) in loop:

                data = data.float().to(config.device)
                target = target.to(config.device)

                optimizer.zero_grad()
                output = model(data)
                output = output.view(-1, output.size(-1))
                target = target.view(-1)

                loss = criterion(output, target.long())
                loss = loss / config.window

                loss.backward()

                optimizer.step()
                running_loss += loss.item()

                train_acc_batch = np.sum(
                    np.argmax(np.array(output.data.cpu()), axis=1) == np.array(target.data.cpu())) / (target.shape[0])
                loop.set_postfix(train_acc=train_acc_batch, loss=loss.item())
                correct += np.sum(np.argmax(np.array(output.data.cpu()), axis=1) == np.array(target.data.cpu()))


            running_loss = running_loss / len(train_loader)
            train_acc = (correct / len(train_loader.dataset)) / config.window
            test_acc, test_loss = evaluate(model, test_loader, config)
            test_loss = test_loss / config.window / len(test_loader)

            print('Epoch: ', epoch,
                  '| train loss: %.4f' % running_loss, '| train acc: %.4f' % train_acc,
                  '| test acc: %.4f' % test_acc, '| test loss: %.4f' % test_loss)

            train_ACC.append(train_acc)
            train_LOSS.append(running_loss)
            test_ACC.append(test_acc)
            test_LOSS.append(test_loss)

            early_stopping(test_acc, model,
                           path='./Kfold_models_{}/fold{}/model_{}_epoch{}.pkl'.format(model_name, fold, fold, epoch))

            if early_stopping.early_stop:
                print("Early stopping at epoch ", epoch)
                break

        np.save('./Kfold_models_{}/fold{}/train_LOSS.npy'.format(model_name, fold), np.array(train_LOSS))
        np.save('./Kfold_models_{}/fold{}/train_ACC.npy'.format(model_name, fold), np.array(train_ACC))
        np.save('./Kfold_models_{}/fold{}/test_LOSS.npy'.format(model_name, fold), np.array(test_LOSS))
        np.save('./Kfold_models_{}/fold{}/test_ACC.npy'.format(model_name, fold), np.array(test_ACC))

        del model


if __name__ == '__main__':
    set_random_seed(2024)
    train(save_all_checkpoint=False)
