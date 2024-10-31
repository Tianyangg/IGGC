import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from network.ImageMIL import ImageMIL
from utils.tools import draw_visdom
from utils.slide_selective_dataset import SelectiveDatasetMIL
from sklearn.metrics import classification_report


def calculate_acc(y_pred, y_gt, num_classes):
    '''calculate the mean AUC'''
    acc_num_each_class = np.zeros((num_classes))
    num_each_class = np.zeros((num_classes))

    for k, label in enumerate(y_gt):
        num_each_class[label] += 1
        if y_pred[k] == label:
            acc_num_each_class[label] += 1

    return acc_num_each_class.sum() / num_each_class.sum(), acc_num_each_class / num_each_class


def loadLatestCheckpoint(ckpt_dir, net, optimizer):
    ''' Identify the lateset checkpoint, load parameters and return the last checkpoint index

    :param ckpt_dir:
    :param net:
    :param optimizer:
    :return:
    '''
    checkpoint_file_list = glob.glob(os.path.join(ckpt_dir, 'ckpt_*.pth'))

    if not checkpoint_file_list:
        print("No checkpoint files are found.Start training from scratch")
        return 0

    checkpoint_file_list.sort()
    labtest_checkpoint_file = checkpoint_file_list[-1]
    print("loading checkpoint: {}".format(labtest_checkpoint_file))
    # load checkpoint
    ckpt = torch.load(labtest_checkpoint_file)
    net.load_state_dict(ckpt['net'])
    name_split = os.path.basename(labtest_checkpoint_file).replace(".pth", "").split("_")
    return int(name_split[-1]) + 1


# train_csv, val_csv, model_save_path
def train(train_csv, val_csv, num_of_class=5, model_save_path=None, ft=None, target="glioma"):
    viz = visdom.Visdom()

    model = ImageMIL(n_class=num_of_class, n_head=1)
    model = nn.parallel.DataParallel(model, device_ids=[0]).cuda()
    
    epoch_start = 0

    if ft is not None:
        epoch_start = loadLatestCheckpoint(ft, model, optimizer=None)

    batch_size = 1
    loader_kwargs = {'num_workers': 8, 'pin_memory': True}
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    baseline_acc = 0.0

    print('Load Train and Test Set')
    train_dataset = SelectiveDatasetMIL(train_csv=train_csv, 
                                    target=target, 
                                    select_number=4, 
                                    istrain=True)
    
    test_datset =   SelectiveDatasetMIL(train_csv=val_csv, 
                                    target=target,
                                    select_number=4,
                                    istrain=False)
    
    train_loader = data_utils.DataLoader(train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        **loader_kwargs)
                                        
    test_loader = data_utils.DataLoader(test_datset, 
                                        batch_size=batch_size, 
                                        shuffle=False,
                                        **loader_kwargs)
    
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(epoch_start, 10000):
        model.train()
        
        losses = 0
        labels = np.zeros(1)
        preds = np.zeros(1)
        for batch_idx, (subID, image_stack, sublabel) in enumerate(train_loader):
            optimizer.zero_grad()

            image_stack, sublabel = image_stack.cuda(), sublabel.cuda()
            image_stack, sublabel = Variable(image_stack), Variable(sublabel)

            y_prob, _ = model(image_stack)
            loss = loss_func(y_prob, sublabel.long())
            loss.backward()
            optimizer.step()
            
            losses += float(loss.data)
            predicted = torch.max(y_prob.data, 1)[1]
            labels = np.concatenate([labels,np.array(sublabel.detach().cpu())])
            preds = np.concatenate([preds,np.array(predicted.detach().cpu())])
            
            if batch_idx % 30 == 0:
                print('[%i/%i]'%(batch_idx * batch_size, len(train_dataset)))
        
        train_losses = losses / len(train_loader)
        correct = (np.array(preds) == np.array(labels)).sum()
        train_accs = correct / len(train_dataset)
        print('Train Set, Loss: {:.4f}, Train accuracy: {:.4f}'.format(train_losses, train_accs))

        epoch_acc_train, acc_each_class_train = calculate_acc(np.array(preds).astype(int), np.array(labels).astype(int), num_of_class)
        print("Training classification report: \n", classification_report(np.array(labels), np.array(preds)))
        

        with torch.no_grad():
            model.eval()
            test_losses = 0
            test_labels = np.zeros(1)
            test_preds = np.zeros(1)
            for batch_idx, (subID, image_stack, sublabel) in enumerate(test_loader):

                image_stack, sublabel = image_stack.cuda(), sublabel.cuda()
                image_stack, sublabel = Variable(image_stack), Variable(sublabel)

                y_prob, _ = model(image_stack)

                # calculate loss and metrics
                loss = loss_func(y_prob, sublabel.long())
                test_loss = loss.data
                test_losses += float(test_loss)

                predicted = torch.max(y_prob.data, 1)[1]
                test_labels = np.concatenate([test_labels, np.array(sublabel.detach().cpu())])
                test_preds = np.concatenate([test_preds, np.array(predicted.detach().cpu())])

                if batch_idx % 30 == 0:
                    print('[%i/%i]' % (batch_idx * batch_size, len(test_datset)))

            test_losses = test_losses / len(test_loader)
            correct = (np.array(test_preds) == np.array(test_labels)).sum()
            test_accs = correct / len(test_datset)


            epoch_acc_test, acc_each_class_test = calculate_acc(np.array(test_preds).astype(int), np.array(test_labels).astype(int), num_of_class)

            print("Testing classification report: \n", classification_report(np.array(test_labels), np.array(test_preds)))
            
            print('Test Set, Loss: {:.4f}, Test accuracy: {:.4f}'.format(test_losses, test_accs))
            
            train_loss_list.append(train_accs)
            val_loss_list.append(test_accs)

            draw_visdom(viz, train_x=list(range(epoch_start, epoch + 1)), train_loss=train_loss_list, 
                        val_x=list(range(epoch_start, epoch + 1)), val_loss=val_loss_list, win_name="MIL_toy")
            
            # logger.info('Test Set, Loss: {:.4f}, Test accuracy: {:.4f}'.format(test_losses, test_accs))
            if test_accs > baseline_acc:
                state = {'net': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, os.path.join(model_save_path, "ckpt_{0:06d}_best.pth".format(epoch)))
                # torch.save(state, '%smodel_%sxfiles_%s_%iclass_%s_head%i.pth' % (chkpt_save,
                #                                                                     datatype, model_name, num_class, target, n_head))
                baseline_acc = test_accs
            
            if epoch % 10 == 0 and epoch > 0:
                state = {'net': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, os.path.join(model_save_path, "ckpt_{0:06d}.pth".format(epoch)))


if __name__ == "__main__":
    # visualize_Dataset()
    train_csv = "./data_csv/Slide_Proposal_train_glioma.csv"
    test_csv = "./data_csv/Slide_Proposal_val_glioma.csv"

    model_save_path = "./Saved_models/cns_glioma/"
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    train(train_csv, test_csv, model_save_path=model_save_path, ft=None, target="glioma", num_of_class=6)
