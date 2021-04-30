"""Author - Rahul Mehta """

import torch 
import pandas as pd
import numpy as np
import seaborn as sns
from SRL_NN_Classifier import SRL_LSTM
from sklearn.metrics import classification_report
from SRL_dataset import SRL_dataset
import torch.optim as optim 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse



def label_encode(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return(y,list(le.classes_),le)

def model_accuracy(predict,y):
  true_predict=(predict==y).float()
  acc=true_predict.sum()/len(true_predict)
  return(acc)

def train_nn(model,dataloader,testloader,epochs,optimizer,criterion):
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc=0.0

        for emb,y in dataloader:
            batch_size = emb.shape[0]
            #print(y.shape)
            #print(emb.view([1, 64,300]).shape)
            preds = model(emb.view([1,batch_size,300]))
            #print(preds[0])
            #print(preds.shape)
            loss = criterion(preds, y)
            #print("Loss {}".format(loss))

            preds = torch.argmax(preds,dim=1)
            #print(preds)
            acc = sum(preds == y) / float(batch_size)
            #acc=model_accuracy(preds, y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc+=acc.item() 

        print("train loss on epoch {epoch}  is {loss} and training accuracy {accuracy}".format(epoch=epoch,loss=(total_loss/len(dataloader)),accuracy=(total_acc/len(dataloader))))
        #print(f"accuracy on epoch {epoch} = {total_acc/len(dataloader)}")

        train_acc_list.append((total_acc/len(dataloader)))
        train_loss_list.append((total_loss/len(dataloader)))

        test_loss = 0.0
        test_acc=0.0
        all_preds =np.zeros(0)
        all_y =  np.zeros(0)
        for emb,y in testloader:
            batch_size = emb.shape[0]
            #print(y.shape)
            #print(emb.view([1, 64,300]).shape)
            preds = model(emb.view([1,batch_size,300]))
            #print(preds[0])
            #print(preds.shape)
            loss = criterion(preds, y)
            #print("Loss {}".format(loss))
            #print(y)
            preds = torch.argmax(preds,dim=1)
            #print(preds)
            acc = sum(preds == y) / float(batch_size)
            #acc=model_accuracy(preds, y)


            all_preds = np.append(all_preds,np.array(preds))
            all_y = np.append(all_y,np.array(y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_loss+=loss.item()
            test_acc+=acc.item() 

        print("test loss on epoch {epoch}  is {loss} and test accuracy {accuracy}".format(epoch=epoch,loss=(test_loss/len(testloader)),accuracy=(test_acc/len(testloader))))
        #print(f"accuracy on epoch {epoch} = {total_acc/len(dataloader)}")
        test_acc_list.append((test_acc/len(testloader)))
        test_loss_list.append((test_loss/len(testloader)))
        epoch_list.append(epoch)

    return(train_loss_list,test_loss_list,train_acc_list,test_acc_list,all_preds,all_y,epoch_list)

if __name__ == "__main__":



    # Read dataset
    df = pd.read_csv("../data/processed/interim.txt")
    df['srl'],classes,le = label_encode(df['srl'])
    #df = df.head(1000)
  
    
    #Train and Test
    X_train, X_test, y_train, y_test  = train_test_split(df.drop(['srl'],axis=1), df['srl'],test_size=0.33,random_state=123)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


    df_train = pd.concat([X_train,y_train],axis=1)
    df_test = pd.concat([X_test,y_test],axis=1)
    
    train_dataset = SRL_dataset(df_train)
    test_dataset = SRL_dataset(df_test)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--EMBEDDING_DIM', type=int)
    parser.add_argument('--NUM_HIDDEN_NODES', type=int)
    #parser.add_argument('--NUM_CLASSES', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--learning_rate', type=float)

    args = parser.parse_args()
    # Hyperparameters
    # EMBEDDING_DIM = 300
    # NUM_HIDDEN_NODES =100
    # NUM_OUTPUT_NODES = 1
    # NUM_CLASSES = 23
    # epochs = 50
    # batchsize=64
    # learning_rate =0.0001

    EMBEDDING_DIM = args.EMBEDDING_DIM
    NUM_HIDDEN_NODES = args.NUM_HIDDEN_NODES
    NUM_OUTPUT_NODES = 1
    NUM_CLASSES = 23
    epochs = args.epochs
    batchsize= args.batchsize
    learning_rate = args.learning_rate

    print(EMBEDDING_DIM,NUM_HIDDEN_NODES,epochs,batchsize,learning_rate)

    model = SRL_LSTM(embeddings_dim=EMBEDDING_DIM,hidden_dim=NUM_HIDDEN_NODES,output_dim =NUM_OUTPUT_NODES,num_class=NUM_CLASSES,pretrained_embeddings=None)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()



    # Dataset
 
    dataloader=DataLoader(dataset=train_dataset,batch_size=batchsize,shuffle=False,num_workers=0)
    testloader=DataLoader(dataset=test_dataset,batch_size=batchsize,shuffle=False,num_workers=0)


    # Conf matrix
    train_loss, test_loss, train_acc,test_acc,preds,Y,epoch_list = train_nn(model,dataloader,testloader,epochs,optimizer,criterion)
    conf_matrix = pd.DataFrame(confusion_matrix(Y, preds))
    print(conf_matrix)
    sns.set(font_scale=1.4) # for label size
    sns_plot = sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 8}).get_figure() # font size
    
    class_report = pd.DataFrame(classification_report(Y, preds,output_dict=True)).transpose()
    
    df_results = pd.DataFrame(list(zip(train_loss,test_loss,train_acc,test_acc)),columns=['Train Loss','Test Loss','Train Accuracy','Test Accuracy'])
    

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    plt.plot(epoch_list, train_acc)
    plt.plot(epoch_list,test_acc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['Train Accuracy','Test Accuracy'])
    plt.savefig('../data/results/NN_train_test_accuracy_bilstm_50e.png')


    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(epoch_list, train_loss)
    plt.plot(epoch_list,test_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['Train Loss','Test Loss'])
    plt.savefig('../data/results/NN_train_test_loss_bilstm_50e.png')

    class_report.to_csv("../data/results/classification_report_bilstm_50e.csv")
    sns_plot.savefig("../data/results/confusion_matrix_bilstm_50e.png")
    df_results.to_csv("../data/results/classifier_report_bilstm_50e.csv",index=None,sep=',')
    PATH = '../models/srl_hindi_bilstm_50e.pth'
    torch.save(model.state_dict(), PATH)
    
    