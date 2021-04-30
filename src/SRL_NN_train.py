import torch 
import pandas as pd
from SRL_NN_Classifier import SRL_LSTM
from SRL_dataset import SRL_dataset
import torch.optim as optim 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import matplotlib.pyplot as plt

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
        for emb,y in testloader:
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

            test_loss+=loss.item()
            test_acc+=acc.item() 

        print("test loss on epoch {epoch}  is {loss} and test accuracy {accuracy}".format(epoch=epoch,loss=(test_loss/len(testloader)),accuracy=(test_acc/len(testloader))))
        #print(f"accuracy on epoch {epoch} = {total_acc/len(dataloader)}")
        test_acc_list.append((test_acc/len(testloader)))
        test_loss_list.append((test_loss/len(testloader)))
        epoch_list.append(epoch)

    return(train_loss_list,test_loss_list,train_acc_list,test_acc_list,epoch_list)

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
    
    # Hyperparameters
    EMBEDDING_DIM = 300
    NUM_HIDDEN_NODES = 100
    NUM_OUTPUT_NODES = 1
    NUM_CLASSES = 23

    model = SRL_LSTM(embeddings_dim=EMBEDDING_DIM,hidden_dim=NUM_HIDDEN_NODES,output_dim =NUM_OUTPUT_NODES,num_class=NUM_CLASSES,pretrained_embeddings=None)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()


    # Dataset
    batchsize=64
    dataloader=DataLoader(dataset=train_dataset,batch_size=batchsize,shuffle=False,num_workers=0)
    testloader=DataLoader(dataset=test_dataset,batch_size=batchsize,shuffle=False,num_workers=0)


    epochs = 50
    train_loss, test_loss, train_acc,test_acc,epoch_list = train_nn(model,dataloader,testloader,epochs,optimizer,criterion)

    df_results = pd.DataFrame(list(zip(train_loss,test_loss,train_acc,test_acc)),columns=['Train Loss','Test Loss','Train Accuracy','Test Accuracy'])
    df_results.to_csv("../data/results/classifier_report_lstm.csv",index=None,sep=',')

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    plt.plot(epoch_list, train_acc)
    plt.plot(epoch_list,test_acc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['Train Accuracy','Test Accuracy'])

    plt.savefig('../data/results/NN_train_test_accuracy.png')


    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(epoch_list, train_loss)
    plt.plot(epoch_list,test_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['Train Loss','Test Loss'])

    plt.savefig('../data/results/NN_train_test_loss.png')