import torch.nn as nn
import torch
import torch.nn.functional as F
class SRL_LSTM(nn.Module):
    def __init__(self,embeddings_dim,hidden_dim,output_dim,num_class,pretrained_embeddings=None):
        """
        Args:
        “pretrained_embeddings (numpy.array): previously trained word embeddings”
        """

        super().__init__()
    
        self.embeddings = pretrained_embeddings

        self.lstm = nn.LSTM(embeddings_dim,hidden_dim,num_layers=1,batch_first=True)

        self.fc = nn.Linear(hidden_dim,num_class)

        #self.dropout = nn.Dropout(dropout)

        #self.act = nn.Sigmoid()


    def forward(self,pretrained_embeddings):
        batch_size = pretrained_embeddings.size(1)
        #self.hidden = self.init_hidden()
        #print(batch_size)


        #pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
        #pretrained_embeddings
        
        outputs, (hidden, cell) = self.lstm(pretrained_embeddings)
        # print(outputs.shape)
        # print(hidden.shape)
        # print(cell.shape)


        #outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #print(outputs.shape)


        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        outputs = self.fc(outputs)
        #outputs=self.act(dense_outputs)
        #print(outputs.shape)
        #outputs= outputs.view(batch_size, -1)

        outputs = outputs.view(batch_size, 23)

        outputs = F.log_softmax(outputs, dim=1)
        #print(outputs.shape)

        return(outputs)
