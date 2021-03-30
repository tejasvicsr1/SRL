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


    def forward(self,pretrained_embeddings):
        batch_size = pretrained_embeddings.size(1)
       
        outputs, (hidden, cell) = self.lstm(pretrained_embeddings)
       
        outputs = self.fc(outputs)
     

        outputs = outputs.view(batch_size, 23)

        outputs = F.log_softmax(outputs, dim=1)
        #print(outputs.shape)

        return(outputs)
