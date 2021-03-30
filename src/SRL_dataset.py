import torch 
import torch.utils.data as data_utils
from torch.utils.data import Dataset
import numpy as np

class SRL_dataset(Dataset):
    "Load SRL dataset"
    def __init__(self,df,transform=None):
        #self.df = pd.read_csv(csv_file)
        cat_col = ['Label', 'chunk', 'postposition', 'head-POS', 'dependency-head', 'dependency', 'srl', 'predicate']
        self.emb = df.drop(cat_col,axis=1)

        #print(self.emb.shape)
        #self.cat = df[cat_col].drop(['srl'],axis=1)
        #self.cat = data_utils.TensorDataset(torch.Tensor(np.array(self.cat)))
        # Error- have to encode categorical data into int first

        #self.emb = data_utils.TensorDataset(torch.Tensor(np.array(self.emb)))

        self.emb = torch.Tensor(np.array(self.emb))
        
        self.y   = torch.tensor(df['srl'].values)

    def __len__(self):
        return(len(self.emb))

    def __getitem__(self,idx):
        #print(idx)
        #return(self.emb.iloc[idx],self.cat.iloc[idx],self.y.iloc[idx])
        return(self.emb[idx],self.y[idx])
        #return(self.df.iloc[idx])

