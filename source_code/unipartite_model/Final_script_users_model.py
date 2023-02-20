#!/usr/bin/env python
# coding: utf-8

# Converted from 2_Unipartite_model.ipynb notebook to py for executing on HPC System

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import matplotlib.pyplot as plt
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear
import torch.nn.functional as F
# PyG
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv


# ##### Processing the merged dataset

#data_df = pd.read_pickle('/N/project/APRS/2_mergedData_final.pkl')

'Removing all records with overall rating less than 3'
# For recommendation we want to keep only books with high ratings

#data_df_3 = data_df[data_df['overall']>=3]

#data_df_3.to_pickle('/N/project/APRS/merged_data_greater3_rating.pkl')
#del data_df # remove to save memory

data_df_3 = pd.read_pickle('/N/project/APRS/merged_data_greater3_rating.pkl')
print(f"length of dataframe : {len(data_df_3)}")


'Sample only those users with minimum 5 transactions so that we have some user history'
sample_users = data_df_3.groupby('reviewerID')['asin'].count().reset_index(drop=False)
sample_users['buy_frequency'] = sample_users['asin'].apply(lambda x: 1 if x >=5 else 0) # minimum 5 transactions


'Filter Dataset'
data_df_3_freq = data_df_3.merge(sample_users, on ='reviewerID', suffixes=('_left', '_right'))
data_df_3_freq =  data_df_3_freq[data_df_3_freq.buy_frequency == 1] # only with 5 or more 
print(len(data_df_3_freq))


'Sampling the dataset as the records are in millions'
data_df_3_freq = data_df_3_freq.rename(columns ={"asin_left": "asin"})
data_df_3_sample = data_df_3_freq.sample(frac=.2)
len(data_df_3_sample)


'Creating label encoders for asin and revieweID as they are alphanumeric'
le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
data_df_3_sample['user_id_idx'] = le_user.fit_transform(data_df_3_sample['reviewerID'].values)
data_df_3_sample['item_id_idx'] = le_item.fit_transform(data_df_3_sample['asin'].values)


'The price column is needed for partioning the users based on max dollar purchases'
data_df_3_sample['price'] = data_df_3_sample['price'].str.replace("$",'')
data_df_3_sample['price'] = pd.to_numeric(data_df_3_sample['price'],errors='coerce')
data_df_3_sample['price'] = data_df_3_sample['price'].fillna(0)

data_df_Price_grouping = data_df_3_sample.groupby('reviewerID')['price'].max()
data_df_Price_grouping = data_df_Price_grouping.reset_index(drop=False)
data_df_Price_grouping["price_category"] = pd.cut(
        x=data_df_Price_grouping["price"],
        bins=[-1, 25, 50, 100, 10000], #Categories
        labels=[0, 1, 2, 3],)
data_df_Price_grouping["price_category"] = data_df_Price_grouping["price_category"].fillna(0)
data_df_3_sample = data_df_3_sample.merge(data_df_Price_grouping, on ='reviewerID', suffixes=('_left', '_right'))
# this will be added as an attribute to the network nodes
nodes_attr = data_df_Price_grouping.set_index('reviewerID').to_dict(orient = 'index')
data_df_Price_grouping.price_category.value_counts()


# ##### Converting the dataset to a network
'Setting up the network'
G = nx.Graph()
G.add_nodes_from(data_df_3_sample['reviewerID'], bipartite='User') 
G.add_nodes_from(data_df_3_sample['item_id_idx'], bipartite='Item') 
G.add_weighted_edges_from(zip(data_df_3_sample['reviewerID'], 
                              data_df_3_sample['item_id_idx'], data_df_3_sample['overall']), weight = 'rating')
print(nx.info(G))

'Conversion to projection network with sum of ratings as weights (edge weight is ignored)'
def my_weight(G, u, v, weight='rating'):
    w = 0
    for nbr in set(G[u]) & set(G[v]):         
         w += G.edges[u,nbr].get(weight, 1) + G.edges[v, nbr].get(weight,1)        
    return w

user_nodes = [n for n in G.nodes() if G.nodes[n]['bipartite'] == 'User'] 
user_graph = bipartite.generic_weighted_projected_graph(G, nodes=user_nodes, weight_function=my_weight)
nx.set_node_attributes(user_graph, nodes_attr) # price attribute added
nodes_list = np.array(list(user_graph.nodes())) # list of nodes

print(nx.info(user_graph))

'Getting the edge index'
for node in user_graph:
    user_graph.nodes[node]['bipartite']=1
user_pyg = from_networkx(user_graph)
edge_index = user_pyg.edge_index


# ##### Conversion to PyG Dataset

'Node Features (degree centrality)'
embeddings = np.array(list(dict(user_graph.degree()).values())) 
scale = StandardScaler()
embeddings = scale.fit_transform(embeddings.reshape(-1,1))

'Classes created in the dataset based on price'
labels = np.asarray([user_graph.nodes[i]['price_category'] for i in user_graph.nodes]).astype(np.int64)

'Custom dataset'
#reference - https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
class AmazonUsers(InMemoryDataset):
    def __init__(self, transform=None):
        super(AmazonUsers, self).__init__('.', transform, None, None)

        data = Data(edge_index=edge_index)
        
        data.num_nodes = user_graph.number_of_nodes()
        
        # embedding 
        data.x = torch.from_numpy(embeddings).type(torch.float32)
        
        # labels
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach()
        
        data.num_classes = 4

        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(user_graph.nodes())), 
                                                            pd.Series(labels),
                                                            test_size=0.30, 
                                                            random_state=42)
        
        n_nodes = user_graph.number_of_nodes()
        
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask

        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
dataset = AmazonUsers()
data = dataset[0]


# ##### Training the Graph Neural Network to get the embeddings of users 

'Graph Convolutional network to obtain user node embeddings'
#reference: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(data.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, data.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = F.log_softmax(self.classifier(h), dim=1)

        return out, h

model = GCN()
print(model)

model = GCN()
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  

def train(data):
    optimizer.zero_grad()  
    out, h = model(data.x, data.edge_index)  
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # only training nodes
    loss.backward()  
    optimizer.step()  
    return loss, h

for epoch in range(401):
    loss, h = train(data) # output loss and embeddings

@torch.no_grad()
def test():
    model.eval()
    logits,_ = model(data.x, data.edge_index)
    mask1 = data['train_mask']
    pred1 = logits[mask1].max(1)[1]
    acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()
    mask = data['test_mask']
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc1,acc

train_acc,test_acc = test()

print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)

# Save model output
torch.save(model.state_dict(), '/N/project/APRS/model_user_results/model_user_final.pkl')
embedding_vector = pd.DataFrame(h.detach().numpy())
embedding_vector.to_csv('/N/project/APRS/model_user_results/embedding_vector_user.csv', index = False)
data_df_3_sample.to_pickle('/N/project/APRS/model_user_results/data_df_3_sample.pkl')
node_list_df = pd.DataFrame(nodes_list).to_pickle('/N/project/APRS/model_user_results/node_list.pkl')


# ##### Finding Top K similar users and the books they bought as suggestions

# Give a reviewer ID
reviwerID_val = data_df_3_sample.reviewerID[10] # Example
index = [i for i, x in enumerate(nodes_list) if reviwerID_val==x]
print(f'reviewerID index in graph {index}')

k = 5 # of similar users
distances = np.linalg.norm(h.detach().numpy() - h[index].detach().numpy(), axis = 1)
# select indices of vectors having the lowest distances from the X
neighbours = np.argpartition(distances, range(0, k))[:k]
node_names = [nodes_list[i] for i in neighbours]
print(f'Top {k} neighbours of {reviwerID_val} :{node_names}')

# Printing the similar user purchases
similar_users_df = data_df_3_sample.loc[data_df_3_sample['reviewerID'].isin(node_names)] # all books are 4 or 5 ratings
print(similar_users_df['asin'])

similar_users_df.sort_values('reviewerID')

####### End ##############################
