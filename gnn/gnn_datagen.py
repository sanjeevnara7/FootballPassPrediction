

import traceback
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder 
path="downloads/GNN_Template.csv"
def df2graph(dfdata,bp_team_value=0.0,case =1,undirected=False,negate_edge=False):
    x_max=105
    y_max=68




    npdata=dfdata.to_numpy()
    nodes= np.arange(len(npdata[:,1]))
    oid=  npdata[:,1].astype(int)
    team=  npdata[:,2].astype(int)
    x=     np.maximum(0,np.minimum(npdata[:,3].astype(float),105))
    y=     np.maximum(0,np.minimum(npdata[:,4].astype(float),68))
    bp=    npdata[:,5].astype(int)
    br=    npdata[:,6].astype(int)

    name = npdata[0,0]
    assert len(np.unique(nodes)) == len(nodes)
    assert np.count_nonzero(bp==1) == 1
    #assert np.count_nonzero(br==1) == 1
    
    assert np.all(x <=x_max)
    assert np.all(y <=y_max)
    assert np.all(x >=0)
    assert np.all(y >=0)

    assert set(team).issubset(set([0, 1]))

    bpID = np.where(bp==1)[0]
    brID = np.where(br==1)[0]
    #assert team[bpID] == team[brID]
    
    offt = team[bpID]

    deft = 0 if offt==1 else 1
    off_nodes=np.where(team==offt)
    def_nodes=np.where(team==deft)
    
    brcop = br.copy()
    br[off_nodes] = bp_team_value
    br[brID] = 1
    br[bpID]=0
    br[def_nodes]=0

    if np.count_nonzero(br==1) != 1:
        aol=7
#         print("multibr",brID,bpID,np.count_nonzero(br==1),name,offt,deft,off_nodes,def_nodes,team)
#         #print(npdata,brcop)
    brcopy = br.copy()

    if bp_team_value == 0:
        
        brcopy[off_nodes] = 1
        brcopy[brID] = 1
        brcopy[bpID]=0
        brcopy[def_nodes]=0

    if case==1:
        src = np.tile(nodes,len(nodes))
        des = np.repeat(nodes,len(nodes))

    elif case==2:
        src_base=np.concatenate([nodes[def_nodes],nodes[bpID]])
        src= np.tile(src_base,len(off_nodes[0]))[None,:]
        des= np.repeat(nodes[off_nodes],len(src_base))[None,:]

    else:
        src = np.tile(nodes,len(nodes))
        des = np.repeat(nodes,len(nodes))

    
    edges= torch.tensor(np.vstack((src,des)))
    
    distance = ((x[np.squeeze(src)]-x[np.squeeze(des)])**2 + (y[np.squeeze(src)]-y[np.squeeze(des)])**2 )**0.5#edge features
    max_distance=(105**2+68**2)**0.5
    #distance[distance==0]=1
    
    di=-(distance/25)**2

    if negate_edge:
        ne= np.where(team[src] != team[des],-1,1)
        distance = (np.exp(di.astype(float)) * np.squeeze(ne))

    else:
        distance = np.exp(di.astype(float))
    

    edge_feats = torch.tensor(distance.astype(float))
    
    
    if undirected:
        edges,edge_feats=to_undirected(edges,edge_feats)
    
    x[:]=x/x_max
    y[:]=y/y_max

    assert np.any(x<=1)
    assert np.any(y<=1)
    ohe=OneHotEncoder(categories=[[0,1]])
    hot_team = torch.tensor(ohe.fit_transform(team[:,None]).toarray().astype(float))
    data=torch.tensor(np.column_stack((x,y,bp)))

    node_features=torch.concat((hot_team,data),dim=1)
    
    # possible_recievers=np.where(brcopy!=0)[0].astype(int)
    Gdata= Data(x=node_features, edge_index=edges,edge_attr = edge_feats, y=torch.tensor(npdata[:,6].astype(float)),oid=oid,name=name) #rcv_idx=possible_recievers,
    return Gdata

def to_undirected(edge_index,edge_attr):
    row1, col1 = edge_index
    row = torch.cat([row1, col1])
    col = torch.cat([col1, row1])

    edge_index = torch.stack([row, col], dim=0)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    
    return edge_index,edge_attr

def csv2data(csv_df="df",path="/content/drive/MyDrive/EECS 6691 Final Project/Data/GNN_Template.csv",graphfunc=2,label_smoothing=0,undirected=False,negEdge=False,inp_df=None):
    """
    Split csv data based on name and convert each group to PyG Data object.
    return list of all such data objects possible from the given csv.

    path: gives the path of the csv file
    graphfunc: controls the graph architecture and links
    
    """
    if csv_df =="csv":
        df = pd.read_csv(path,names=["Gname", "pID", "team", "x", "y", "bp", "br"])
    elif csv_df =="df":
        df = inp_df.loc[:, ['image_name', 'OID', 'pred_team', 'gnd_x', 'gnd_y', 'bp', 'Ball_Reciever']]
        df = df.rename(columns={'image_name': 'Gname', 'OID': 'pID','pred_team':'team', 'gnd_x': 'x', 'gnd_y': 'y', 'bp': 'bp', 'Ball_Reciever': 'br'})
    
    elif csv_df =="df2":
        df = inp_df.loc[:, ['image_name', 'OID', 'pred_team', 'gnd_x', 'gnd_y', 'bp']]
        df = df.rename(columns={'image_name': 'Gname', 'OID': 'pID','pred_team':'team', 'gnd_x': 'x', 'gnd_y': 'y', 'bp': 'bp'})
        df['br']=0

    df=df.dropna()
    mask = df['team'].astype(int) > 1
    rows_to_drop = df[mask].index
    df = df.drop(rows_to_drop)
    
    grouped_df = df.groupby("Gname")

    data_list=[]
    i=0
    for group_name, group_df in grouped_df:
        try:
            data_list.append(df2graph(group_df,bp_team_value=label_smoothing,case=graphfunc, undirected=undirected,negate_edge=negEdge))
        except Exception as e:
#             traceback.print_exc()
#             print(group_name,i)
            i+=1

    
    print("Processed {} objects and {} skipped".format(len(data_list),i))

    return data_list


def generate(path, repeat=5, noise=0.1):
    df = pd.read_csv(path+"gnn_train.csv",names=["Gname", "pID", "team", "x", "y", "bp", "br"])
    
    gauss_dfs = []
    for i in range(repeat):
        #mask=df['Gname'].str.startswith('gaus').astype(bool)
        mask=df['Gname'].astype(bool)
        new_df = df[mask].copy()
        new_df['Gname'] = (new_df['Gname'] + f"_{i}").astype(str)
        gauss_dfs.append(new_df.dropna())

    for df in gauss_dfs:
        df['x'] = df['x'] + np.random.normal(0,noise, size=df.shape[0])
        df['y'] = df['y'] + np.random.normal(0,noise, size=df.shape[0])

    
    n_df = pd.concat(gauss_dfs, ignore_index=True)
    cols_to_convert = ['pID', 'team', 'bp','br']

    for col in cols_to_convert:
        n_df[col] = n_df[col].astype(int)

    n_df.to_csv(path+"gnn_train_generated.csv", index=False,header=False)

