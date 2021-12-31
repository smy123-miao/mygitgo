# Facebook friend link prediction based on GCN network
基于GCN网络的Facebook好友关系链接预测
## Introduction
Link prediction is an important topic in social network analysis. The goal of link prediction is to predict the link relationships that will appear in the network in the future, using node attributes and observed links to predict whether a link exists. Link prediction is widely used, such as predicting the friendships and partnerships between people in social networks. This project is based on the analysis of social networks composed of Facebook web pages, and the model uses a two-layer graph neural network (GCN) to predict the link between facebook users' friend relationships.
## Related Work
### Knowledge preparation
#### Learning about link prediction:
for the method of link prediction, all the method assign a connection weight score(x,y) to pairs of nodes x,y, based on the input graph, and the produce a ranked list in decreasing order of score(x,y), which can be viewed as comuting a measure of proximity or "similarity" between nodes x and y.
#### Learning about GCN:
Graph convolutional neural networks are actually the same as convolutional neural networks (CNNs) as feature extractors, but the object of the GCN is graph data. GCN is very versatile, it designed a method to extract features from graph data, so that we can use these features to node classification of graph data (node classification), graph classification (link prediction), but also incidentally get the embedded representation of the graph (graph embedding).

![Image text](https://github.com/smy123-miao/mygitgo/blob/master/img/GCNmoxing.JPG)

### Using Pytorch_Geometric(PyG)
PyG (PyTorch Geometric) is a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.
It consists of various methods for deep learning on graphs and other irregular structures, also known as geometric deep learning, from a variety of published papers. In addition, it consists of easy-to-use mini-batch loaders for operating on many small and single giant graphs, multi GPU-support, distributed graph learning via Quiver, a large number of common benchmark datasets. After downloading the package, you can use the dataset in it, which contains the Facebook dataset.The network model is encapsulated and can be used arbitrarily.Websites for PyG:https://pytorch-geometric.readthedocs.io/en/latest.
## Methodology
### Problem(Definition)
Social interaction between people constitutes a social network, through which we can discover the relationship between people. Facebook is a social platform. Taking people as a node, the characteristics that this person shows in social life, such as his hobbies, his school, his beliefs, etc., can be used as node features, and through features extraction we can find the connection between people, and this link is reflected in whether two people are friends or not. So link prediction on social platforms is to determine whether there may be a friendship relationship between two people.
### Dataset
This project uses the Facebook-page data included in PyG, and the reference data through TORCH_GEOMETRIC.DATASETS. It can be downloaded from the cloud to the local area, and the specific data content is as follows:

![Image text](https://github.com/smy123-miao/mygitgo/blob/master/img/datashow.png)

### Algorithm
#### Using GCVconv

Suppose we now have a dataset with N nodes in the data, each node has its own characteristics, we set the characteristics of these nodes to form an N×D dimensional matrix X, and then the relationship between the nodes will also form an N× N-dimensional matrix A, also known as the adjacency matrix. X and A are the inputs to our model.The propagation between GCN layers is as follows:

<img src="https://github.com/smy123-miao/mygitgo/blob/master/img/formula-GCN.png" width="400" height="200" alt="hh"/>

#### model details

<img src="https://github.com/smy123-miao/mygitgo/blob/master/img/model_pro.png" width="400" height="200" alt="hh"/>

