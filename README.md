# Facebook friend link prediction based on GCN network
基于GCN网络的Facebook好友关系链接预测
## Introduction
Link prediction is an important topic in social network analysis. The goal of link prediction is to predict the link relationships that will appear in the network in the future, using node attributes and observed links to predict whether a link exists. Link prediction is widely used, such as predicting the friendships and partnerships between people in social networks. This project is based on the analysis of social networks composed of Facebook web pages, and the model uses a two-layer graph neural network (GCN) to predict the link between facebook users' friend relationships.
## Related Work
### Knowledge preparation
#### Learning about link prediction:
for the method of link prediction, all the method assign a connection weight score(x,y) to pairs of nodes x,y, based on the input graph, and the produce a ranked list in decreasing order of score(x,y), which can be viewed as comuting a measure of proximity or "similarity" between nodes x and y.
#### Learning about GNN:
Graph convolutional neural networks are actually the same as convolutional neural networks (CNNs) as feature extractors, but the object of the GCN is graph data. GCN is very versatile, it designed a method to extract features from graph data, so that we can use these features to node classification of graph data (node classification), graph classification (link prediction), but also incidentally get the embedded representation of the graph (graph embedding).
Suppose we now have a dataset with N nodes in the data, each node has its own characteristics, we set the characteristics of these nodes to form an N×D dimensional matrix X, and then the relationship between the nodes will also form an N× N-dimensional matrix A, also known as the adjacency matrix. X and A are the inputs to our model. Then the core formula of GCN is: 

A wave = A + I, I is the identity matrix, D wave is the degree matrix of A wave, H is the feature of each layer, and for the input layer, H is Xσ is a nonlinear activation function.
