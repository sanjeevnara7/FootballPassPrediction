<!-- #region -->
# Deep Learning for Soccer Pass Receiver Prediction in Broadcast Images
<img src="https://img.shields.io/badge/python-3.9-green" /> <img src="https://img.shields.io/badge/torch-1.13.0-orange" /> <img src="https://img.shields.io/badge/PyG-2.3-blue" /><br>

<p align="center">
    <img src="https://github.com/sanjeevnara7/FootballPassPrediction/blob/main/figures/pred_readme_gif_alt" width="60%">
</p>
We propose a multi-stage system for Football/Soccer Pass Receiver Prediction. The system consists of 4 major stages: <br>
<ol>
    <li>Player and Ball Detection using YOLO</li>
    <li>Team Identification using Clustering on Images</li>
    <li>Perspective Transformation to aerial view</li>
    <li>Construction of Player Graph and Pass Receiver Prediction using a Graph Attention Network (GATv2)</li>
</ol>

<p align="center">
    <img src="https://github.com/sanjeevnara7/FootballPassPrediction/blob/main/figures/e2e_figure.png"  width="60%" height="30%">
</p>

## Dataset:
We develop a new dataset nicknamed **'SoccerPass'** to train/evaluate our system. The dataset is loosely based on data collected from the [SoccerNetv2 database](https://www.soccer-net.org/data). The SoccerPass dataset is constructed by hand-picking passing frames from over 30 top European broadcast matches. The matches cover a wide range of teams and competitions such as English Premier League, Bundesliga, French Ligue 1 and UEFA Champions League. We selected $\sim1.2$k frames where a pass was about to be performed, and annotate each image with the desired attributes. <br>
#### Our dataset can be found at [this link](https://drive.google.com/drive/folders/19VYIeUQN9b_BN69X3NcEvwYPj1Ba0wIl?usp=share_link).
<br>

## Organization:
<br>


`Detection.ipynb` - Contains results of object detection models for player/ball localization. <br>
`Clustering.ipynb` - Contains results of clustering for Team Identification. <br>
`PerspectiveTransform.ipynb` - Contains results of Perspective Transformation to obtain ground coordinates. <br>
`Receiver Prediction.ipynb` - Contains results of Pass Receiver Prediction using GNNs. <br>
<br>

`E2E.ipynb` - Contains results of end-to-end system evaluated on our dataset. <br>
`E2E_Video.ipynb` - Contains results of end-to-end system evaluate on a video clip. Note that we only use short clips as this is not a real-time solution.<br>

<br>

`/configs` - Contains config files for models <br>
`/gnn` - Contains build + training code for GNN models <br>
`/PerspectiveTransform` - Contains code for perspective transform based on [this original implementation](https://github.com/FootballAnalysis/footballanalysis/tree/main/Perspective%20Transformation) <br>
`/yolomodels` - Contains build + training code for YOLOv7,v8 models based on original implementations <br>

## Collaborators:

#### Sanjeev Narasimhan | sn3007@columbia.edu
#### Pranav Deevi | pid2104@columbia.edu
#### Vishal Bhardwaj | vb2573@columbia.edu
<!-- #endregion -->

```python

```
