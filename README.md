# Crop recommendation

Recommends a crop from soil and weather measurements. Built with James Sanford-Luevano.

## Data

`datasets/crop.csv` is the [Kaggle crop recommendation set](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset): 2,200 rows of nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall, and the crop label.

The notebook drops missing rows and standardizes the measurements. Potassium is dropped because it is highly correlated with phosphorus. Crop names are grouped into cereals, pulses, fruits, and fiber or cash crops.

## Models

`crop-recommender.ipynb` fits a decision tree and a scikit-learn neural net (`MLPClassifier`).

The tree is checked across depths with cross-validation, a classification report, and a confusion matrix. The split criterion is Gini impurity. Humidity is the most important split. Deeper trees lower bias until they overfit.

The network is tried as three architectures: one hidden layer of 3 nodes, one hidden layer of 5 nodes, and two hidden layers of 10 and 5 nodes.

## Results

Cross-validated tree accuracy peaks at 69.09% at depth 10, then flattens. A depth-1 tree scores 69% on the held-out set. A depth-9 tree scores 100% there, which is the overfit side of that curve.

| Network | Hidden layers | Accuracy |
| --- | --- | ---: |
| 1 | 3 nodes | 0.85 |
| 2 | 5 nodes | 0.93 |
| 3 | 10, then 5 | 0.97 |

The depth-9 tree and the two-layer network are the strongest of the models that were trained. Edge weights in the network did not line up with the tree splits.

## Run

```bash
pip install -r requirements.txt
```

Graphviz is needed only if you export a tree diagram. Open `crop-recommender.ipynb`.
