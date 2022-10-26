# DL HW1

## [Regression](./regression.ipynb)

### Network Architecture

```python
INPUT_FEATURES = train_feature.shape[1] # 17

Model([
    layer.Linear(INPUT_FEATURES, 16),
    layer.Sigmoid(),
    layer.Linear(16, 4),
    layer.Sigmoid(),
    layer.Linear(4, 1),
])
```

### Learning Curve

![learning curve](./images/regression/loss.png)

### Prediction

- train
![train](./images/regression/train_pred.png)

- test
![test](./images/regression/test_pred.png)

### Feature Importance

## Classification
