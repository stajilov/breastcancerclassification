# Breast Cancer Classification - Malignant or Benign


Trained on the following data
https://data.world/health/breast-cancer-wisconsin/workspace/file?filename=DatasetDescription.txt

The model uses classic and modern Machine Learning algorithms and the best classifier is AdaBoost (97-99%).

Random Forest trees and Gradient Boosting work fine as well with precition of 94-97%.

To our surprise, classical MLP classifier in keras performed much worse than other non-DL ML algorithms, giving just 60% precition
The only way to boost MLP sequential classifier was to use One Shot Learning add-on.
https://github.com/sorenbouma/keras-oneshot
Although in the example above it was used with LSTM, the usage of the regulizer, resulted in 30% precition increase for our case.


```
1. Run python3 diganosis_classic.py - to train with classical Machine Learning Method
2. Run python3 app.py - to Run as a web service
```

