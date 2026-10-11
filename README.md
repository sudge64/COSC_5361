# COSC 5361

How to install the dependencies.

```bash
pip install -r requirements.txt
```

To train a model with logistic regression.

```bash
python main.py --data $DATA --ml
```

where `$DATA` is the path to the yaml file inside a YOLO formatted dataset.

To train a model using a CNN.

```bash
python main.py --data $DATA --train --epochs $EPOCHS --optim $OPTIM
```

where `$DATA` is the path to the yaml file inside a YOLO formatted dataset, `$EPOCHS` is the number of epochs and `$OPTIM` is the name of the optimizer algorithm to use, i.e. `[None, sgd, adamw]`.

To display results of a model.

```bash
python dataframe_plotting.py --data $DATA --plot
```

where `$DATA` is the path to the directory full of `*.csv` files.

## Results

### Non-Augmented

| Model | Epochs | Optimizer | Test Accuracy |
| -- | -- | -- | -- |
| `LogisticRegression` | 0 | `None` | $89.95$% |
| `SimpleCNN` | 50 | `None` | $31.96$% |
| `SimpleCNN` | 100 | `None` | $$% |
| `SimpleCNN` | 50 | `SGD` | $$% |
| `SimpleCNN` | 100 | `SGD` | $$% |
| `SimpleCNN` | 50 | `ADAMW` | $$% |
| `SimpleCNN` | 100 | `ADAMW` | $$% |

SimpleCNN | None | 50:
Weighted Precision: 0.1022
Weighted Recall: 0.3196
Weighted F1-Score: 0.1548

SimpleCNN | None | 100:
Weighted Precision: 0.1111
Weighted Recall: 0.3333
Weighted F1-Score: 0.1667


SimpleCNN | SGD | 50:
SimpleCNN | SGD | 100:
SimpleCNN | ADAMW | 50:
SimpleCNN | ADAMW | 100:

### Augmented

| Model | Epochs | Optimizer | Test Accuracy |
| -- | -- | -- | -- |
| `LogisticRegression` | 0 | `None` | $89.95$% |
| `SimpleCNN` | 50 | `None` | $34.70$% |
| `SimpleCNN` | 100 | `None` | $34.70$% |
| `SimpleCNN` | 50 | `SGD` | $97.26$% |
| `SimpleCNN` | 100 | `SGD` | $99.09$% |
| `SimpleCNN` | 50 | `ADAMW` | $99.54$% |
| `SimpleCNN` | 100 | `ADAMW` | $99.54$% |

SimpleCNN | None | 50:
Weighted Precision: 0.1204
Weighted Recall: 0.3470
Weighted F1-Score: 0.1788

SimpleCNN | None | 100:
Weighted Precision: 0.1022
Weighted Recall: 0.3196
Weighted F1-Score: 0.1548

SimpleCNN | SGD | 50:
Weighted Precision: 0.9819
Weighted Recall: 0.9817
Weighted F1-Score: 0.9818

SimpleCNN | SGD | 100:
Weighted Precision: 0.9779
Weighted Recall: 0.9772
Weighted F1-Score: 0.9773

SimpleCNN | ADAMW | 50:
Weighted Precision: 0.9863
Weighted Recall: 0.9863
Weighted F1-Score: 0.9863

SimpleCNN | ADAMW | 100:
Weighted Precision: 0.9774
Weighted Recall: 0.9772
Weighted F1-Score: 0.9772


## References

* [https://github.com/DrGFreeman/rps-cv](https://github.com/DrGFreeman/rps-cv)
* [https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* []()
