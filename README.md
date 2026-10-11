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

Mis-classifications:

```
paper_back_2_wade_jpeg.rf.c9f4f1beefdc93b57636f0fde6449160.jpg: Rock (96.05%)
paper_back_4_huggins_jpeg.rf.f7c60b3b80d8389b99ccca8e3694dbf2.jpg: Rock (52.86%)
paper_back_5_hopkins_jpeg.rf.e3897f4e1e702a2aed173df6d174d53a.jpg: Paper (99.95%)
paper_front_2_wade_jpeg.rf.1c1aab77038d910031ddd9f6e44c93e7.jpg: Rock (85.64%)
paper_front_4_huggins_jpeg.rf.3537ff5ee92502051036fb8a876f8821.jpg: Rock (95.55%)
paper_front_5_hopkins_jpeg.rf.a4a1a6da9cca9d821240ab841221cea4.jpg: Rock (96.34%)
rock_back_3_huggins_jpeg.rf.914410b1b3093681b0cc15d38d6141b0.jpg: Rock (79.69%)
rock_back_3_wirsdorfer_jpeg.rf.8a889919821139717f22da4c34e2be5b.jpg: Rock (90.94%)
rock_back_4_hopkins_jpeg.rf.5b82dbfe521e059f012031620ffe1734.jpg: Rock (96.16%)
rock_front_3_huggins_jpeg.rf.f338e9feeb4788ef37591ae38b793d79.jpg: Paper (88.99%)
rock_front_3_wirsdorfer_jpeg.rf.f35675f0fc3cf6f5467f37aae41b85f4.jpg: Rock (75.38%)
rock_front_4_hopkins_jpeg.rf.a5939ec7ac6e2c484462c0ee07191d00.jpg: Rock (95.84%)
scissors_back_2_wirsdorfer_jpeg.rf.df5257bd6b0463514d404d78b118464a.jpg: Rock (95.97%)
scissors_back_3_wade_jpeg.rf.be5d9e6434d67e1d3bb2fc887b30d951.jpg: Paper (76.16%)
scissors_back_4_huggins_jpeg.rf.09a615d96663414bb56c2b2f2fbfd139.jpg: Paper (68.46%)
scissors_front_2_wirsdorfer_jpeg.rf.dee9cce09d86fa8eb06ad01b51dad11c.jpg: Rock (51.42%)
scissors_front_3_wade_jpeg.rf.fb2837d8e57c70ee4840f8eb2c68ebe5.jpg: Paper (90.27%)
scissors_front_4_huggins_jpeg.rf.a950219a42a65899b045106d42143332.jpg: Paper (79.73%)
```


## References

* [https://github.com/DrGFreeman/rps-cv](https://github.com/DrGFreeman/rps-cv)
* [https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* []()
