# COSC 5361

How to install the dependencies.

```bash
pip install -r requirements.txt
```

To train a model with logistic regression.

```bash
python main.py --data $DATA --ml TRUE
```

where `$DATA` is the path to the yaml file inside a YOLO formatted dataset.

To train a model using a CNN.

```bash
python main.py --data $DATA --train --epochs $EPOCHS --optim $OPTIM
```

where `$DATA` is the path to the yaml file inside a YOLO formatted dataset, `$EPOCHS` is the number of epochs and `$OPTIM` is the name of the optimizer algorithm to use, i.e. `[sgd, adamw]`.

To display results of a model.

```bash
python dataframe_plotting.py --data $DATA --plot
```

where `$DATA` is the path to the directory full of `*.csv` files.

## Results

| Model | Epochs | Optimizer | Test Accuracy |
| -- | -- | -- | -- |
| `SimpleCNN` | 50 | `None` | $34.70$% |
| `SimpleCNN` | 100 | `None` | $31.96$% |
| `SimpleCNN` | 50 | `SGD` | $$% |
| `SimpleCNN` | 100 | `SGD` | $$% |
| `SimpleCNN` | 50 | `ADAMW` | $$% |
| `SimpleCNN` | 100 | `ADAMW` | $$% |


## References

* [https://github.com/DrGFreeman/rps-cv](https://github.com/DrGFreeman/rps-cv)
* [https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* []()
