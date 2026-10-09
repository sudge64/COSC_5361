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
python main.py --data $DATA --train --epochs $EPOCHS
```

where `$DATA` is the path to the yaml file inside a YOLO formatted dataset and `$EPOCHS` is the number of epochs.

To display results of a model.

```bash
python dataframe_plotting.py --data $DATA --plot
```

where `$DATA` is the path to the directory full of `*.csv` files.


## References

* [](https://github.com/DrGFreeman/rps-cv)
* [](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* []()
