import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Run training')
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs for training (Recommended: 256).",
    )
    parser.add_argument('--ckpt', type=str, default='checkpoints/last.pt')
    parser.add_argument('--data', type=str,
                        default='./rock-paper-scissors.v14i.yolov11/data.yaml')
    parser.add_argument(
        "--camera",
        type=int,
        help="Video device to chose (Recommended: 3 -> indices `ls /dev/video*`).",
    )
    parser.add_argument(
        "--ml",
        type=bool,
        help="Choose whether to use machine learning or deep learning.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Width of video frame (Recommended: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Height of video frame (Recommended: 480).",
    )
    parser.add_argument(
        "--optim",
        type=str,
        help="Optimizer algorithm to use (Recommended: sgd).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate to use (Recommended: 0.001).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Alpha to use (Recommended: 0.99).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Beta to use (Recommended: -0.8).",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        help="Beta for the Adams to use (Recommended: -0.8).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Epsilon to use (Recommended: 1e-6).",
    )
    parser.add_argument(
        "--decay",
        type=float,
        help="Beta 2 Decay for Adafactor to use (Recommended: -0.8).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Weight Decay for AdamW to use (Recommended: 0.01).",
    )
    parser.add_argument(
        "--epsilon2",
        type=float,
        help="Epsilon2 for Adafactor to use (Recommended: 1e-3).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help="Momentum to use (Recommended: 0).",
    )
    return parser.parse_args()
