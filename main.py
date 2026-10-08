from utils.arguments import set_args
from utils.camera import run_camera
from utils.get_paths import get_paths, load_yaml_config
from utils.image_data import extract_image_data
from utils.simple_cnn import SimpleCNN
from utils.torch_device import torch_device
from utils.training import load_checkpoint
from utils.training import train
from utils.YOLODatasetLoader import YOLODatasetLoader

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Main
def main():
    args = set_args()

    device = torch_device()

    # Transforms (Have to be the same for train & camera)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset paths
    cfg = load_yaml_config(args.data)
    print(cfg['names'])
    train_img, train_lbl = get_paths(cfg, args.data, 'train')
    test_img,  test_lbl  = get_paths(cfg, args.data, 'test')

    # Datasets & loaders
    train_set = YOLODatasetLoader(train_img, train_lbl, transforms=transform)
    test_set  = YOLODatasetLoader(test_img,  test_lbl,  transforms=transform)

    train_loader = DataLoader(train_set,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    test_loader  = DataLoader(test_set, 
                              batch_size=32,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=True)
    if args.ml:
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        X_train, y_train, = extract_image_data(train_loader)
        X_test, y_test = extract_image_data(test_loader)

        # multi_class was removed following scikit v1.7
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean() * 100
        print(f"\nTest Accuracy (Logistic Regression): {acc:.2f}%")
        sys.exit(0)
    else:
        # Model, loss, optimiser, scaler
        model = SimpleCNN(num_classes=len(cfg['names'])).to(device)
        criterion = nn.CrossEntropyLoss()

        scaler = torch.amp.GradScaler(device=device)

        if args.optim:
            optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.8, 0.999), eps=1e-8, weight_decay=0.01)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scaler = torch.amp.GradScaler(device=device)

    ckpt_path = Path(args.ckpt)
    start_epoch = 0

    # If a checkpoint exists, load it
    if ckpt_path.is_file():
        start_epoch, _ = load_checkpoint(ckpt_path, model,
                                         optimizer=optimizer,
                                         scaler=scaler,
                                         map_location=device)

    # Training
    if args.train:
        train(
            start_epoch, 
            args.epochs, 
            model, 
            train_loader, 
            test_loader, 
            criterion, 
            device, 
            optimizer, 
            scaler, 
            ckpt_path
        )

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            _, pred = torch.max(logits, 1)
            total   += targets.size(0)
            correct += (pred == targets).sum().item()
    print(f'\nTest accuracy: {100.0 * correct / total:.2f}%')

    # LIVE CAMERA DEMO
    if args.camera:
        print('\nStarting webcam demo – press "q" to quit')
        run_camera(model, cfg['names'], device, transform, args.camera, args.width, args.height)

if __name__ == '__main__':
    main()
