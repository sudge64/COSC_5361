from utils.arguments import set_args
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

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

# Main
def main():
    args = set_args()

    device = torch_device()

    # Transforms (Must match training & inference)
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
    test_set = YOLODatasetLoader(test_img,  test_lbl,  transforms=transform)

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
        X_train, y_train = extract_image_data(train_loader)
        X_test, y_test = extract_image_data(test_loader)

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

    # BATCH INFERENCE FROM DIRECTORY
    if getattr(args, 'image', None) is not None:
        print(f'\nRunning batch inference on directory: {args.image}')
        img_dir = Path(args.image)
        
        if not img_dir.is_dir():
            print(f"Error: {args.image} is not a valid directory.")
        else:
            # Supported image extensions
            img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
            image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]
            
            if not image_files:
                print(f"No supported images found in {args.image}")
            else:
                # Sort files for consistent output
                image_files.sort()
                for img_path in image_files:
                    img_pil = Image.open(img_path).convert('RGB')
                    
                    with torch.no_grad():
                        inp = transform(img_pil).unsqueeze(0).to(device)
                        logits = model(inp)
                        probs = torch.softmax(logits, dim=1)
                        pred_id = logits.argmax(dim=1).item()
                        pred_name = cfg['names'][pred_id]
                        confidence = probs[0, pred_id].item()
                        
                    print(f'{img_path.name}: {pred_name} ({confidence:.2%})')

if __name__ == '__main__':
    main()

