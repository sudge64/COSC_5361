import torch
from pathlib import Path
import pandas as pd
import time

def saving_helper(optimizer, epoch, folder: Path, name: str | None = None):
    subdir_name = type(optimizer).__name__ if optimizer is not None else "NoOptimizer"
    (folder / subdir_name).mkdir(parents=True, exist_ok=True)

    if name is not None:
        return subdir_name, name

    if optimizer is not None:
        param_group = optimizer.param_groups[0].copy()
        param_group.pop('params', None)
        param_str = "_".join(f"{k}_{v}" for k, v in param_group.items())
    else:
        param_str = "none"

    filename = f"epochs_{epoch}_{param_str}"

    return subdir_name, filename

def save_checkpoint(optimizer, epoch, state, folder: Path, name=None):
    subdir_name, filename = saving_helper(optimizer, epoch, folder, name)
    checkpoint = folder / subdir_name / f"{filename}.pt"
    torch.save(state, checkpoint)
    print(f'Checkpoint saved -> {checkpoint}')

def save_dataframe(df, optimizer, epoch, folder: Path, name=None):
    subdir_name, filename = saving_helper(optimizer, epoch+1, folder, name)
    path = folder / subdir_name / f"{filename}.csv"
    df.to_csv(path, index=False)
    print(f'DataFrame saved -> {folder}')

def load_checkpoint(path: Path, model, optimizer=None, scaler=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    epoch = ckpt.get('epoch', 0)
    loss  = ckpt.get('loss')
    print(f'Loaded checkpoint {path} (epoch {epoch})')
    return epoch, loss

def train(
        start_epoch, 
        epochs, 
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        device, 
        optimizer, 
        scaler, 
        ckpt_path,
        test_img_paths=None,
        class_names=None
):
    if optimizer is not None:
        raise ValueError("Optimizer can not be None when training. Pass a valid optimizer or skip training.")
    data = []
    val_acc_old = 0.0
    for epoch in range(start_epoch, epochs):
        start = time.perf_counter()
        print(f'\n--- Epoch {epoch+1}/{epochs} ---')
        model.train()
        running_loss = 0.0

        for imgs, targets in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, targets)
                scaler.scale(loss).backward()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'  train loss = {epoch_loss:.4f}')

        # Validation after each epoch (optional)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        batch_size = test_loader.batch_size
        
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(test_loader):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(imgs)
                loss   = criterion(logits, targets)
                val_loss += loss.item()
                _, pred = torch.max(logits, 1)
                total   += targets.size(0)
                correct += (pred == targets).sum().item()

                # Print wrongly classified images
                if test_img_paths is not None and class_names is not None:
                    wrong_mask = (pred != targets)
                    if wrong_mask.any():
                        start_idx = i * batch_size
                        end_idx = min(start_idx + batch_size, len(test_img_paths))
                        for local_idx in torch.where(wrong_mask)[0].tolist():
                            orig_idx = start_idx + local_idx
                            if orig_idx < end_idx and orig_idx < len(test_img_paths):
                                fname = Path(test_img_paths[orig_idx]).name
                                true_cls = class_names[targets[local_idx].item()]
                                pred_cls = class_names[pred[local_idx].item()]
                                print(f"x Wrong: {fname}")
                                print(f"True: {true_cls} | Pred: {pred_cls}")

        val_loss /= len(test_loader)
        val_acc = 100.0 * correct / total
        print(f'  val loss = {val_loss:.4f}, acc = {val_acc:.2f}%')

        # Checkpoint
        if optimizer is not None:
            ckpt_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
        else:
            ckpt_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
        if val_acc > val_acc_old:
            val_acc_old = val_acc
            save_checkpoint(optimizer, epoch, ckpt_state, ckpt_path.parent, name='best')
        end = time.perf_counter()
        elapsed = end - start
        print(f'  time: {elapsed:.4f} s')
        data.append({"Epoch": epoch, "Train Loss": epoch_loss, "Val Loss": val_loss, "Val Acc": val_acc, "Time": elapsed})

    df = pd.DataFrame(data)
    save_dataframe(df, optimizer, epochs, ckpt_path.parent)
