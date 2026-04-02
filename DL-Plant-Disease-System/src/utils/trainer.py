import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds = []
    targets = []

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(out, dim=1).cpu().numpy())
        targets.extend(y.cpu().numpy())

    perf = {
        'loss': np.mean(losses),
        'acc': accuracy_score(targets, preds)
    }
    return perf


def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    perf = {
        'loss': np.mean(losses),
        'acc': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, average='macro', zero_division=0),
        'recall': recall_score(targets, preds, average='macro', zero_division=0),
        'f1': f1_score(targets, preds, average='macro', zero_division=0),
        'confusion': confusion_matrix(targets, preds)
    }
    return perf


def plot_loss(history, out_path, title='Loss'):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_accuracy(history, out_path, title='Accuracy'):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(cm, classes, out_path):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(out_path)
    plt.close()
