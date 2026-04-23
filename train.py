import torch
import torch.optim as optim
import numpy as np

def train_classifiers_only(model, dataloader, epochs, optimizer, criterion_fn, device):
    model.train()
    print("--- Phase 1: Warmup (Classifiers Only) ---")
    for p in model.policies.parameters():
        p.requires_grad = False

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            criterion = criterion_fn(energy_lambda=0.0)
            if "gates" in outputs:
                loss = criterion(outputs["logits"], outputs["p_exits"], y, outputs["gates"])
            else:
                loss = criterion(outputs["logits"], outputs["p_exits"], y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

def train_joint(model, dataloader, epochs, optimizer, criterion_fn, device, energy_lambda=0.05):
    model.train()
    print(f"\n--- Phase 2: Joint Training (Lambda={energy_lambda}) ---")
    for p in model.policies.parameters():
        p.requires_grad = True

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            criterion = criterion_fn(energy_lambda=energy_lambda)
            if "gates" in outputs:
                loss = criterion(outputs["logits"], outputs["p_exits"], y, outputs["gates"])
            else:
                loss = criterion(outputs["logits"], outputs["p_exits"], y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

def calibrate_thresholds(model, val_loader, device, target_acc=0.95):
    model.eval()
    print(f"\n--- Phase 3: Calibrating Exits (Target Acc: {target_acc*100:.1f}%) ---")
    stage_confidences = {i: [] for i in range(model.num_stages)}
    stage_corrects = {i: [] for i in range(model.num_stages)}

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            logits_list = outputs["logits"]
            for i, logits in enumerate(logits_list):
                probs = torch.softmax(logits, dim=-1)
                conf, preds = torch.max(probs, dim=-1)
                stage_confidences[i].extend(conf.cpu().numpy())
                stage_corrects[i].extend((preds == y).cpu().numpy())

    thresholds = []
    for i in range(model.num_stages - 1):
        confs = np.array(stage_confidences[i])
        corrects = np.array(stage_corrects[i])
        sort_idx = np.argsort(confs)[::-1]
        sorted_confs = confs[sort_idx]
        sorted_corrects = corrects[sort_idx]
        cum_acc = np.cumsum(sorted_corrects) / np.arange(1, len(sorted_corrects) + 1)
        valid_idx = np.where(cum_acc >= target_acc)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[-1]
            t = sorted_confs[best_idx]
        else:
            t = 0.99
        thresholds.append(t)
        print(f"Stage {i} Threshold T_{i}: {t:.4f}")
    return thresholds
