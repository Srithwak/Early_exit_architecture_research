import torch
import torch.optim as optim
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_classifiers_only(model, dataloader, epochs, optimizer, criterion_fn, device,
                           scheduler=None, max_grad_norm=1.0):
    model.train()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if scheduler is not None:
            scheduler.step(avg_loss)
        print(f"  Warmup {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    for p in model.policies.parameters():
        p.requires_grad = True


def train_joint(model, dataloader, epochs, optimizer, criterion_fn, device,
                energy_lambda=0.05, scheduler=None, max_grad_norm=1.0):
    model.train()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if scheduler is not None:
            scheduler.step(avg_loss)
        print(f"  Joint {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")


def calibrate_thresholds(model, val_loader, device, strategy="confidence",
                         target_acc=0.95, entropy_percentile=80, patience=2):
    model.eval()

    stage_confidences = {i: [] for i in range(model.num_stages)}
    stage_corrects = {i: [] for i in range(model.num_stages)}
    stage_entropies = {i: [] for i in range(model.num_stages)}
    stage_predictions = {i: [] for i in range(model.num_stages)}

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            for i, logits in enumerate(outputs["logits"]):
                probs = torch.softmax(logits, dim=-1)
                conf, preds = torch.max(probs, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                stage_confidences[i].extend(conf.cpu().numpy())
                stage_corrects[i].extend((preds == y).cpu().numpy())
                stage_entropies[i].extend(entropy.cpu().numpy())
                stage_predictions[i].extend(preds.cpu().numpy())

    if strategy == "confidence":
        return _calibrate_confidence(model, stage_confidences, stage_corrects, target_acc)
    elif strategy == "entropy":
        return _calibrate_entropy(model, stage_entropies, stage_corrects, entropy_percentile)
    elif strategy == "patience":
        return _calibrate_patience(model, stage_predictions, stage_corrects, patience)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _calibrate_confidence(model, stage_confidences, stage_corrects, target_acc):
    thresholds = []
    metadata = {"per_stage": []}

    for i in range(model.num_stages - 1):
        confs = np.array(stage_confidences[i])
        corrects = np.array(stage_corrects[i])
        sort_idx = np.argsort(confs)[::-1]
        sorted_confs = confs[sort_idx]
        sorted_corrects = corrects[sort_idx]
        cum_acc = np.cumsum(sorted_corrects) / np.arange(1, len(sorted_corrects) + 1)
        valid_idx = np.where(cum_acc >= target_acc)[0]

        t = sorted_confs[valid_idx[-1]] if len(valid_idx) > 0 else 0.99
        thresholds.append(t)

        exit_rate = np.mean(confs > t)
        metadata["per_stage"].append({
            "stage": i, "threshold": float(t),
            "exit_rate": float(exit_rate), "mean_conf": float(confs.mean()),
        })

    return {"strategy": "confidence", "thresholds": thresholds,
            "target_acc": target_acc, "metadata": metadata}


def _calibrate_entropy(model, stage_entropies, stage_corrects, percentile):
    thresholds = []
    metadata = {"per_stage": []}

    for i in range(model.num_stages - 1):
        entropies = np.array(stage_entropies[i])
        corrects = np.array(stage_corrects[i])
        correct_entropies = entropies[corrects.astype(bool)]

        t = np.percentile(correct_entropies, percentile) if len(correct_entropies) > 0 else 0.01
        thresholds.append(t)

        exit_rate = np.mean(entropies < t)
        metadata["per_stage"].append({
            "stage": i, "threshold": float(t),
            "exit_rate": float(exit_rate), "mean_entropy": float(entropies.mean()),
        })

    return {"strategy": "entropy", "thresholds": thresholds,
            "percentile": percentile, "metadata": metadata}


def _calibrate_patience(model, stage_predictions, stage_corrects, patience_count):
    num_stages = model.num_stages
    num_samples = len(stage_predictions[0])

    total_correct = 0
    total_exits_at = {i: 0 for i in range(num_stages)}

    for s in range(num_samples):
        preds_seq = [stage_predictions[i][s] for i in range(num_stages)]
        corrects_seq = [stage_corrects[i][s] for i in range(num_stages)]

        exited = False
        consecutive = 1
        for i in range(1, num_stages):
            consecutive = consecutive + 1 if preds_seq[i] == preds_seq[i-1] else 1
            if consecutive >= patience_count:
                total_exits_at[i] += 1
                total_correct += int(corrects_seq[i])
                exited = True
                break

        if not exited:
            total_exits_at[num_stages - 1] += 1
            total_correct += int(corrects_seq[-1])

    val_acc = total_correct / num_samples if num_samples > 0 else 0

    return {"strategy": "patience", "thresholds": [],
            "patience": patience_count, "val_acc": val_acc,
            "metadata": {"exit_distribution": dict(total_exits_at)}}
