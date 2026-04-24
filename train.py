import torch
import torch.optim as optim
import numpy as np
import random

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Reproducibility] Seed set to {seed}")

# ──────────────────────────────────────────────
# Phase 1: Warmup (Classifiers Only)
# ──────────────────────────────────────────────

def train_classifiers_only(model, dataloader, epochs, optimizer, criterion_fn, device,
                           scheduler=None, max_grad_norm=1.0):
    """
    Train only the classifier heads (exit policies frozen).
    This ensures all classifier branches learn good representations
    before the exit policy is optimized.
    """
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if scheduler is not None:
            scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # Re-enable policy gradients for subsequent phases
    for p in model.policies.parameters():
        p.requires_grad = True

# ──────────────────────────────────────────────
# Phase 2: Joint Training
# ──────────────────────────────────────────────

def train_joint(model, dataloader, epochs, optimizer, criterion_fn, device,
                energy_lambda=0.05, scheduler=None, max_grad_norm=1.0):
    """
    Jointly train all parameters (classifiers + exit policies + gates).
    The energy_lambda controls the trade-off between classification
    accuracy and computational cost.
    """
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if scheduler is not None:
            scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# ──────────────────────────────────────────────
# Phase 3: Threshold Calibration
# ──────────────────────────────────────────────

def calibrate_thresholds(model, val_loader, device, strategy="confidence",
                         target_acc=0.95, entropy_percentile=80, patience=2):
    """
    Calibrate exit thresholds using one of three strategies:
    
    - "confidence": Exit when max softmax probability > threshold.
      Threshold is chosen so that accuracy of exited samples >= target_acc.
    - "entropy": Exit when prediction entropy < threshold.
      Threshold is chosen at a percentile of the validation entropy distribution.
    - "patience": Exit when the same class is predicted for `patience`
      consecutive stages. Returns the patience count (not per-stage thresholds).
    
    Returns:
        dict with keys:
            "strategy": str
            "thresholds": list of per-stage thresholds (for confidence/entropy)
            "patience": int (for patience strategy)
            "metadata": dict of calibration metadata
    """
    model.eval()
    print(f"\n--- Phase 3: Calibrating Exits (Strategy: {strategy}) ---")

    # Collect per-stage statistics
    stage_confidences = {i: [] for i in range(model.num_stages)}
    stage_corrects = {i: [] for i in range(model.num_stages)}
    stage_entropies = {i: [] for i in range(model.num_stages)}
    stage_predictions = {i: [] for i in range(model.num_stages)}

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            logits_list = outputs["logits"]
            for i, logits in enumerate(logits_list):
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
        raise ValueError(f"Unknown strategy: {strategy}. Use 'confidence', 'entropy', or 'patience'.")


def _calibrate_confidence(model, stage_confidences, stage_corrects, target_acc):
    """Confidence-based threshold calibration (original approach)."""
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
        if len(valid_idx) > 0:
            best_idx = valid_idx[-1]
            t = sorted_confs[best_idx]
        else:
            t = 0.99
        thresholds.append(t)
        exit_rate = np.mean(confs > t)
        metadata["per_stage"].append({
            "stage": i, "threshold": float(t),
            "exit_rate": float(exit_rate),
            "mean_conf": float(confs.mean()),
        })
        print(f"  Stage {i} Threshold: {t:.4f} (exit rate: {exit_rate:.2%})")

    return {"strategy": "confidence", "thresholds": thresholds,
            "target_acc": target_acc, "metadata": metadata}


def _calibrate_entropy(model, stage_entropies, stage_corrects, percentile):
    """Entropy-based threshold calibration.
    
    Lower entropy = more confident. We set the threshold at a percentile
    of the entropy distribution of correctly-classified samples.
    """
    thresholds = []
    metadata = {"per_stage": []}

    for i in range(model.num_stages - 1):
        entropies = np.array(stage_entropies[i])
        corrects = np.array(stage_corrects[i])

        # Use entropy distribution of correct predictions to set threshold
        correct_entropies = entropies[corrects.astype(bool)]
        if len(correct_entropies) > 0:
            t = np.percentile(correct_entropies, percentile)
        else:
            t = 0.01  # Very strict if no correct predictions

        thresholds.append(t)
        exit_rate = np.mean(entropies < t)
        metadata["per_stage"].append({
            "stage": i, "threshold": float(t),
            "exit_rate": float(exit_rate),
            "mean_entropy": float(entropies.mean()),
        })
        print(f"  Stage {i} Entropy Threshold: {t:.4f} (exit rate: {exit_rate:.2%})")

    return {"strategy": "entropy", "thresholds": thresholds,
            "percentile": percentile, "metadata": metadata}


def _calibrate_patience(model, stage_predictions, stage_corrects, patience_count):
    """Patience-based calibration.
    
    The model exits when `patience_count` consecutive stages predict
    the same class. This doesn't produce per-stage thresholds but
    returns the patience parameter for use during inference.
    """
    # Validate the patience parameter on the validation set
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
            if preds_seq[i] == preds_seq[i - 1]:
                consecutive += 1
            else:
                consecutive = 1

            if consecutive >= patience_count:
                total_exits_at[i] += 1
                total_correct += int(corrects_seq[i])
                exited = True
                break

        if not exited:
            total_exits_at[num_stages - 1] += 1
            total_correct += int(corrects_seq[-1])

    val_acc = total_correct / num_samples if num_samples > 0 else 0
    print(f"  Patience={patience_count} | Val Accuracy: {val_acc:.2%}")
    print(f"  Exit distribution: {dict(total_exits_at)}")

    return {"strategy": "patience", "thresholds": [],
            "patience": patience_count, "val_acc": val_acc,
            "metadata": {"exit_distribution": dict(total_exits_at)}}
