import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

from datasets.fmri_dataset import get_full_dataset
from models.citca import CITCA
from utils import set_seed


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    probabilities = []
    targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.batch, data.text_tokens).view(-1)
            target = data.y.float()
            loss_sum += criterion(logits, target).item() * target.size(0)
            sample_count += target.size(0)
            probabilities.append(torch.sigmoid(logits).cpu())
            targets.append(target.cpu())

    probabilities = torch.cat(probabilities).numpy()
    targets = torch.cat(targets).numpy()
    predictions = (probabilities > 0.5).astype(np.float32)
    accuracy = float((predictions == targets).mean())
    try:
        auc = float(roc_auc_score(targets, probabilities))
    except ValueError:
        auc = 0.0
    f1 = float(f1_score(targets, predictions, zero_division=0))

    true_positive = int(((predictions == 1) & (targets == 1)).sum())
    true_negative = int(((predictions == 0) & (targets == 0)).sum())
    false_positive = int(((predictions == 1) & (targets == 0)).sum())
    false_negative = int(((predictions == 0) & (targets == 1)).sum())
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)

    return {
        "loss": loss_sum / sample_count,
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def labels_from_dataset(dataset):
    labels = []
    for sample in dataset.data:
        raw_label = int(sample["label"])
        labels.append(0 if raw_label == 2 else raw_label)
    return np.asarray(labels, dtype=np.int64)


def positive_weight(labels, indices, device):
    subset_labels = labels[indices]
    positive_count = int((subset_labels == 1).sum())
    negative_count = int((subset_labels == 0).sum())
    value = negative_count / positive_count if positive_count else 1.0
    print(
        "[Loss] training class count -> "
        f"negative: {negative_count}, positive: {positive_count}, "
        f"pos_weight: {value:.4f}"
    )
    return torch.tensor(value, dtype=torch.float32, device=device)


class Experiment:
    def __init__(self, args, output_dir="./output"):
        self.args = args
        self.output_dir = output_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)

    def build_model(self):
        return CITCA(
            input_dim=self.dataset.num_rois,
            output_dim=1,
            **vars(self.args),
        ).to(self.device)

    def build_optimizer(self, model):
        classifier_ids = {id(parameter) for parameter in model.classifier.parameters()}
        fusion_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in classifier_ids
        ]
        return torch.optim.AdamW(
            [
                {"params": fusion_parameters, "lr": self.args.lr_fusion},
                {"params": model.classifier.parameters(), "lr": self.args.lr_clf},
            ],
            weight_decay=self.args.wd,
        )

    def train(self):
        if self.args.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self.dataset = get_full_dataset(
            self.args.data_path,
            self.args.text_path,
            threshold=self.args.threshold,
        )
        print(f"Device: {self.device}; subjects: {len(self.dataset)}; ROIs: {self.dataset.num_rois}")
        return self._run_kfold()

    def _run_kfold(self):
        labels = labels_from_dataset(self.dataset)
        splitter = StratifiedKFold(
            n_splits=self.args.n_folds,
            shuffle=True,
            random_state=self.args.seed,
        )
        fold_metrics = []
        start_time = time.time()

        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels), start=1
        ):
            print(f"\n{'=' * 20} Fold {fold}/{self.args.n_folds} {'=' * 20}")
            set_seed(self.args.seed + fold)
            inner_train_indices, validation_indices = train_test_split(
                train_indices,
                test_size=0.15,
                random_state=self.args.seed,
                stratify=labels[train_indices],
            )
            train_loader = DataLoader(
                torch.utils.data.Subset(self.dataset, inner_train_indices),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
            validation_loader = DataLoader(
                torch.utils.data.Subset(self.dataset, validation_indices),
                batch_size=self.args.batch_size,
                shuffle=False,
            )
            test_loader = DataLoader(
                torch.utils.data.Subset(self.dataset, test_indices),
                batch_size=self.args.batch_size,
                shuffle=False,
            )

            model = self.build_model()
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=positive_weight(
                    labels, inner_train_indices, self.device
                )
            )
            optimizer = self.build_optimizer(model)
            checkpoint_path = os.path.join(
                self.output_dir, f"best_model_fold_{fold}.pth"
            )
            best_validation_accuracy = -1.0

            for epoch in range(1, self.args.epochs + 1):
                model.train()
                train_loss_sum = 0.0
                train_correct = 0
                train_samples = 0
                for data in train_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    logits = model(data.x, data.batch, data.text_tokens).view(-1)
                    target = data.y.float()
                    loss = criterion(logits, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss_sum += loss.item() * target.size(0)
                    train_correct += ((logits > 0).float() == target).sum().item()
                    train_samples += target.size(0)

                validation = evaluate(
                    model, validation_loader, criterion, self.device
                )
                if validation["accuracy"] > best_validation_accuracy:
                    best_validation_accuracy = validation["accuracy"]
                    torch.save(model.state_dict(), checkpoint_path)

                if epoch == 1 or epoch % 10 == 0 or epoch == self.args.epochs:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train loss/acc: {train_loss_sum / train_samples:.3f}/"
                        f"{train_correct / train_samples:.3f} | "
                        f"val loss/acc/auc/f1: {validation['loss']:.3f}/"
                        f"{validation['accuracy']:.3f}/{validation['auc']:.3f}/"
                        f"{validation['f1']:.3f}"
                    )

            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            test = evaluate(model, test_loader, criterion, self.device)
            test["best_validation_accuracy"] = best_validation_accuracy
            fold_metrics.append(test)
            print(
                f"Fold {fold} test ACC/AUC/F1: {test['accuracy']:.4f}/"
                f"{test['auc']:.4f}/{test['f1']:.4f}"
            )

        summary = {"folds": fold_metrics}
        for key in (
            "best_validation_accuracy",
            "accuracy",
            "auc",
            "f1",
            "sensitivity",
            "specificity",
        ):
            values = np.asarray([fold[key] for fold in fold_metrics])
            summary[f"mean_{key}"] = float(values.mean())
            summary[f"std_{key}"] = float(values.std())
        summary["execution_time_seconds"] = time.time() - start_time

        print("\nK-fold summary")
        print(
            f"ACC: {summary['mean_accuracy']:.4f} "
            f"(±{summary['std_accuracy']:.4f})"
        )
        print(
            f"AUC: {summary['mean_auc']:.4f} "
            f"(±{summary['std_auc']:.4f})"
        )
        print(
            f"F1 : {summary['mean_f1']:.4f} "
            f"(±{summary['std_f1']:.4f})"
        )
        return summary
