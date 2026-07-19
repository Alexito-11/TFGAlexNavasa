# trainer.py
# entrenament de la resnet18 pre-entrenada en dues fases: primer nomes el cap
# amb el backbone congelat, i despres fine-tuning complet amb lr diferencial

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_utils import EarlyStopping, get_optimizer, get_scheduler, get_criterion
import config


class Trainer:
    """
    Entrenament en dues fases per a ResNet18 pre-entrenada:

    Fase 1 — Backbone congelat (epochs_phase1 èpoques):
        Entrena només intermediate + classifier.

    Fase 2 — Fine-tuning complet:
        Descongela el backbone amb lr diferencial.
    """

    def __init__(self, model, device, save_path="best_model_resnet18.pth"):
        self.model     = model
        self.device    = device
        self.save_path = save_path
        self.criterion = get_criterion()

        # comenco sempre amb el backbone congelat (fase 1)
        self._freeze_backbone()
        self.optimizer = get_optimizer(model, unfreeze_backbone=False)
        self.scheduler = get_scheduler(self.optimizer)

        if config.USE_EARLY_STOPPING:
            self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
        else:
            self.early_stopping = None

        self.history = {
            "accuracy": [], "val_accuracy": [],
            "loss":     [], "val_loss":     [],
            "phase":    []
        }
        self.best_val_acc = 0.0
        self.best_epoch   = 0

    def _freeze_backbone(self):
        """Congela totes les capes convolucionals"""
        backbone_layers = [
            self.model.conv1, self.model.bn1,
            self.model.layer1, self.model.layer2,
            self.model.layer3, self.model.layer4
        ]
        for layer in backbone_layers:
            for param in layer.parameters():
                param.requires_grad = False

        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Backbone congelat — entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def _unfreeze_backbone(self):
        """Descongela el backbone per a fine-tuning"""
        backbone_layers = [
            self.model.conv1, self.model.bn1,
            self.model.layer1, self.model.layer2,
            self.model.layer3, self.model.layer4
        ]
        for layer in backbone_layers:
            for param in layer.parameters():
                param.requires_grad = True

        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Backbone descongelat — entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def train_epoch(self, train_loader):
        # una passada completa sobre el train, amb backprop
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(xb)
            loss    = self.criterion(outputs, yb)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, 1)
            total   += yb.size(0)
            correct += (preds == yb).sum().item()

        return running_loss / total, correct / total

    def validate_epoch(self, val_loader):
        # igual que train_epoch pero sense actualitzar pesos (no_grad)
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss    = self.criterion(outputs, yb)
                running_loss += loss.item() * xb.size(0)
                _, preds = torch.max(outputs, 1)
                total   += yb.size(0)
                correct += (preds == yb).sum().item()

        return running_loss / total, correct / total

    def save_checkpoint(self, epoch, val_acc, val_loss):
        # nomes guardo si millora la val_acc, aixi sempre tinc el millor model al disc
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc':              val_acc,
                'val_loss':             val_loss,
                'best_val_acc':         self.best_val_acc
            }, self.save_path)
            print(f"  Millor model guardat! Val Acc: {val_acc:.4f}")
            return True
        return False

    def train(self, train_loader, val_loader,
              num_epochs=config.NUM_EPOCHS,
              epochs_phase1=10):

        print(f"\n{'='*60}")
        print(f"FASE 1: Backbone congelat ({epochs_phase1} èpoques)")
        print(f"{'='*60}")

        for epoch in range(1, num_epochs + 1):

            # quan acaben les epoques de fase 1, descongelo i canvio optimizer/scheduler
            if epoch == epochs_phase1 + 1:
                print(f"\n{'='*60}")
                print(f"FASE 2: Fine-tuning complet (backbone descongelat)")
                print(f"{'='*60}")
                self._unfreeze_backbone()
                self.optimizer = get_optimizer(self.model, unfreeze_backbone=True)
                self.scheduler = get_scheduler(self.optimizer)
                if self.early_stopping is not None:
                    # reinicio l'early stopping perque la loss "salta" en canviar de fase
                    self.early_stopping = EarlyStopping(
                        patience=config.EARLY_STOPPING_PATIENCE
                    )

            phase = 1 if epoch <= epochs_phase1 else 2

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss,   val_acc   = self.validate_epoch(val_loader)

            self.scheduler.step(val_loss)

            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["phase"].append(phase)

            print(f"[F{phase}] Epoch {epoch:3d}/{num_epochs} | "
                  f"loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

            self.save_checkpoint(epoch, val_acc, val_loss)

            if self.early_stopping is not None and self.early_stopping(val_loss):
                print(f"\nEarly stopping activat a l'època {epoch}")
                break

        print(f"\n{'='*60}")
        print(f"MILLOR MODEL: Època {self.best_epoch} | Val Acc: {self.best_val_acc:.4f}")
        print(f"{'='*60}")
        return self.history

    def load_best_model(self):
        # carrego el checkpoint amb millor val_acc, per fer l'avaluacio final amb ell
        checkpoint = torch.load(self.save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n Millor model carregat "
              f"(Època {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f})")
        return checkpoint

    def plot_history(self):
        # grafica accuracy i loss de train/val, marcant on comença la fase 2
        # i on esta el millor model
        epochs       = range(1, len(self.history['accuracy']) + 1)
        phase_change = next(
            (i + 1 for i, p in enumerate(self.history['phase']) if p == 2), None
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, (train_key, val_key), title in zip(
            axes,
            [('accuracy', 'val_accuracy'), ('loss', 'val_loss')],
            ['Accuracy', 'Loss']
        ):
            ax.plot(epochs, self.history[train_key], label='Train')
            ax.plot(epochs, self.history[val_key],   label='Val')
            if phase_change:
                ax.axvline(x=phase_change, color='purple', linestyle='--',
                           label=f'Inici fine-tuning (è.{phase_change})')
            ax.axvline(x=self.best_epoch, color='green', linestyle=':',
                       label=f'Millor model (è.{self.best_epoch})')
            ax.set_title(title)
            ax.set_xlabel('Època')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("  Gràfica guardada: training_history.png")
