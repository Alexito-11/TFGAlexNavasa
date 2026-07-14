# trainerxarxa0.py
# Entrenador per la CNN des de zero. Estructura equivalent al Trainer de la ResNet
# però amb una única fase d'entrenament (la CNN no és preentrenada, de manera que
# no calen fases de backbone congelat / fine-tuning).
#
# Per garantir la comparació en igualtat de condicions amb la ResNet, es comparteixen:
#   - Optimitzador Adam amb el mateix learning rate base (config.LEARNING_RATE) i weight_decay
#   - Scheduler ReduceLROnPlateau amb els mateixos paràmetres (via train_utils.get_scheduler)
#   - Funció de pèrdua CrossEntropyLoss (via train_utils.get_criterion)
#   - Criteri de selecció del millor model: val_acc màxima
#   - EarlyStopping opcional amb la mateixa configuració de config

import torch
import torch.optim as optim

import config
from train_utils import get_scheduler, get_criterion, EarlyStopping


class TrainerCNN:
    def __init__(self, model, train_loader, val_loader, device, class_names,
                 lr=None, weight_decay=1e-3):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.class_names  = class_names

        # Mateix LR base que la ResNet si no s'especifica un altre
        lr = config.LEARNING_RATE if lr is None else lr

        self.criterion = get_criterion()
        # Una sola fase: Adam sobre TOTS els paràmetres (la CNN s'entrena de zero)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.scheduler = get_scheduler(self.optimizer)   # ReduceLROnPlateau, igual que ResNet

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   []
        }
        self.best_val_acc = 0.0
        self.best_epoch   = 0

    def _train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            out  = self.model(xb)
            loss = self.criterion(out, yb)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += xb.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            out  = self.model(xb)
            loss = self.criterion(out, yb)

            total_loss += loss.item() * xb.size(0)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += xb.size(0)
        return total_loss / total, correct / total

    def train(self, num_epochs, checkpoint_path="best_model_cnn.pth"):
        print(f"\n Entrenament CNN — {num_epochs} èpoques (una sola fase)")
        print("=" * 70)

        early_stopping = None
        if getattr(config, "USE_EARLY_STOPPING", False):
            early_stopping = EarlyStopping(
                patience=config.EARLY_STOPPING_PATIENCE, min_delta=1e-4
            )

        for epoch in range(1, num_epochs + 1):
            tr_loss, tr_acc   = self._train_one_epoch()
            val_loss, val_acc = self._evaluate(self.val_loader)

            # Scheduler sobre la val_loss (igual que la ResNet)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(tr_loss)
            self.history['train_acc' ].append(tr_acc)
            self.history['val_loss'  ].append(val_loss)
            self.history['val_acc'   ].append(val_acc)

            print(f"Època {epoch:3d}/{num_epochs} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"lr={current_lr:.1e}")

            # Selecció del millor model per val_acc (mateix criteri que la ResNet)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch   = epoch
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch':   epoch,
                    'val_acc': val_acc,
                }, checkpoint_path)

            # Early stopping opcional (mateixa lògica que la ResNet)
            if early_stopping is not None and early_stopping(val_loss):
                print(f"\n Early stopping a l'època {epoch} "
                      f"(sense millora en {config.EARLY_STOPPING_PATIENCE} èpoques)")
                break

        print(f"\n Millor Val Acc: {self.best_val_acc:.4f} (època {self.best_epoch})")
        print(f" Checkpoint guardat: {checkpoint_path}")
        return self.history