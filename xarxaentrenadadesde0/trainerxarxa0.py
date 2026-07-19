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
    # Classe que encapsula tot el procés d'entrenament i validació de la CNN des de zero.
    def __init__(self, model, train_loader, val_loader, device, class_names,
                 lr=None, weight_decay=1e-3):
        # Guarda el model, els dataloaders, el dispositiu (CPU/GPU) i els noms de classe
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.class_names  = class_names

        # Mateix LR base que la ResNet si no s'especifica un altre
        lr = config.LEARNING_RATE if lr is None else lr

        # Funció de pèrdua compartida amb el Trainer de la ResNet (CrossEntropyLoss)
        self.criterion = get_criterion()
        # Una sola fase: Adam sobre TOTS els paràmetres (la CNN s'entrena de zero,
        # a diferència de la ResNet que té fases de backbone congelat/fine-tuning)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.scheduler = get_scheduler(self.optimizer)   # ReduceLROnPlateau, igual que ResNet

        # Diccionari on es va acumulant l'evolució de pèrdua i accuracy per època
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   []
        }
        # Es guarda quina és la millor accuracy de validació aconseguida i en quina època
        self.best_val_acc = 0.0
        self.best_epoch   = 0

    def _train_one_epoch(self):
        # Executa una passada completa (una època) sobre el conjunt d'entrenament
        self.model.train()  # activa mode entrenament (dropout, batchnorm, etc.)
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()      # neteja gradients acumulats de l'iteració anterior
            out  = self.model(xb)            # forward pass
            loss = self.criterion(out, yb)   # càlcul de la pèrdua
            loss.backward()                  # backpropagation
            self.optimizer.step()            # actualització dels pesos

            # Acumulació de mètriques ponderades pel batch size per fer la mitjana final
            total_loss += loss.item() * xb.size(0)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += xb.size(0)
        # Retorna la pèrdua mitjana i l'accuracy de tota l'època
        return total_loss / total, correct / total

    @torch.no_grad()  # desactiva el càlcul de gradients (no cal en avaluació, estalvia memòria/temps)
    def _evaluate(self, loader):
        # Avalua el model sobre un dataloader donat (validació o test) sense actualitzar pesos
        self.model.eval()  # activa mode avaluació (desactiva dropout, fixa batchnorm)
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
        # Bucle principal d'entrenament: entrena i valida durant num_epochs èpoques,
        # guardant el millor model trobat segons l'accuracy de validació
        print(f"\n Entrenament CNN — {num_epochs} èpoques (una sola fase)")
        print("=" * 70)

        # Inicialitza l'early stopping només si està activat a la configuració
        early_stopping = None
        if getattr(config, "USE_EARLY_STOPPING", False):
            early_stopping = EarlyStopping(
                patience=config.EARLY_STOPPING_PATIENCE, min_delta=1e-4
            )

        for epoch in range(1, num_epochs + 1):
            # Una època = un pas d'entrenament + una avaluació sobre validació
            tr_loss, tr_acc   = self._train_one_epoch()
            val_loss, val_acc = self._evaluate(self.val_loader)

            # Scheduler sobre la val_loss (igual que la ResNet): redueix el LR si no millora
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Es desa l'evolució de les mètriques per poder graficar-les després
            self.history['train_loss'].append(tr_loss)
            self.history['train_acc' ].append(tr_acc)
            self.history['val_loss'  ].append(val_loss)
            self.history['val_acc'   ].append(val_acc)

            print(f"Època {epoch:3d}/{num_epochs} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"lr={current_lr:.1e}")

            # Selecció del millor model per val_acc (mateix criteri que la ResNet):
            # si aquesta època supera la millor accuracy vista fins ara, es guarda el checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch   = epoch
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch':   epoch,
                    'val_acc': val_acc,
                }, checkpoint_path)

            # Early stopping opcional (mateixa lògica que la ResNet):
            # atura l'entrenament si la val_loss no millora durant "patience" èpoques
            if early_stopping is not None and early_stopping(val_loss):
                print(f"\n Early stopping a l'època {epoch} "
                      f"(sense millora en {config.EARLY_STOPPING_PATIENCE} èpoques)")
                break

        # Un cop acabat el bucle (per èpoques exhaurides o early stopping), es mostra el resum final
        print(f"\n Millor Val Acc: {self.best_val_acc:.4f} (època {self.best_epoch})")
        print(f" Checkpoint guardat: {checkpoint_path}")
        return self.history
