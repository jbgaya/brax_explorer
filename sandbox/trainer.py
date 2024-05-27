import torch

class Trainer:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

    def iterate(self):
        """
        Perform one epoch of training and evaluation
        """
        self.model.train()
        total_acc = 0
        total_count = 0
        losses = []
        for idx, (label, text, offsets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
        train_loss = sum(losses) / len(losses)
        eval_metrics = self.eval()
        
        return {
            'train_loss': train_loss,
            'train_accuracy': total_acc / total_count,
            'val_loss': eval_metrics['loss'],
            'val_accuracy': eval_metrics['accuracy']
        }

    def eval(self):
        self.model.eval()
        total_acc = 0
        total_count = 0
        losses = []
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(self.val_loader):
                predicted_label = self.model(text, offsets)
                loss = self.criterion(predicted_label, label)
                losses.append(loss.item())
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
            val_loss = sum(losses) / len(losses)
        return {'accuracy': total_acc / total_count, 'loss': val_loss}
