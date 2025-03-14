import torch.utils.data.dataloader
import torch, torchaudio
import torchmetrics

import pytorch_helpers as pth
import models
import datasets

class ModelTrainer:
    def __init__(self, device):
        self.device = device
        self.loss = torch.nn.CrossEntropyLoss().to(device)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=4).to(device)

    def train(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, max_epochs=5, lr=1e-3, lambda_val=0.0):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model = model.to(self.device)
        model.train()

        self.loss_history = []
        self.accuracy_history = []

        print("Training start")

        for i in range(max_epochs):
            for j, (melspecs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                preds = model(melspecs.to(self.device))
                loss = self.loss(preds, labels.to(self.device)) + lambda_val * sum(torch.abs(param).sum() for param in model.parameters()).item()
                self.loss_history.append(loss.cpu())

                loss.backward()
                optimizer.step()
            
                accuracy = self.accuracy(preds, labels.to(self.device)).item()
                print(f'Epoch {i+1}    Batch {j+1}    Loss={loss:.4f}    Accuracy={accuracy :.4f}')
        
        model.eval()
        print('Training finished')

    def evaluate_accuracy(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader):
        model.eval()
        preds_labels = [(model.to(self.device)(melspec.to(self.device)), label.to(self.device)) for melspec, label in train_loader]
        pred_ls, labels_ls = zip(*preds_labels)
        preds = torch.cat(pred_ls)
        labels = torch.cat(labels_ls)
        
        return self.accuracy(preds, labels).item()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = datasets.SoundTracksDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = models.MelSpecCNN().to(device)

    trainer = ModelTrainer(device)
    print(trainer.evaluate_accuracy(model, dataloader))
    trainer.train(model, dataloader, lr=0.1)
    print(trainer.evaluate_accuracy(model, dataloader))


    