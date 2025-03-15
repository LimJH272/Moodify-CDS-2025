import torch.utils.data.dataloader
import torch, torchaudio, torchvision
import torchmetrics

import pytorch_helpers as pth
import models
import datasets

class MultiClassTrainer:
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
            model.train()
            curr_loss_history = []
            curr_accuracy_history = []

            for melspecs, labels in train_loader:
                melspecs = melspecs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                preds = model(melspecs)
                loss = self.loss(preds, labels) + lambda_val * self.regularisation_term(model)

                loss.backward()
                optimizer.step()

                accuracy = self.accuracy(preds, labels)
                curr_loss_history.append(loss.item())
                curr_accuracy_history.append(accuracy.item())
            
            all_preds, all_labels = self.predict(model, train_loader)   # model.eval() is called here
            epoch_loss = self.evaluate_loss(model, all_preds, all_labels, lambda_val)
            epoch_accuracy = self.evaluate_accuracy(all_preds, all_labels)
            self.loss_history.append((curr_loss_history, epoch_loss))
            self.accuracy_history.append((curr_accuracy_history, epoch_accuracy))

            print(f'Epoch {i+1}    Loss={epoch_loss:.4f}    Accuracy={epoch_accuracy :.4f}')
        
        model.eval()
        print('Training finished')
    
    def predict(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        model.eval()

        with torch.no_grad():
            preds_labels = [(model.to(self.device)(melspec.to(self.device)), label.to(self.device)) for melspec, label in dataloader]
            pred_ls, labels_ls = zip(*preds_labels)
            preds = torch.cat(pred_ls)
            labels = torch.cat(labels_ls)

        return preds, labels

    def evaluate_accuracy(self, preds: torch.Tensor, labels: torch.Tensor):
        return self.accuracy(preds, labels).item()

    def evaluate_loss(self, model: torch.nn.Module, preds: torch.Tensor, labels: torch.Tensor, lambda_val=0.0):
        return (self.loss(preds, labels).item() + lambda_val * self.regularisation_term(model)).item()

    def regularisation_term(self, model: torch.nn.Module):
        return sum(torch.abs(param).sum() for param in model.parameters())


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = datasets.SoundTracksDataset()
    dataset = dataset.NHW_to_NCHW()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = models.NilsHMeierCNN().to(device)
    print(model)

    trainer = MultiClassTrainer(device)
    preds, labels = trainer.predict(model, dataloader)
    print(trainer.evaluate_accuracy(preds, labels))

    trainer.train(model, dataloader, lr=0.001, max_epochs=20, lambda_val=0.001)
    preds, labels = trainer.predict(model, dataloader)
    print(trainer.evaluate_accuracy(preds, labels))


    