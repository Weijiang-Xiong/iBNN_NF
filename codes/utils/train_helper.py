import torch
import torch.optim as optim 

from models import StoModel
from .metrics import compute_accuracy, compute_ece_loss

def train_sto_model(sto_model:StoModel, trainloader=None, testloader=None, base_model=None, 
                    num_epochs=30, device=None, n_samples=128, fix_samples=False):

    if isinstance(base_model, sto_model.DET_MODEL_CLASS):
        sto_model.migrate_from_det_model(base_model)
        print("Loaded weights from a base model")

    det_params, sto_params = sto_model.det_and_sto_params()
    optimizer = optim.Adam([
                    {'params': det_params, 'lr': 2e-4},
                    {'params': sto_params, 'lr': 2e-3}
                ])

    loss_list, ll_list, kl_list, acc_list, ece_list = [[] for _ in range(5)]
    for epoch in range(num_epochs):
        sto_model.train()
        batch_loss, batch_ll, batch_kl = [[] for _ in range(3)]
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            pred = sto_model(img)
            log_likelihood, kl = sto_model.calc_loss(pred, label)
            loss = -log_likelihood + kl / len(trainloader.dataset)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
            batch_ll.append(log_likelihood.item()) 
            batch_kl.append(kl.item()/ len(trainloader.dataset))
        avg = lambda l: sum(l)/len(l)
        avg_loss, avg_ll, avg_kl = avg(batch_loss), avg(batch_ll), avg(batch_kl)
        sto_acc = compute_accuracy(sto_model, testloader, n_samples=n_samples, fix_samples=fix_samples)
        sto_ece = compute_ece_loss(sto_model, testloader, n_samples=n_samples, fix_samples=fix_samples)
        print("Sto Model Epoch {} Avg Loss {:.4f} Likelihood {:.4f} KL {:.4f} Acc {:.4f} ECE {:.4f}".format(
                            epoch, avg_loss, avg_ll, avg_kl,sto_acc, sto_ece))
        loss_list.append(avg_loss)
        ll_list.append(avg_ll)
        kl_list.append(avg_kl)
        acc_list.append(sto_acc)
        ece_list.append(sto_ece)
        sto_model.clear_stored_samples()
        
    return loss_list, ll_list, kl_list, acc_list, ece_list