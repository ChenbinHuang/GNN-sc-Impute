import torch
import numpy as np
from tqdm import trange
from torch.utils import data
def spilt_dataset(train_all_data, shuffle_index):
    """
    SPlit the data set into test,train validate sets
    """
    train_size, val_size = int(len(shuffle_index)* 0.8), int(len(shuffle_index)* 0.9)
    train_data = np.asarray(train_all_data).astype(np.float32)[shuffle_index[0:train_size],:]
    val_data = np.asarray(train_all_data).astype(np.float32)[shuffle_index[train_size:val_size],:]
    test_data = np.asarray(train_all_data).astype(np.float32)[shuffle_index[val_size:],:]

    return train_data, val_data, test_data
    
def generate_loader(train_data, val_data, test_data, batchsize):
    """
    from the result of split datasets, change the data into loaders
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = torch.FloatTensor(train_data).to(device)
    val_data = torch.FloatTensor(val_data).to(device)
    test_data = torch.FloatTensor(test_data).to(device)
    
    dset_train = data.TensorDataset(train_data)
    train_loader = data.DataLoader(dset_train, batch_size = batchsize, shuffle = True)
    dset_val = data.TensorDataset(val_data)
    val_loader = data.DataLoader(dset_val, batch_size = len(dset_val), shuffle = False)
    dset_test = data.TensorDataset(test_data)
    test_loader = data.DataLoader(dset_test, batch_size = len(dset_test), shuffle = False)
    return train_loader, val_loader, test_loader


def train(model, train_loader, test_loader, optimizor, adj, args):

    stats = dict(epoch=[], loss=[])
    best_loss = float('inf')
    for epoch in trange(args.epochs):
        total_loss = 0
        model.train()
        # for batch_idx, (data, dataindex) in enumerate(train_loader)
        for batch in train_loader:
            optimizor.zero_grad()

            z = model.encode(batch[0].view(-1,1), adj)
            # print(z.shape)
            loss = model.recon_loss(z, adj)

            loss.backward()
            optimizor.step()
            total_loss += loss.item()
        total_loss /= len(train_loader.dataset)

        if epoch % 10 == 0:
            test_loss = test(test_loader, model, adj)
            # print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
            #     epoch, total_loss, test_acc))
            stats["epoch"].append(epoch)
            stats["loss"].append(test_loss)
            # stats["acc"].append(test_acc)
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), "./tmp/model_GAE.pt")

    return {"model":model, "stats":stats}

def test(test_loader, model, adj):
    model.eval()

    total = 0
    for batch in test_loader:
        with torch.no_grad():
            z = model.encode(batch[0].view(-1,1), adj)
            loss = model.recon_loss(z, adj)
            total += loss
    total /= len(test_loader.dataset)
    return total