from utils import *
import numpy as np

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, epoch, writer, report_iters=10, num_pixels=32*32*3):
    size = len(dataloader)
    avg_loss = 0
    for batch, (X, _) in enumerate(dataloader):
        # Transfer to GPU
        X = pre_process(X)
        X = X.to(cfg["device"])
        
        # Compute prediction and loss
        y, s, norms, scale = model(X)
        loss, comps = loss_fn(y, s, norms, scale, batch_size)

        if torch.any(torch.isinf(loss)):
            continue

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        loss, current = loss.item(), batch
        # Account for preprocessing
        loss -= torch.sum(log_preprocessing_grad(X)) / batch_size
        avg_loss += loss
        
        if batch % report_iters == 0:
            print(f"loss: {loss:.2f} = -logpx[{comps[0]:.1f}], -det[{comps[1]:.1f}], -norms[{comps[2]:.1f}], reg[{comps[3]:.4f}]"
                  f"; bits/pixel: {loss / num_pixels / np.log(2):>.2f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar("Learning Rate", optimizer.state_dict()["param_groups"][0]["lr"], size*epoch+current)
            writer.add_scalar("Loss/train", loss, size*epoch+current)
            writer.add_scalar("BPP/train", (avg_loss / num_pixels / np.log(2)), size*epoch+current)
        
    avg_loss = avg_loss / (batch + 1)
    print(f"Train Error: \n Avg loss: {avg_loss:.2f}; {avg_loss / num_pixels / np.log(2):.2f} \n")
    
    return avg_loss / num_pixels / np.log(2)
            
        
def test_loop(dataloader, model, loss_fn, epoch, writer, num_pixels=32*32*3):
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        model.validate()
        for X, _ in dataloader:
            X = pre_process(X)
            X = X.to(cfg["device"])
            y, s, norms, scale = model(X)
            loss, _ = loss_fn(y, s, norms, scale, cfg["batch_size"])
            test_loss += loss
            test_loss -= torch.sum(log_preprocessing_grad(X)) / cfg["batch_size"]
        model.train()

    test_loss /= num_batches
    # Account for preprocessing
    print(f"Test Error: \n Avg loss: {test_loss:.2f}; {test_loss / num_pixels / np.log(2):.2f} \n")
    writer.add_scalar("Loss/test", test_loss, )
    writer.add_scalar("BPP/test", (test_loss / num_pixels / np.log(2)), size*epoch)
    return test_loss

def dual_train_loop(dataloader, ood_dataloader, model, loss_fn, ood_loss_fn, optimizer, batch_size, epoch, writer, report_iters=10, num_pixels=32*32*3):
    size = len(dataloader)
    avg_loss = 0

    data_iter = iter(dataloader)
    ood_iter = iter(ood_dataloader)

    batch = 0

    device = cfg["device"]

    while True:
        try:
            data, _ = next(data_iter)
            ood, _ = next(ood_iter)
            batch += 1
        except StopIteration:
            break

        # Transfer to GPU
        X = pre_process(data)
        X = X.to(device)

        X_ood = pre_process(ood)
        X_ood = X_ood.to(device)
        
        # Compute prediction and loss
        y, s, norms, scale = model(X)
        loss_in, comps_in = loss_fn(y, s, norms, scale, batch_size)

        if torch.any(torch.isinf(loss_in)):
            continue
        
        # Compute prediction and loss
        y, s, norms, scale = model(X_ood)
        ood_loss_out, ood_comps_out = ood_loss_fn(y, s, norms, scale, batch_size)
        loss_out, comps_out = loss_fn(y, s, norms, scale, batch_size)

        if torch.any(torch.isinf(ood_loss_out)):
            continue

        loss = loss_in + ood_loss_out

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        loss, current = loss.item(), batch
        # Account for preprocessing
        loss -= torch.sum(log_preprocessing_grad(X)) / batch_size
        avg_loss += loss
        
        if batch % report_iters == 0:
            print(f"loss: {loss_in:.2f} = -logpx[{comps_in[0]:.1f}], -det[{comps_in[1]:.1f}], -norms[{comps_in[2]:.1f}], reg[{comps_in[3]:.4f}]"
                  f"; bits/pixel: {loss / num_pixels / np.log(2):>.2f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar("Learning Rate", optimizer.state_dict()["param_groups"][0]["lr"], size*epoch+current)
            writer.add_scalar("Gaussian Loss (in)/train", loss_in, size*epoch+current)
            writer.add_scalar("Gaussian Loss (out)/train", loss_out, size*epoch+current)
            writer.add_scalar("OOD Loss (out)/train", ood_loss_out, size*epoch+current)
            writer.add_scalar("BPP (in)/train", (loss_in / num_pixels / np.log(2)), size*epoch+current)
            writer.add_scalar("BPP (out)/train", (ood_loss_out / num_pixels / np.log(2)), size*epoch+current)
        
    avg_loss = avg_loss / (batch + 1)
    print(f"Train Error: \n Avg loss: {avg_loss:.2f}; {avg_loss / num_pixels / np.log(2):.2f} \n")
    
    return avg_loss / num_pixels / np.log(2)


def dual_test_loop(dataloader, ood_dataloader, model, loss_fn, ood_loss_fn, epoch, writer, num_pixels=32*32*3):
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss = 0
    test_loss_ood = 0
    test_ring_loss_ood = 0

    batch_size = cfg["batch_size"] // 2

    with torch.no_grad():
        model.validate()
        for X, _ in dataloader:
            X = pre_process(X)
            X = X.to(cfg["device"])
            y, s, norms, scale = model(X)
            loss, _ = loss_fn(y, s, norms, scale, batch_size)
            test_loss += loss
            test_loss -= torch.sum(log_preprocessing_grad(X)) / batch_size

        for X, _ in ood_dataloader:
            X = pre_process(X)
            X = X.to(cfg["device"])
            y, s, norms, scale = model(X)
            loss_ood, _ = loss_fn(y, s, norms, scale, batch_size)
            test_loss_ood += loss_ood
            test_loss_ood -= torch.sum(log_preprocessing_grad(X)) / batch_size

        for X, _ in ood_dataloader:
            X = pre_process(X)
            X = X.to(cfg["device"])
            y, s, norms, scale = model(X)
            ring_loss_ood, _ = ood_loss_fn(y, s, norms, scale, batch_size)
            test_ring_loss_ood += ring_loss_ood
            test_ring_loss_ood -= torch.sum(log_preprocessing_grad(X)) / batch_size

        model.train()

    test_loss /= num_batches
    test_loss_ood /= num_batches
    test_ring_loss_ood /= num_batches
    # Account for preprocessing
    print(f"Test Error: \n Avg loss: {test_loss:.2f}; {test_loss / num_pixels / np.log(2):.2f} \n")
    print(f"Test Error (OOD): \n Avg loss: {test_loss_ood:.2f}; {test_loss_ood / num_pixels / np.log(2):.2f} \n")
    print(f"Ring Test Error (OOD): \n Avg loss: {test_ring_loss_ood:.2f}; {test_ring_loss_ood / num_pixels / np.log(2):.2f} \n")
    writer.add_scalar("Gaussian Loss (in)/test", test_loss, size*epoch)
    writer.add_scalar("Gaussian Loss (out)/test", test_loss_ood, size*epoch)
    writer.add_scalar("Ring Loss (out)/test", test_ring_loss_ood, size*epoch)
    writer.add_scalar("BPP (in)/test", (test_loss / num_pixels / np.log(2)), size*epoch)
    return test_loss + test_ring_loss_ood