def train(model, data_loader, optimizer, criterion):
    model.train()
    for data in data_loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
