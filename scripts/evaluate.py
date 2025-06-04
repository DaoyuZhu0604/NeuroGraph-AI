def evaluate(model, data_loader):
    model.eval()
    results = []
    for data in data_loader:
        output = model(data.x, data.edge_index)
        results.append(output)
    return results
