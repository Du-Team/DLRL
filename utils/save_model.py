import os
import torch


def save_model(model_path, model, optimizer, current_epoch, filename=None, metrics=None):
    out = os.path.join(model_path, filename or "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    if metrics is not None:
        state['metrics'] = metrics
    torch.save(state, out)
