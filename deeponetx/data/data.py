"""
Data type. Inspired from https://github.com/lululxvi/deepxde/blob/master/deepxde/data/data.py
"""
import equinox as eqx
import abc 

class Data(eqx.Module):
    """Data base class"""

    def losses(self, targets, outputs, loss_fn, inputs, model, **kwargs):
        "Return a list of losses. i.e. boundary losses, inital loss"
        raise NotImplementedError("Data.losses is not implemented.")
    
    def losses_train(self, targets, outputs, loss_fn, inputs, model, **kwargs):
        return self.losses(targets, outputs, loss_fn, inputs, model)
    
    def losses_test(self, targets, outputs, loss_fn, inputs, model, **kwargs):
        return self.losses(targets, outputs, loss_fn, inputs, model)
    
    @abc.abstractmethod 
    def train_next_batch(self, batch_size=None):
        "Return a training dataset of the size `batch_size`"
    
    @abc.abstractmethod 
    def test(self):
        "Return a test dataset"