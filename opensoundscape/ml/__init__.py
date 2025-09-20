from . import cam
from . import cnn_architectures
from . import cnn
from . import datasets
from . import loss
from . import safe_dataset
from . import sampling
from . import utils
import torch.multiprocessing

# using 'file_system' avoids errors with "Too many open files",
# "Pin memory thread exited unexpectedly", and RuntimeError('received %d items of ancdata')
# when using parallelized DataLoader. This is the recommended solution according to
# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
torch.multiprocessing.set_sharing_strategy("file_system")
