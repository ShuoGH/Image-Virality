import os
import os.path
import errno


def makedir_exist_ok(dirpath):
    """
    mkdir, to support the Reddit_Img_Pair data set
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def freeze_pretrained(model):
    '''
    Freeze the pretrained alexnet layers. 
    For now, the layers which need to be frozen is the `STN.alex_conv`
    '''
    for param in model.branch.locnet.alex_conv.parameters():
        param.requires_grad = False
    return model
