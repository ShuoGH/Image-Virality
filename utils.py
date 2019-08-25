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
