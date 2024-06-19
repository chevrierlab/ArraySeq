
import warnings
from math import floor
import numpy as np
import pandas as pd
import skimage

warnings.filterwarnings('ignore')

def check_arrayseq_signature(input_signature, accepted_signature=None, accepted_signatures=None):
    if accepted_signature is not None:
        if input_signature == accepted_signature:
            pass
        elif accepted_signature == "Auto Aligned" or accepted_signature == "Manually Aligned":
            raise ValueError("Unexpected adata signature. Expected auto-aligned or manually aligned adata. Check that the adata object was generated from the ___ or ___ functions.")
        elif accepted_signature == "Full ArraySeq Object":
            raise ValueError("Unexpected adata signature. Expected a full ArraySeq object. Check that the adata object was generated from the ___ function.")
        elif accepted_signature == "Separated ArraySeq Object":
            raise ValueError("Unexpected adata signature. Expected a full ArraySeq object. Check that the adata object was generated from the ___ function.")
        else:
            raise ValueError("Unexpected adata signature.")
    elif accepted_signatures is not None:
        if input_signature in accepted_signatures:
            pass
        else:
            raise ValueError("Unexpected adata signature.")
    else:
        raise ValueError("Either accepted_signature or accepted_signatures must be provided.")
