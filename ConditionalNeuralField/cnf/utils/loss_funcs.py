import numpy as np
import torch

def maskedMSELoss(output, fois, sdf):
    mask = sdf > 0 # Get mask from SDF
    mask = mask.expand_as(fois).float() # Reshape for multiplying to squared error
    sq_err = (output-fois)**2 # Get squared error
    masked_err = sq_err * mask # Apply mask
    if mask.sum().item() > 0:
        loss = masked_err.sum()/mask.sum().item() # Compute MSE only inside geometry
    else: 
        loss = masked_err.sum()*0.0 # Outside geometry, return 0 so NN doesn't learn there
    return loss





