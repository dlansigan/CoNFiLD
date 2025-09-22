# read in the data
import numpy as np 
import torch

def ReconstructFrame(data, mask, shape, fill_value = np.nan ):
    temp_data = np.empty((*shape,data.shape[-1]))
    temp_data[:] = fill_value
    temp_data[mask] = data
    return temp_data

def pass_through_model_batch(coords, latents, model, x_normalizer, y_normalizer, batch_size, device):
    t_size, latent_size= latents.shape
    m_size, coords_size= coords.shape
    if t_size % batch_size == 0 :
        num_batches = t_size // batch_size 
    else: 
        num_batches = t_size // batch_size + 1
    output_all= []
    for i in range(num_batches):
        sid = int(i*batch_size)
        if i < num_batches-1:
            eid = int((i+1)*batch_size)
        else: 
            eid = int(t_size)
        batch_latent = latents[sid:eid].reshape(-1,1,latent_size)
        batch_coords = coords.reshape(1,m_size,coords_size).to(device)  #<1, meshsize, cin>
        batch_output = y_normalizer.denormalize(model(x_normalizer.normalize(batch_coords),batch_latent))
        #<batch, meshsize, cout>
        output_all.append(batch_output)
    output_all = torch.cat(output_all, dim = 0)
    return output_all

def decoder(coords, latents, model, x_normalizer, y_normalizer, batch_size, device):
    t_size, latent_size= latents.shape
    m_size, coords_size= coords.shape
    if t_size % batch_size == 0 :
        num_batches = t_size // batch_size 
    else: 
        num_batches = t_size // batch_size + 1
    output_all= []
    with torch.no_grad():
        for i in range(num_batches):
            sid = int(i*batch_size)
            if i < num_batches-1:
                eid = int((i+1)*batch_size)
            else: 
                eid = int(t_size)
            batch_latent = latents[sid:eid].reshape(-1,1,latent_size)
            batch_coords = coords.reshape(1,m_size,coords_size).to(device)  #<1, meshsize, cin>
            if batch_coords.size()[-1]<=3:
                input = x_normalizer.normalize(batch_coords)
            else: # Handle if SDF is used
                if x_normalizer is None:
                    coords_normed = batch_coords[:,:,:3]
                else:
                    coords_normed = x_normalizer.normalize(batch_coords[:,:,:3])
                input = torch.cat([coords_normed,batch_coords[:,:,[3]]],axis=2)
            batch_output = y_normalizer.denormalize(
                model(input,batch_latent))
            #<batch, meshsize, cout>
            output_all.append(batch_output.cpu())
        output_all = torch.cat(output_all, dim = 0)
    return output_all
