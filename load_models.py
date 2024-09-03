from models.encoder import BaseModel as EctEncoder
from models.vae import BaseModel as VAE

def load_encoder(encoder_path):
    model = EctEncoder.load_from_checkpoint(encoder_path)
    return model


def load_vae(modelpath):
    model = VAE.load_from_checkpoint(modelpath)
    return model
