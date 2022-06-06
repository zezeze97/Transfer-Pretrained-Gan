from gan_training.models import (
    resnet, resnet2, resnet3, resnet4,
)

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'resnet2_omit_class_embedding': resnet2.Generator_Omit_ClassEmbedding,
    'resnet2_small': resnet2.Generator_Small,
    'resnet3': resnet3.Generator,
    'resnet4': resnet4.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
    'resnet3': resnet3.Discriminator,
    'resnet4': resnet4.Discriminator,
}

im2latent_model_dict = {
    'resnet3': resnet3.Im2Latent
}
