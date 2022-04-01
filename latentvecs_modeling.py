import os
import numpy as np
from sklearn import mixture

def main(method, prefix):
    
    # latentvec_dir = 'outputs/generate_results/im2latent_save_dir_v2/latentvecs/'
    latentvec_dir = prefix + '/latentvecs/'

    # load latent vectors npy file
    for i,filename in enumerate(os.listdir(latentvec_dir)):
        if i == 0:
            latent_vecs = np.load(latentvec_dir + filename)
        else:
            current_vecs = np.load(latentvec_dir + filename)
            latent_vecs = np.concatenate((current_vecs,latent_vecs),axis=0)

    print('latentvecs shape: ', latent_vecs.shape)
    if method == 'shift gauss':
        # compute mean and cov of latent_vecs
        mean = np.mean(latent_vecs, axis= 0)
        cov = np.cov(latent_vecs, rowvar=False)

        np.save(prefix+'/mean.npy', mean)
        print('mean shape', mean.shape)
        np.save(prefix+'/cov.npy', cov)
        print('cov shape', cov.shape)
        print('mean',np.mean(mean))
        print('var',np.mean(cov.diagonal()))

if __name__  == '__main__':
    prefix = 'output/vec2img/lsun_kitchen_mini_train_batch_mode'
    method = 'shift gauss'
    main(method, prefix)