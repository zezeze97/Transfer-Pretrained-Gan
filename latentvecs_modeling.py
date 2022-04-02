import os
import numpy as np
from sklearn import mixture
from scipy.stats import normaltest
import matplotlib.pyplot as plt

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
    
    if method == 'gauss mixture':
        # select best gmm model
        aic_list = []
        best_aic = np.infty
        best_n_components = 0
        for n_components in range(10,150,10):
            model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=2, verbose_interval=1)
            gm = model.fit(latent_vecs)
            aic = model.aic(latent_vecs)
            print('the aic of ',n_components,' is ', aic)
            aic_list.append(aic)
            if aic < best_aic:
                best_aic = aic
                best_n_components = n_components
        print("best num of components is ", best_n_components," aic is ", best_aic)
        model = mixture.GaussianMixture(n_components=best_n_components, covariance_type='full', verbose=2, verbose_interval=1)
        gm = model.fit(latent_vecs)
        np.save(prefix+'/gmm_components_weights.npy', gm.weights_)
        np.save(prefix+'/gmm_mean.npy', gm.means_)
        np.save(prefix+'/gmm_cov.npy', gm.covariances_)
    if method == 'normal test':
        p_value_list = []
        for latentdim in range(latent_vecs.shape[1]):
            result = normaltest(latent_vecs[:,latentdim]).pvalue
            p_value_list.append(result)
        plt.plot([i for i in range(1,latent_vecs.shape[1]+1)], p_value_list)
        plt.axhline(0.05, c='r')
        plt.xlabel('Dim')
        plt.ylabel('P_value')
        plt.title('Normal Test for Latentvecs')
        plt.savefig(prefix + '/normal_test.png')

if __name__  == '__main__':
    prefix = 'output/vec2img/flowers_512dim_batchmode'
    method = 'normal test'
    main(method, prefix)