import os
import numpy as np
from sklearn import mixture
from scipy.stats import normaltest
import matplotlib.pyplot as plt

def main(method, prefix, best_n_components):
    
    # latentvec_dir = 'outputs/generate_results/im2latent_save_dir_v2/latentvecs/'
    latentvec_dir = prefix + '/latentvecs/latentvecs.npy'
    latent_vecs = np.load(latentvec_dir)
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
        # best_n_components = 5
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
    
    if method == 'hist':
        plt.hist(latent_vecs[:,15],bins=50)
        plt.title("hist of latentvecs")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(prefix + '/hist.png')


if __name__  == '__main__':
    import argparse
    # Arguments
    parser = argparse.ArgumentParser(
        description='Modeling the latent vectors.'
    )
    parser.add_argument('--prefix', type=str, help='Path to vec2img root dir.')
    parser.add_argument('--components_num', type=int, help='Number of components in GMM.')
    

    args = parser.parse_args()
    prefix = args.prefix
    best_n_components = args.components_num
    method = 'gauss mixture'
    main(method, prefix, best_n_components)