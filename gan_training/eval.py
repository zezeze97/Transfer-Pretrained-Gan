import torch
from gan_training.metrics import inception_score
from gan_training.metrics.fid_score import calculate_fid_given_paths
import math
import numpy as np
import os
import cv2
from tqdm import tqdm
class Evaluator(object):
    def __init__(self, generator, zdist_type, zdist, ydist, batch_size=64,
                 inception_nsamples=60000, device=None):
        self.generator = generator
        self.zdist_type = zdist_type
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device

    def compute_inception_score(self, use_gmm=False):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            if self.zdist_type == 'gmm2gauss':
                ztest = self.zdist.sample((self.batch_size,),use_gmm)
            else:
                ztest = self.zdist.sample((self.batch_size,))
            ytest = self.ydist.sample((self.batch_size,))

            samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def compute_fid_score(self, generated_img_path, gt_path, img_size):
        paths = [generated_img_path, gt_path]
        fid = calculate_fid_given_paths(paths, batch_size=self.batch_size, img_size=img_size, device=self.device, dims=2048, num_workers=1)
        return fid

    def save_samples(self, sample_num, save_dir, use_gmm=False):
        self.generator.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_of_img = 0
        flag = True
        while flag:
            if self.zdist_type == 'gmm2gauss':
                z = self.zdist.sample((self.batch_size,), use_gmm)
            else:
                z = self.zdist.sample((self.batch_size,))
            y = self.ydist.sample((self.batch_size,))
            with torch.no_grad():
                x = self.generator(z, y)
            imgs = x.detach().cpu().numpy()
            imgs = imgs * 0.5 + 0.5
            imgs = np.uint8(imgs * 255)
            for img in imgs:
                img = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2BGR)  # for saving using cv2.imwrite
                num_of_img += 1
                cv2.imwrite(os.path.join(save_dir, '%08d.png' % num_of_img), img)
                if num_of_img % 1000 == 0:
                    print('Generated ', num_of_img, ' images')
                if num_of_img >= sample_num:
                    flag = False
                    break
        

    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            x = self.generator(z, y)
        return x

class Learnable_GMM_Evaluator(object):
    def __init__(self, generator, gmm_layer, ydist, batch_size=64,
                 inception_nsamples=60000, device=None):
        self.generator = generator
        self.gmm_layer = gmm_layer
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device

    def compute_inception_score(self):
        self.gmm_layer.eval()
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            with torch.no_grad():
                ztest = self.gmm_layer(self.batch_size)
                ytest = self.ydist.sample((self.batch_size,))
                samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def compute_fid_score(self, generated_img_path, gt_path, img_size):
        paths = [generated_img_path, gt_path]
        fid = calculate_fid_given_paths(paths, batch_size=self.batch_size, img_size=img_size, device=self.device, dims=2048, num_workers=1)
        return fid

    def save_samples(self, sample_num, save_dir):
        self.gmm_layer.eval()
        self.generator.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_of_img = 0
        flag = True
        while flag:
            with torch.no_grad():
                z = self.gmm_layer(self.batch_size)
            y = self.ydist.sample((self.batch_size,))
            with torch.no_grad():
                x = self.generator(z, y)
            imgs = x.detach().cpu().numpy()
            imgs = imgs * 0.5 + 0.5
            imgs = np.uint8(imgs * 255)
            for img in imgs:
                img = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2BGR)  # for saving using cv2.imwrite
                num_of_img += 1
                cv2.imwrite(os.path.join(save_dir, '%08d.png' % num_of_img), img)
                if num_of_img % 1000 == 0:
                    print('Generated ', num_of_img, ' images')
                if num_of_img >= sample_num:
                    flag = False
                    break
        

    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            x = self.generator(z, y)
        return x
