import torch
from gan_training.metrics import inception_score
from gan_training.metrics.fid_score import calculate_fid_given_paths
import math
import numpy as np
import os
import cv2
from tqdm import tqdm
import glob

class Evaluator(object):
    def __init__(self, generator, zdist, ydist, batch_size=64,
                 inception_nsamples=60000, device=None):
        self.generator = generator
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
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

    def save_samples(self, sample_num, save_dir):
        self.generator.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_of_img = 0
        flag = True
        while flag:
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


class Evaluator_autoshift(object):
    def __init__(self, generator, autoshift, eval_loader, ydist, batch_size=64,
                 inception_nsamples=60000, device=None):
        self.autoshift = autoshift
        self.generator = generator
        # self.zdist = zdist
        self.eval_loader = eval_loader
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device

    def compute_inception_score(self):
        #! TO DO: modify as fid 
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
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

    def save_samples(self, sample_num, save_dir):
        self.autoshift.eval()
        self.generator.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_of_img = 0
        flag = True
        while flag:
            # z = self.zdist.sample((self.batch_size,))
            x_real, _ = next(iter(self.eval_loader))
            x_real = x_real.to(self.device)
            y = self.ydist.sample((self.batch_size,))
            with torch.no_grad():
                z = self.autoshift(x_real)[0]
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
        

    def create_samples(self, y=None, batch_size=64):
        self.autoshift.eval()
        self.generator.eval()
        batch_size = batch_size
        # batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        x_real, _ = next(iter(self.eval_loader))
        x_real = x_real.to(self.device)
        with torch.no_grad():
            z = self.autoshift(x_real)[0]
            x = self.generator(z, y)
        return x


class Evaluator_autoshift_save(object):
    def __init__(self, generator, autoshift, eval_loader, ydist, batch_size=64,
                 inception_nsamples=60000, device=None, config=None):
        self.autoshift = autoshift
        self.generator = generator
        # self.zdist = zdist
        self.eval_loader = eval_loader
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.config = config

    def compute_inception_score(self):
        #! TO DO: modify as fid 
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
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

    def save_samples(self, sample_num, save_dir, feed_whole_data=True):
        self.autoshift.eval()
        self.generator.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_of_img = 0
        
        if self.config is not None:
            latent_dim = self.config['z_dist']['dim']
        else:
            laten_dim = 512
      
        # Go through the whole dataset to calculate the average mean and std.
        print('Calculate mean and std of VAE encoder...')
        if feed_whole_data:
            # avg_mu, avg_var = torch.zeros(latent_dim), torch.zeros(latent_dim)
            # avg_mu, avg_var = avg_mu.to(self.device), avg_var.to(self.device)
            avg_mu, avg_std = torch.zeros(latent_dim), torch.zeros(latent_dim)
            avg_mu, avg_std = avg_mu.to(self.device), avg_var.to(self.device)
            for i, (x_real, _) in enumerate(self.eval_loader):
                x_real = x_real.to(self.device)
                with torch.no_grad():
                    # mu, var = self.autoshift.encode(x_real)
                    mu, var = self.autoshift.module.encode(x_real)  # use .module since DataParallel 
                    # mu, var = mu.mean(dim=0), var.mean(dim=0)
                    mu = mu.mean(dim=0)
                    std = torch.exp(0.5 * var)
                    std = std.mean(dim=0)
                avg_mu = (avg_mu * i + mu) / (i+1)    
                # avg_var = (avg_var * i + var) / (i+1)
                avg_std = (avg_std * i + std) / (i+1)
            # avg_std = torch.exp(0.5 * avg_var)
        else:
            avg_mu, avg_std = torch.zeros(latent_dim), torch.zeros(latent_dim)
            avg_mu, avg_std = avg_mu.to(self.device), avg_std.to(self.device)
            mean_paths = glob.glob(save_dir+'/batch*mean.npy')
            std_paths = glob.glob(save_dir+'/batch*std.npy')
            mean_paths.sort()
            std_paths.sort()
            for i, (mean_path, std_path) in enumerate(zip(mean_paths, std_paths)):
                mu = torch.from_numpy(np.load(mean_path)).to(self.device)
                std = torch.from_numpy(np.load(std_path)).to(self.device)
                avg_mu = (avg_mu * 0 + mu) / (i+1) 
                avg_std = (avg_std * 0 + std) / (i+1) 
        np.save(os.path.join(self.config['training']['out_dir'], 'autoshift_mean.npy'), avg_mu.cpu().numpy())
        np.save(os.path.join(self.config['training']['out_dir'], 'autoshift_std.npy'), avg_std.cpu().numpy())

        flag = True
        while flag:
            # eps = torch.randn_like(avg_std)
            eps = torch.randn((self.batch_size, latent_dim)).to(self.device)
            z = eps * avg_std + avg_mu
            # z = self.zdist.sample((self.batch_size,))
            # x_real, _ = next(iter(self.eval_loader))
            # x_real = x_real.to(self.device)
            y = self.ydist.sample((self.batch_size,))
            with torch.no_grad():
                # z = self.autoshift(x_real)[0]
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
        

    def create_samples(self, y=None, batch_size=64):
        self.autoshift.eval()
        self.generator.eval()
        batch_size = batch_size
        # batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        mean_path = os.path.join(self.config['training']['out_dir'], 'autoshift_mean.npy')
        std_path = os.path.join(self.config['training']['out_dir'], 'autoshift_std.npy')
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mu = torch.from_numpy(np.load(mean_path)).to(self.device)
            std = torch.from_numpy(np.load(std_path)).to(self.device)
            # eps = torch.randn_like(std)
            eps = torch.randn((batch_size, mu.size(0))).to(self.device)
            z = eps * std + mu
        else:
            z = torch.randn((batch_size, self.config['z_dist']['dim'])).to(self.device)
        
        # x_real, _ = next(iter(self.eval_loader))
        # x_real = x_real.to(self.device)
        with torch.no_grad():
            # z = self.autoshift(x_real)[0]
            x = self.generator(z, y)
        return x



