import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

class LatentNet(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(LatentNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 4 * latent_dim, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(4 * latent_dim, 3 * latent_dim, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(3 * latent_dim, 2 * latent_dim, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_sigma = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = self.gap(x)
        x = torch.flatten(x, 1)  

        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x))  

        prob_distribution = Independent(Normal(loc=mu, scale=torch.exp(sigma)), 1)

        return prob_distribution, mu, sigma

class PriorNet(LatentNet):
    def __init__(self, latent_dim):
        super(PriorNet, self).__init__(input_channels=6, latent_dim=latent_dim) 

class PosteriorNet(LatentNet):
    def __init__(self, latent_dim):
        super(PosteriorNet, self).__init__(input_channels=7, latent_dim=latent_dim)  

# SaliencyNet
class DenseASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        return torch.cat([x1, x2, x3], dim=1)  

class SaliencyNet(nn.Module):
    def __init__(self, in_channels=4, output_channels=1): 
        super(SaliencyNet, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        vgg_features = list(vgg16.features.children())
        
        self.stage1 = nn.Sequential(*vgg_features[:4])  
        self.stage2 = nn.Sequential(*vgg_features[4:9])  
        self.stage3 = nn.Sequential(*vgg_features[9:16]) 
        self.stage4 = nn.Sequential(*vgg_features[16:23])  
        self.stage5 = nn.Sequential(*vgg_features[23:])  
        
        self.daspp1 = DenseASPP(in_channels=64, out_channels=64)
        self.daspp2 = DenseASPP(in_channels=128, out_channels=128)
        self.daspp3 = DenseASPP(in_channels=256, out_channels=256)
        self.daspp4 = DenseASPP(in_channels=512, out_channels=512)
        self.daspp5 = DenseASPP(in_channels=512, out_channels=512)
        
        self.conv_final = nn.Conv2d(64*3 + 128*3 + 256*3 + 512*3 + 512*3, output_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        
        d1 = self.daspp1(x1)
        d2 = self.daspp2(x2)
        d3 = self.daspp3(x3)
        d4 = self.daspp4(x4)
        d5 = self.daspp5(x5)
        
        concat_features = torch.cat([d1, d2, d3, d4, d5], dim=1)
        
        saliency_map = self.conv_final(concat_features)
        
        return saliency_map

# PredictionNet
class PredictionNet(nn.Module):
    def __init__(self, K, M):
        super(PredictionNet, self).__init__()
        
        self.r = nn.Parameter(torch.randn(K + M))  
        
        self.conv1 = nn.Conv2d(K + M, K, kernel_size=1)
        self.conv2 = nn.Conv2d(K, K // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(K // 2, 1, kernel_size=1)
    
    def forward(self, Sd, Ss):

        Ssd = torch.cat((Sd, Ss), dim=1)  
        
        _, r_idx = torch.sort(self.r, descending=True)
        Ssd = Ssd[:, r_idx, :, :]  
        
        x = F.relu(self.conv1(Ssd))
        x = F.relu(self.conv2(x))
        P = self.conv3(x)  
        
        return P

class DepthCorrectionNet(nn.Module):
    def __init__(self, in_channels): 
        super(DepthCorrectionNet, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        vgg_features = list(vgg16.features.children())
        
        self.stage1 = nn.Sequential(*vgg_features[:4])  
        self.stage2 = nn.Sequential(*vgg_features[4:9])  
        self.stage3 = nn.Sequential(*vgg_features[9:16])  
        self.stage4 = nn.Sequential(*vgg_features[16:23])  
        self.stage5 = nn.Sequential(*vgg_features[23:]) 
        
        self.daspp1 = DenseASPP(in_channels=64, out_channels=64)
        self.daspp2 = DenseASPP(in_channels=128, out_channels=128)
        self.daspp3 = DenseASPP(in_channels=256, out_channels=256)
        self.daspp4 = DenseASPP(in_channels=512, out_channels=512)
        self.daspp5 = DenseASPP(in_channels=512, out_channels=512)
        
        self.deconv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deconv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x):

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        
        d1 = self.daspp1(x1)
        d2 = self.daspp2(x2)
        d3 = self.daspp3(x3)
        d4 = self.daspp4(x4)
        d5 = self.daspp5(x5)
        
        concat_features = torch.cat([d1, d2, d3, d4, d5], dim=1)
        
        x = F.relu(self.deconv1(concat_features))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        
        x = self.upsample(x)
        
        return x

def boundary_iou_loss(D_prime, I_g):
    grad_D_prime = torch.abs(F.conv2d(D_prime, weight=torch.tensor([[[-1, 1]]], dtype=torch.float32, device=D_prime.device), padding=1))
    grad_I_g = torch.abs(F.conv2d(I_g, weight=torch.tensor([[[-1, 1]]], dtype=torch.float32, device=I_g.device), padding=1))
    
    intersection = torch.sum(torch.min(grad_D_prime, grad_I_g), dim=[1, 2])
    
    magnitude_D_prime = torch.sum(grad_D_prime, dim=[1, 2])
    magnitude_I_g = torch.sum(grad_I_g, dim=[1, 2])
    
    return 1 - (2 * intersection) / (magnitude_D_prime + magnitude_I_g + 1e-6)

def depth_correction_loss(D_prime, D_raw, I_g):
    Lsl = F.smooth_l1_loss(D_prime, D_raw, reduction='mean')
    
    Lioub = boundary_iou_loss(D_prime, I_g)
    
    return Lsl + Lioub

class SaliencyConsensusModule(nn.Module):
    def __init__(self, C):
        super(SaliencyConsensusModule, self).__init__()
        self.C = C  
    
    def adaptive_threshold(self, P, tau=0.5):
        return (P > tau).float()
    
    def forward(self, predictions):

        batch_size, num_preds, height, width = predictions.size()
        
        binary_predictions = []
        for c in range(num_preds):
            binary_predictions.append(self.adaptive_threshold(predictions[:, c, :, :]))
        binary_predictions = torch.stack(binary_predictions, dim=1) 

        majority_vote = binary_predictions.sum(dim=1) 
        
        Pmjv = (majority_vote >= (self.C / 2)).float() 

        indicator = (binary_predictions == Pmjv.unsqueeze(1))
        
        weighted_predictions = binary_predictions * indicator.float()
        final_saliency_map = weighted_predictions.sum(dim=1) / self.C  

        return final_saliency_map  

class UCNet(nn.Module):
    def __init__(self, M, latent_dim):
        super(UCNet, self).__init__()
        self.prior_net = PriorNet(latent_dim)
        self.post_net = PosteriorNet(latent_dim)
        self.depth_correction_net = DepthCorrectionNet(6)
        self.saliency_net = SaliencyNet(6, M)
        self.prediction_net = PredictionNet(latent_dim, M)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div
    
    def concatenate_features(self, *features):
        return torch.cat(features, dim=1)
    
    def forward(self, x, depth, y=None, is_training=True):
        if is_training:
            corrected_depth = self.depth_correction_net(depth)
            Sd = self.saliency_net(self.concatenate_features(x, corrected_depth))

            concatenated_features_posterior = self.concatenate_features(x, depth, y)
            self.post_dist, mu_post, sigma_post = self.self.post_net(concatenated_features_posterior) 
            concatenated_features_prior = self.concatenate_features(x, depth)
            self.prior_dist, mu_prior, sigma_prior = self.self.prior_net(concatenated_features_prior) 
            kl_loss = self.kl_divergence(self.post_dist, self.prior_dist)
            z_post = self.reparametrize(mu_post, sigma_post)
            z_prior = self.reparametrize(mu_prior, sigma_prior)
            prediction = self.prediction_net(self.concatenate_features(Sd, z_post))

            return prediction, kl_loss


