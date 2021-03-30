import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from .ssim import MSSSIM, SSIM
import losses.lpips as lpips

class LossBuilder:
    def __init__(self, device):
        self.vgg_path = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
        self.vgg_used = False
        self.device = device

    # create a module to normalize input image so we can easily put it in a
    # nn.Sequential
    class VGGNormalization(nn.Module):
        def __init__(self, device):
            super().__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    def mse(self):
        #class MSE(nn.Module):
        #    def __init__(self):
        #        super(MSE, self).__init__()
        #        self.mseLoss = nn.MSELoss()
        #    def forward(self, gt, pred):
        #        #gt = LossBuilder._preprocess(gt)
        #        #pred = LossBuilder._preprocess(pred)
        #        return self.mseLoss(gt, pred)
        #return MSE()
        return nn.MSELoss()

    def inverse_mse(self, gt, pred):
        class InverseMSE(nn.Module):
            def __init__(self):
                super(InverseMSE, self).__init__()
                self.mseLoss = nn.MSELoss()
            def forward(self, gt, pred):
                gt = LossBuilder._preprocess(gt)
                pred = LossBuilder._preprocess(pred)
                pred = F.interpolate(pred, size=[gt.shape[1], gt.shape[2]], mode='bilinear')
                return self.mseLoss(gt, pred)
        return InverseMSE()
    
    def fft_mse(self):
        class FFT_MSE(nn.Module):
            def __init__(self):
                super(FFT_MSE, self).__init__()
                self.mseLoss = nn.MSELoss()
            def forward(self, gt, pred):
                gt = LossBuilder._preprocess(gt)
                pred = LossBuilder._preprocess(pred)
                gt = torch.rfft(gt, signal_ndim=3)
                pred = torch.rfft(pred, signal_ndim=3)
                return self.mseLoss(gt, pred)
        return FFT_MSE()

    def l1_loss(self):
        #class L1Loss(nn.Module):
        #    def __init__(self):
        #        super(L1Loss, self).__init__()
        #        self.l1Loss = nn.L1Loss()
        #    def forward(self, gt, pred):
        #        gt = LossBuilder._preprocess(gt)
        #        pred = LossBuilder._preprocess(pred)
        #        return self.l1Loss(gt, pred)
        #return L1Loss()
        return nn.L1Loss()

    class TemporalL2(nn.Module):
        def __init__(self):
            super().__init__()
            self.mseLoss = nn.MSELoss()
            self.threshold = 0.5
        def forward(self, pred_with_mask, prev_warped_with_mask):
            mask = torch.ge(pred_with_mask[:,3:4,:,:], self.threshold).float() * \
                    torch.ge(prev_warped_with_mask[:,3:4,:,:], self.threshold).float()
            mask.requires_grad = False
            return self.mseLoss(pred_with_mask[:,0:3,:,:] * mask, 
                                prev_warped_with_mask[:,0:3,:,:] * mask)
    def temporal_l2(self):
        return LossBuilder.TemporalL2()

    class TemporalL1(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1Loss = nn.L1Loss()
            self.threshold = 0.5
        def forward(self, pred_with_mask, prev_warped_with_mask):
            mask = torch.ge(pred_with_mask[:,3:4,:,:], self.threshold).float() * \
                    torch.ge(prev_warped_with_mask[:,3:4,:,:], self.threshold).float()
            mask.requires_grad = False
            return self.l1Loss(pred_with_mask[:,0:3,:,:] * mask, 
                                prev_warped_with_mask[:,0:3,:,:] * mask)
    def temporal_l1(self):
        return LossBuilder.TemporalL1()

    @staticmethod
    def _gram_matrix(features):
        dims = features.shape
        #features = torch.reshape(features, [-1, dims[-3], dims[-2] * dims[-1]])
        features = features.view(-1, dims[-3], dims[-2] * dims[-1])
        
        gram_matrix = torch.matmul(features, torch.transpose(features, 1, 2))
        normalized_gram_matrix = gram_matrix / (dims[-3] * dims[-2] * dims[-1])
        
        return normalized_gram_matrix #tf.matmul(features, features, transpose_a=True)

    @staticmethod
    def _normalize(features):
        dims = features.get_shape().as_list()
        return features / (dims[1] * dims[2] * dims[3])
    
    @staticmethod
    def _preprocess(images):
        #return (images / 255.0) * 2.0 - 1.0
        return images * 2.0 - 1.0 # images are already in [0,1]

    @staticmethod
    def _texture_loss(features, patch_size=16):
        '''
        the front part of features : gt features
        the latter part of features : pred features
        I will do calculating gt and pred features at once!
        '''

        #features = self._normalize(features)
        if patch_size is not None:
            batch_size, c, h, w = features.shape
            features = torch.unsqueeze(features, 1) # Bx1xCxHxW
            features = F.pad(features, (0, w % patch_size, 0, h % patch_size)) #Bx1xCxHpxWp
            features = torch.cat(torch.split(features, patch_size, 3), 1) # B x (H//patch_size) x C x patch_size x W
            features = torch.cat(torch.split(features, patch_size, 4), 1) # B x (H//patch_size)*(W//patchsize) x C x patch_size x patch_size
        patches_gt, patches_pred = torch.chunk(features, 2, dim=0)

        #features = tf.space_to_batch_nd(features, [patch_size, patch_size], [[0, 0], [0, 0]])
        #features = torch.reshape(features, [patch_size, patch_size, -1, h // patch_size, w // patch_size, c])
        #features = tf.transpose(features, [2, 3, 4, 0, 1, 5])
        #patches_gt, patches_pred = tf.split(features, 2, axis=0)
        #patches_gt = tf.reshape(patches_gt, [-1, patch_size, patch_size, c])
        #patches_pred = tf.reshape(patches_pred, [-1, patch_size, patch_size, c])
        
        gram_matrix_gt = LossBuilder._gram_matrix(patches_gt)
        gram_matrix_pred = LossBuilder._gram_matrix(patches_pred)
        
        #tl_features = torch.mean(torch.sum((gram_matrix_gt - gram_matrix_pred)**2, dim=1))
        tl_features = F.mse_loss(gram_matrix_gt, gram_matrix_pred)
        return tl_features
    
    class TextureLoss(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, x):
            self.loss = LossBuilder._texture_loss(x) * self.weight
            return x

    class PerceptualLoss(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, x):
            gt_pool, pred_pool = torch.chunk(x, 2, dim=0)
            #self.loss = torch.mean(torch.sum((gt_pool - pred_pool)**2, dim=1))
            self.loss = F.mse_loss(gt_pool, pred_pool) * self.weight
            return x
   
    def get_style_and_content_loss(self,
                                   content_layers,
                                   style_layers):

        cnn = models.vgg19(pretrained=True).features.eval()
        for p in cnn.parameters():
            p.requires_grad = False

        # normalization module
        normalization = LossBuilder.VGGNormalization(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers.keys():
                # add content loss:
                weight = content_layers[name]
                content_loss = LossBuilder.PerceptualLoss(weight)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers.keys():
                # add style loss:
                weight = style_layers[name]
                style_loss = LossBuilder.TextureLoss(weight)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], LossBuilder.PerceptualLoss) or isinstance(model[i], LossBuilder.TextureLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    class SSIMLoss(nn.Module):
        def __init__(self, channels):
            super().__init__();
            self._m = SSIM()
        def forward(self, gt, pred):
            return self._m(pred, gt)

    def ssim_loss(self, channels):
        import warnings
        warnings.warn("SSIM requested, note that it can't be used for training because it grows with a better result. Use DSSIM instead")
        return LossBuilder.SSIMLoss(channels)

    class DSSIMLoss(nn.Module):
        def __init__(self, channels):
            super().__init__();
            self._m = SSIM()
        def forward(self, gt, pred):
            return (1-self._m(pred, gt))/2

    def dssim_loss(self, channels):
        return LossBuilder.DSSIMLoss(channels)


    class LPIPSLoss(nn.Module):
        NETWORK = None

        def __init__(self, channels, min_value, max_value):
            super().__init__()
            assert channels==1 or channels>=3, 'illegal channel count, only 1 or >=3 supported'
            self._channels = channels
            self._min_value = min_value
            self._max_value = max_value

        def forward(self, gt, pred):
            # load network, now that we know if on GPU or not
            if LossBuilder.LPIPSLoss.NETWORK is None:
                LossBuilder.LPIPSLoss.NETWORK = lpips.PerceptualLoss(
                    model='net-lin', net='alex', 
                    use_gpu=gt.is_cuda, gpu_ids=[0])
                print("LPIPS network loaded")

            # scale to [-1, +1]
            gt = (2*gt - (self._min_value+self._max_value)) / (self._max_value-self._min_value)
            pred = (2*pred - (self._min_value+self._max_value)) / (self._max_value-self._min_value)
            # broadcast
            assert gt.shape[1]==self._channels, "expected %d channels, but got %d"%(self._channels, gt.shape[1])
            if self._channels==1:
                gt = torch.cat([gt]*3, dim=1)
                pred = torch.cat([pred]*3, dim=1)
            elif self._channels>3:
                # simply delete the additional channels
                gt = gt[:,:3,:,:]
                pred = pred[:,:3,:,:]

            return torch.mean(LossBuilder.LPIPSLoss.NETWORK(gt, pred))

    def lpips_loss(self, channels, min_value, max_value):
        return LossBuilder.LPIPSLoss(channels, min_value, max_value)

    class AdversarialLoss(nn.Module):
        TRUE_LABEL = 1
        FAKE_LABEL = 0
        def __init__(self):
            super().__init__()
        def forward(self, x): # x=prediction
            #generator loss
            adv_g_loss = F.binary_cross_entropy_with_logits(
                x, 
                torch.full(x.shape, LossBuilder.AdversarialLoss.TRUE_LABEL, device=x.device))
            return adv_g_loss

        def train_discr(self, gt_input, pred_input, discr, use_checkpoint=False):
            if use_checkpoint:
                gt_logits = torch.utils.checkpoint.checkpoint(discr, gt_input) #real
                pred_logits = torch.utils.checkpoint.checkpoint(discr, pred_input) #fake
            else:
                gt_logits = discr(gt_input) #real
                pred_logits = discr(pred_input) #fake
            #statistics
            mean_gt_logits = torch.mean(torch.sigmoid(gt_logits)).item()
            mean_pred_logits = torch.mean(torch.sigmoid(pred_logits)).item()
            #discriminator loss
            adv_gt_loss = F.binary_cross_entropy_with_logits(
                gt_logits, 
                torch.full(gt_logits.shape, LossBuilder.AdversarialLoss.TRUE_LABEL, device=gt_logits.device))
            adv_pred_loss = F.binary_cross_entropy_with_logits(
                pred_logits,
                torch.full(pred_logits.shape, LossBuilder.AdversarialLoss.FAKE_LABEL, device=pred_logits.device))
            adv_d_loss = adv_gt_loss + adv_pred_loss
            return adv_d_loss, mean_gt_logits, mean_pred_logits

    def _gan_loss(self):
        # gt_logits -> real, pred_logits -> fake
        # all values went through tf.nn.sigmoid

        return LossBuilder.AdversarialLoss()

    class WassersteinAdversarialLoss(nn.Module):
        def __init__(self, gradient_penalty, lambda_):
            super().__init__()
            self.use_gradient_penalty = gradient_penalty
            self.lambda_ = lambda_
        def forward(self, x): # x=prediction
            # x = F.sigmoid(x) # IMPORTANT, no sigmoid!
            #cost for training the generator
            adv_g_loss = -torch.mean(x)
            return adv_g_loss

        def train_discr(self, gt_input, pred_input, discr, use_checkpoint=False):
            if use_checkpoint:
                disc_gt = torch.utils.checkpoint.checkpoint(discr, gt_input) #real
                disc_pred = torch.utils.checkpoint.checkpoint(discr, pred_input) #fake
            else:
                disc_gt = discr(gt_input) #real
                disc_pred = discr(pred_input) #fake
            #statistics
            gt_score = torch.mean(disc_gt).item()
            pred_score = torch.mean(disc_pred).item()
            #discriminator loss
            disc_cost = torch.mean(disc_pred) - torch.mean(disc_gt)
            #gradient penalty
            if self.use_gradient_penalty:
                alpha = torch.rand((b//2, 1, 1, 1), dtype=x.dtype, device=x.device, requires_grad=True)
                interpolates = gt_input + alpha*(pred_input - gt_input)
                grad_outputs = torch.ones(b//2, 1, dtype=x.dtype, device=x.device, requires_grad=True)
                gradients = torch.autograd.grad(
                    discr(interpolates), 
                    interpolates, 
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    only_inputs=True)[0]
                slopes = torch.norm(gradients.view(b//2, -1))
                gradient_penalty = torch.mean((slopes-1.0)**2)
                disc_cost += self.lambda_ * gradient_penalty
            #done
            return disc_cost, gt_score, pred_score
    def _wgan_loss(self, gradient_penalty):
        LAMBDA = 10 # Gradient penalty lambda hyperparameter
        # gt_logits -> real, pred_logits -> fake
        return LossBuilder.WassersteinAdversarialLoss(gradient_penalty, LAMBDA)

    def build_discriminator(self, model, resolution, input_channels, additional_opt):
        '''
        build_discriminator is only used for training!
        '''
        module = None
        if model.lower()=='enhanceNetSmall'.lower():
            from .enhancenetsmall import EnhanceNetSmallDiscriminator
            module = EnhanceNetSmallDiscriminator(resolution, input_channels)
        elif model.lower()=='enhanceNetLarge'.lower():
            from .enhancenetlarge import EnhanceNetLargeDiscriminator
            module = EnhanceNetLargeDiscriminator(resolution, input_channels)
        elif model.lower()=='tecoGAN'.lower():
            from .tecogan import TecoGANDiscriminator
            module = TecoGANDiscriminator(resolution, input_channels)
        else:
            raise ValueError('Unsupported discriminator model: %s'%model)
        return module

    def gan_loss(self, model, resolution, input_channels, additional_opt):
        """
        Builds the adversarial loss. Model is the name of the discriminator model.
        Currently supported: EnhanceNetSmall, EnhanceNetLarge, TecoGAN
        """
        discriminator = self.build_discriminator(model, resolution, input_channels, additional_opt)
        adv_loss = self._gan_loss()
        return discriminator, adv_loss

    def wgan_loss(self, model, resolution, input_channels, additional_opt, gradient_penalty=False):
        """
        Builds the adversarial loss with the Wasserstein GAN loss. 
        Model is the name of the discriminator model.
        Currently supported: EnhanceNetSmall, EnhanceNetLarge, TecoGAN
        """
        discriminator = self.build_discriminator(model, resolution, input_channels, additional_opt)
        adv_loss = self._wgan_loss(gradient_penalty)
        return discriminator, adv_loss

    class DownsampleLoss(nn.Module):
        def __init__(self, loss, downsample_factor, downsample_mode, gt_low_res, reduction):
            """
            Computes the L2-Loss on a downsampled image,
            i.e. it checks if the superresolution image is consistent with the
            low resolution input.

            loss:
             The loss function that is applied, can be 'l1' or 'l2'
            downsample_factor: 
             The factor for downsampling, e.g. 2 or 4. 
             Should ideally be equal to the up factor used for super-resolution
            downsample_mode:
             The interpolation mode: nearest, bilinear (default), bicubic
            low_res_gt:
             If set to true, the ground truth (first parameter of the loss function)
             is already expected to be in the low-resolution version
            reduction:
             The reduction applied in the loss, can be 'none', 'mean' (default) or 'sum'.
             See the parameter 'reduction' of torch.nn.MSELoss for further details
            """
            super().__init__()
            self.sample = nn.Upsample(scale_factor = 1.0/downsample_factor, mode=downsample_mode)
            if loss=='l1':
                self.loss = nn.L1Loss(reduction=reduction)
            elif loss=='l2':
                self.loss = nn.MSELoss(reduction=reduction)
            else:
                raise ValueError("unknown loss: "+str(loss)+", expected 'l1' or 'l2'")
            self.gt_low_res = gt_low_res
        def forward(self, gt, pred):
            if self.gt_low_res:
                return self.loss(gt, self.sample(pred))
            else:
                return self.loss(self.sample(gt), self.sample(pred))

    def downsample_loss(self, 
                        loss,
                        downsample_factor, 
                        downsample_mode='bilinear', 
                        low_res_gt=False,
                        reduction='mean'):
        """
        Computes the L2-Loss on a downsampled image,
        i.e. it checks if the superresolution image is consistent with the
        low resolution input.

        loss:
         The loss function that is applied, can be 'l1' or 'l2'
        downsample_factor: 
         The factor for downsampling, e.g. 2 or 4. 
         Should ideally be equal to the up factor used for super-resolution
        downsample_mode:
         The interpolation mode: nearest, bilinear (default), bicubic
        low_res_gt:
         If set to true, the ground truth (first parameter of the loss function)
         is already expected to be in the low-resolution version
        reduction:
         The reduction applied in the loss, can be 'none', 'mean' (default) or 'sum'.
         See the parameter 'reduction' of torch.nn.MSELoss for further details
        """
        loss = DownsampleLoss(loss, downsample_factor, downsample_mode, low_res_gt, reduction)
        loss.to(self.device)
        return loss
