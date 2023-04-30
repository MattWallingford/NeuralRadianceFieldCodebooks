"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn
import torchvision
import util
from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler
import torch.autograd as autograd
from collections import OrderedDict

class BinarizeIndictator(autograd.Function):
    @staticmethod
    def forward(ctx, indicator):
        # Get the subnetwork by sorting the scores and using the top k%
        out = (indicator >= .5).float()
        return out
    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class BinarizeIndictator_thresh(autograd.Function):
    @staticmethod
    def forward(ctx, indicator, thresh):
        # Get the subnetwork by sorting the scores and using the top k%
        out = (indicator >= .5).float()
        return out
    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None



class freeze_conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weight_shape = self.weight.shape
        self.register_parameter(name = 'residual', param = torch.nn.Parameter(torch.zeros(weight_shape)))
        self.register_parameter(name = 'indicator', param = torch.nn.Parameter(torch.ones([1])*1))
        self.weight.requires_grad = False
        self.weight.grad = None


    def forward(self, x):
        #I = BinarizeIndictator.apply(self.indicator)
        w = self.weight
        x = F.conv2d(x,w,self.bias,self.stride,self.padding,self.dilation,self.groups)
        x = BinarizeIndictator.apply(x)
        return x

def conv1x1_nf(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv5x5_nf(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return freeze_conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def decoder(in_planes, out_planes, stride = 1):
    model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(in_planes, 128, kernel_size=3, stride=stride, bias=False)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(128, 512, kernel_size=3, stride=stride, bias=False)),
          ('relu2', nn.ReLU()),
        ]))
    return model

def basic_decoder(in_planes, out_planes, stride = 1):
    model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(in_planes, 512, kernel_size=3, stride=stride, bias=False)),]))
    return model


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        args = None,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)
        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
        if args.nlatent:
            self.latent_size_sl = args.extra_bits
        else:
            self.latent_size_sl = self.latent_size
        self.num_layers = num_layers
        self.nlatent = args.nlatent
        self.decouple = args.decouple
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        print("creating encoder of dict size equal to: {}".format(args.dict_size))
        self.dict_size = args.dict_size
        if self.dict_size > 0:
            self.freeze_conv = conv1x1(512,self.dict_size) #hard coded for now
        self.args = args
        if args:
            if args.extra_bits != 0:
                self.softmax = nn.Softmax(dim=1)
                self.extra_bits_conv = conv1x1_nf(512,args.extra_bits)
                self.latent_dict = decoder(self.dict_size+args.extra_bits, 512)
                #torch.nn.Conv2d(self.dict_size+args.extra_bits, 512, kernel_size = 5)
            else:
                self.latent_dict = self.latent_dict = decoder(self.dict_size+args.extra_bits, 512)
                #torch.nn.Conv2d(self.dict_size, 512, kernel_size = 1)
        # self.latent (B, L, H, W)
    def index_sl(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.shape_latent.shape[0] > 1:
                uv = uv.expand(self.shape_latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0
            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            self.shape_latent = self.shape_latent.to(uv.device)
            samples = F.grid_sample(
                self.shape_latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0
            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)
    
    def get_latent_code(self):
        return self.latent_code

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)
            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
            #print(self.latent.shape)
            if self.dict_size > 0:
                self.latent_code = self.freeze_conv(self.latent) #MY CODE
            #print(self.latent_code.shape)
        if not(self.args.standard):
            if self.args.extra_bits:
                self.latent_code_cont = self.extra_bits_conv(self.latent)
                print(self.latent_code_cont.shape)
                self.latent_code_cont = self.softmax(self.latent_code_cont)
            if self.dict_size > 0:
                self.latent_code = torch.cat((self.latent_code_cont, self.latent_code), dim = 1)
            else:
                self.latent_code = self.latent_code_cont
            if self.args.ste:
                self.latent_code = BinarizeIndictator.apply(self.latent_code)
            if self.decouple:
                if self.nlatent:
                    self.shape_latent = self.latent_code
                else:
                    self.shape_latent = self.latent_dict(self.latent_code)#self.latent_code#
            else: 
                self.latent = self.latent_dict(self.latent_code)
        else:
            self.latent = self.latent
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        return self.latent

    def get_code(self, parameter_list):
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)
            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
            self.latent_code = self.freeze_conv(self.latent_code)
        return self.latent_code

    @classmethod
    def from_conf(cls, conf, args = None):
        return cls(
            args,
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)


    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )
