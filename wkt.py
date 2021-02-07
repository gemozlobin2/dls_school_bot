import torch
import torchfile
import numpy as np
import torchvision
from torch import nn
from PIL import Image


class ConvWhitening(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        content_features = input[0]

        c, h, w = content_features.shape

        content_features = content_features.view(c, h * w)
        content_features = content_features - torch.mean(content_features, 1, True).expand_as(content_features)
        content_covariation = (content_features @ content_features.T) / (h * w - 1)
        cU, cD, cV = torch.svd(content_covariation)
        cD = torch.diag(cD.pow(-0.5))
        content_whiteness = cU @ cD @ cV.T @ content_features

        style_features = input[1]
        style_features = style_features.view(c, h * w)
        style_mean = torch.mean(style_features, 1, True).expand_as(style_features)
        style_features = style_features - style_mean
        style_covariation = (style_features @ style_features.T) / (h * w - 1)
        sU, sD, sV = torch.svd(style_covariation)
        sD = torch.diag(sD.pow(0.5))

        target_features = sU @ sD @ sV.T @ content_whiteness + style_mean
        target = self.alpha * target_features.view(c, h, w) + (1 - self.alpha) * input[0]
        return target.view(1, c, h, w)


class WKTModel:
    def __init__(self, weights_path, style_alpha=0.5, max_size=1000):
        self.weights_path = weights_path
        self.device = "cpu"# torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.style_alpha = style_alpha
        self.max_size = max_size

    def convert(self, content_img: Image, style_img: Image):
        content = img_to_tensor(content_img, self.max_size)
        style = img_to_tensor(style_img, shape=content.shape[1:])

        # собираем обе картинки в батч для обработки в один проход, сначала контент, потом стилизация
        batch = torch.Tensor([content.tolist(), style.tolist()])
        batch = batch.to(self.device)

        with torch.no_grad():
            for n in range(5, 0, -1):
                model = nn.Sequential(
                    convert_lua_to_py(f"{self.weights_path}/vgg_normalised_conv{n}_1.t7"),
                    ConvWhitening(self.style_alpha),
                    convert_lua_to_py(f"{self.weights_path}/feature_invertor_conv{n}_1.t7")
                )
                model.to(self.device)
                model.eval()
                content = model(batch)

                # Из-за того что размер изображений не фиксирован, после обработки свертками он может быть
                # незначительно изменен. Поэтому ресайзим стилевое изображение под новый размер
                style = img_to_tensor(style_img, shape=content.shape[2:])
                batch = torch.Tensor([content.squeeze().tolist(), style.tolist()])
                batch = batch.to(self.device)

                del model
        return content


def get_module_from_lua(obj):
    type_name = obj.torch_typename()
    o = obj._obj
    if type_name == b'nn.SpatialConvolution':
        conv = nn.Conv2d(o.nInputPlane, o.nOutputPlane, kernel_size=(o.kH, o.kW), stride=(o.dH, o.dW), padding=(o.padH, o.padW))
        conv.weight = torch.nn.Parameter(torch.FloatTensor(o.weight))
        conv.bias = torch.nn.Parameter(torch.FloatTensor(o.bias))
        return conv
    elif type_name == b'nn.SpatialReflectionPadding':
        return nn.ReflectionPad2d((o.pad_l, o.pad_r, o.pad_t, o.pad_b))
    elif type_name == b'nn.ReLU':
        return nn.ReLU(inplace=True)
    elif type_name == b'nn.SpatialMaxPooling':
        return nn.MaxPool2d(kernel_size=(o.kH, o.kW), stride=(o.dH, o.dW), padding=(o.padH, o.padW), ceil_mode=o.ceil_mode)
    elif type_name == b'nn.SpatialUpSamplingNearest':
        return nn.UpsamplingNearest2d(scale_factor=o.scale_factor)


def convert_lua_to_py(file_name):
    lua = torchfile.load(file_name)
    modules = []
    for i in lua._obj.modules:
        modules.append(get_module_from_lua(i))
    return nn.Sequential(*modules)


def img_to_tensor(img: Image, max_size=256, shape=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if shape is None:
        if max(img.size) > max_size:
            width, height = img.size
            if width >= height:
                height = height * max_size // width
                width = max_size
            else:
                width = width * max_size // height
                height = max_size
            shape = (height, width)
        else:
            shape = (img.size[1], img.size[0])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(shape),
        torchvision.transforms.Normalize(mean, std)
    ])
    img = transform(img)
    img = img[:3, :, :]
    return img


def tensor_to_img(img_data):
    img_data = img_data.to("cpu").clone().detach().numpy().squeeze()
    img_data = img_data.transpose(1, 2, 0)
    img_data = img_data * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img_data = img_data.clip(0, 1)
    print(img_data.shape)
    return Image.fromarray((img_data * 255).astype(np.uint8))