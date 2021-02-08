import os
import numpy as np
import torchvision
import torch
from PIL import Image
from wkt import WKTModel
from CartoonGAN import Transformer


def style_transfer_handler(config, content_img: Image, style_img: Image):
    max_size = int(config["wkt"]["max_size"])
    max_layers = int(config["wkt"]["max_layers"])

    step_size = 2 ** max_layers
    if max_size < step_size:
        max_size = step_size
    elif (max_size % step_size) != 0:
        max_size = max_size // step_size * step_size

    model = WKTModel(
        config["wkt"]["weights_path"],
        float(config["wkt"]["style_alpha"]),
        max_layers
    )
    content = img_to_tensor(content_img, max_size)
    style = img_to_tensor(style_img, shape=content.shape[1:])
    img_data = model.convert(content, style)
    del model
    return tensor_to_img(img_data)


def cycle_gan_handler(config, img: Image):
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join(config["cartoongan"]["weights_path"], config["cartoongan"]["style_name"] + '_net_G_float.pth')))
    model.eval()

    img = img_to_tensor(img, int(config["cartoongan"]["max_size"]), mean=0.5, std=0.5)[[2, 1, 0], :, :].unsqueeze(0)
    img_data = model(img).squeeze()[[2, 1, 0], :, :]
    del model

    return tensor_to_img(img_data, mean=0.5, std=0.5)


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


def tensor_to_img(img_data, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img_data = img_data.to("cpu").clone().detach().numpy().squeeze()
    img_data = img_data.transpose(1, 2, 0)
    img_data = img_data * np.array(std) + np.array(mean)
    img_data = img_data.clip(0, 1)
    return Image.fromarray((img_data * 255).astype(np.uint8))
