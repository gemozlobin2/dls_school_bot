import os
import numpy as np
import torch
from PIL import Image
from wkt import WKTModel, img_to_tensor
from CartoonGAN import Transformer


def style_transfer_handler(config, content_img: Image, style_img: Image):
    model = WKTModel(config["wkt"]["weights_path"], float(config["wkt"]["style_alpha"]), int(config["wkt"]["max_size"]))
    img_data = model.convert(content_img, style_img)
    del model

    img_data = img_data.to("cpu").clone().detach().numpy().squeeze()
    img_data = img_data.transpose(1, 2, 0)
    img_data = img_data * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img_data = img_data.clip(0, 1)

    return Image.fromarray((img_data * 255).astype(np.uint8))


def cycle_gan_handler(config, img: Image):
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join(config["cartoongan"]["weights_path"], config["cartoongan"]["style_name"] + '_net_G_float.pth')))
    model.eval()

    img = img_to_tensor(img, int(config["cartoongan"]["max_size"]), mean=0.5, std=0.5)[[2, 1, 0], :, :].unsqueeze(0)
    img_data = model(img).squeeze()[[2, 1, 0], :, :]
    del model

    img_data = img_data.to("cpu").clone().detach().numpy().squeeze()
    img_data = img_data.transpose(1, 2, 0)
    img_data = img_data * 0.5 + 0.5
    img_data = img_data.clip(0, 1)

    return Image.fromarray((img_data * 255).astype(np.uint8))
