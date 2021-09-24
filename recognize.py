import torch
from model import CaptchaModel
from dataset import CaptchaDataset
from sklearn import preprocessing

import argparse
import glob
import os
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


def recognize(opt):
    image_files = glob.glob(os.path.join(opt.data, "*.png"))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1

    img_path = './captcha_images_v2/n8pfe.png'
    image = Image.open(img_path).convert("RGB")
    image = image.resize(
        (opt.width, opt.height), resample=Image.BILINEAR
    )
    image = np.array(image)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((opt.height, opt.width)),
        transforms.Normalize(mean, std, inplace=True)
    ])

    image = transform(image)
    print(image.shape)
    image = image.unsqueeze(0)
    print(image.shape)

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.load_state_dict(torch.load('final.pth'))

    with torch.no_grad():
        model.eval()
        pred, _ = model(image)
        current_preds = decode_predictions(pred, lbl_enc)
        print(current_preds)
        print(current_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Captcha recognition using RNN')
    parser.add_argument('--data', type=str, default='./captcha_images_v2', help='path to data folder')
    parser.add_argument('--batch_size', type=int, default=2, help='set the batch size')
    parser.add_argument('--height', type=int, default=50, help='height of the input image')
    parser.add_argument('--width', type=int, default=200, help='width of the input image')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers in dataloader')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    opt = parser.parse_args()

    recognize(opt)
