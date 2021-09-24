import os
import glob
import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn

from sklearn import preprocessing
from sklearn import model_selection

from utils.dataset import CaptchaDataset
from nets.nn import CaptchaModel


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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


def run_training(opt):
    image_files = glob.glob(os.path.join(opt.data, "*.png"))
    targets_orig = [x.split("\\")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1

    X_train, X_test, y_train, y_test, _, y_test_orig = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42)

    train_dataset = CaptchaDataset(image_paths=X_train, targets=y_train, resize=(opt.height, opt.width))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                               shuffle=True)
    test_dataset = CaptchaDataset(image_paths=X_test, targets=y_test, resize=(opt.height, opt.width))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size // 2, num_workers=opt.num_workers,
                                              shuffle=False)

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.apply(weights_init)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.cuda:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)
    for epoch in range(opt.num_epochs):
        model.train()
        total_loss = 0
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, targets) in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            _, loss = model(images, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().item()

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, opt.num_epochs), total_loss / (i + 1), mem)
            progress_bar.set_description(s)

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
            progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))

            for i, (images, targets) in progress_bar:
                images = images.to(device)
                targets = targets.to(device)
                batch_preds, loss = model(images, targets)
                valid_loss += loss.detach().cpu().item()

                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, opt.num_epochs), total_loss / (i + 1), mem)
                progress_bar.set_description(s)

        scheduler.step(valid_loss)

    torch.save(model.state_dict(), 'final.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Captcha recognition using RNN')
    parser.add_argument('--data', type=str, default='./captcha_images_v2', help='path to data folder')
    parser.add_argument('--batch_size', type=int, default=2, help='set the batch size')
    parser.add_argument('--height', type=int, default=50, help='height of the input image')
    parser.add_argument('--width', type=int, default=200, help='width of the input image')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers in dataloader')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    opt = parser.parse_args()

    run_training(opt)






