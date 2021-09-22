import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.layer3 = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.layer4 = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)

        self.layer5 = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        batch_size, _, _, _ = images.size()
        out = self.layer1(images)
        out = self.layer2(out)

        out = out.permute(0, 3, 1, 2)
        out = out.view(batch_size, out.size(1), -1)

        out = self.layer3(out)
        out, _ = self.layer4(out)
        out = self.layer5(out)

        out = out.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(out, 2)
            input_lengths = torch.full(
                size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(batch_size,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return out, loss

        return out, None


if __name__ == "__main__":
    model = CaptchaModel(19)
    img = torch.rand((1, 3, 50, 200))
    out, _ = model(img, torch.rand((1, 5)))
    print(out.size())
