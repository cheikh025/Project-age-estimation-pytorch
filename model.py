import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import torchvision.models as models


def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = models.resnext50_32x4d(pretrained=pretrained)
    input_size = model.fc.in_features
    model.fc = nn.Linear(in_features=input_size, out_features=num_classes)
    return model


def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
