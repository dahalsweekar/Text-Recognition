import torch
from PIL import Image


class Demo:
    def __init__(self, model, img_path, transform=None):
        self.model = model
        self.img_path = img_path
        self.transform = transform
        self.model = model

    def predict(self):
        image = Image.open(self.img_path)
        image = Image.merge('RGB', [image, image, image])
        image = self.transform(image).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            output = self.model(image)

        _, predict = torch.max(output, 1)
        return predict.item()
