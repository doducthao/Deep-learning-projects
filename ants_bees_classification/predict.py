from lib import *
from utils import *
from config import *
from transform import ImageTransform

class_index = ['ants', 'bees']

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max_id(self, output):
        max_id = np.argmax(output.detach().numpy())
        predict_label = self.class_index[max_id]
        return predict_label

predictor = Predictor(class_index)

def predict(img):
    # prepare network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(4096,2)
    net.eval()

    # prepare model
    model = load_model(net)

    # prepare input img

    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase='test')
    img = img.unsqueeze_(0) # (chan, height, width) -> (1, chan, height, width)
    
    # predict
    output = model(img)
    label_name = predictor.predict_max_id(output)
    return label_name