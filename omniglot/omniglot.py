import os

from PIL import Image
import torchvision

class OmniglotImage:
    def __init__(self, data_type, dir_path='./data/omniglot'):
        self.data_type = data_type
        self.dir_path = dir_path

    def get_info(self, language, char, drawer):
        self.language = language
        self.char = char
        if drawer < 10:
            self.drawer = '0' + str(drawer)
        else:
            self.drawer = str(drawer)
    
    def get_image(self, transform=None):
        dirname = 'images_' + self.data_type
        image_path = os.path.join(self.dir_path, dirname, self.language, self.char)
        image_list = os.listdir(image_path)
        filename = ''

        for image_file in image_list :
            if image_file[len(image_file)-6:len(image_file)-4] == self.drawer :
                filename = image_file
                break
        
        image = Image.open(os.path.join(image_path, filename))

        if transform :
            return transform(image)
        else :
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor()
                ])
            return transform(image)


class OmniglotInfo:
    def __init__(self, dir_path='./data/omniglot'):
        self.image_size = 105
        self.background = os.listdir(os.path.join(dir_path, 'images_background'))
        self.evaluation = os.listdir(os.path.join(dir_path, 'images_evaluation'))

        char_gnd = {language : os.listdir(os.path.join(dir_path, 'images_background', language)) for language in self.background}
        char_eval = {language : os.listdir(os.path.join(dir_path, 'images_evaluation', language)) for language in self.evaluation}

        self.char = char_eval.copy()
        self.char.update(char_gnd)