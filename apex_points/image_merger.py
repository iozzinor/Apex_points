import math

from PIL import Image, ImageDraw

class ImageMerger:
    def __init__(self, **kwargs):
        self.images = []
        self.names = []
        self.config = {
            'image_mode': 'RGB',
            'caption_color': (255, 255, 255),
            'caption_offset': (0.1, 0.95)
        }
        for key, value in kwargs.items():
            self.config[key] = value

    def add_image(self, image, name):
        """
        Append an image to display.

        Parameters
        ----------
        image: PIL.Image
            The image to display
        name: str
            The caption
        """
        self.images.append(image)
        self.names.append(name)

    def generate_image(self):
        if len(self.images) < 1:
            raise ValueError('no images have been added')
        width = 0
        height = 0
        for image in self.images:
            if image.size[0] > width:
                width = image.size[0]
            if image.size[1] > height:
                height = image.size[1]

        width_size = int(math.ceil(math.sqrt(len(self.images))))
        height_size = width_size - 1 if len(self.images) < width_size * (width_size - 1) + 1 else width_size
        result_image = Image.new(self.config['image_mode'], (width_size * width, height_size * height))

        caption_offset = (int(self.config['caption_offset'][0] * width), int(self.config['caption_offset'][1] * height))
    
        for i, image in enumerate(self.images):
            name = self.names[i] if i < len(self.names) else None
            x = int((i % width_size) * width)
            y = int((i // width_size) * height)
            image = image.convert(self.config['image_mode']).resize((width, height))
            if name is not None:
                ImageDraw.Draw(image).text(caption_offset, name, self.config['caption_color'])
            result_image.paste(image, (x, y))
        return result_image
