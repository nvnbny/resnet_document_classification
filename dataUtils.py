from torch.utils.data import Dataset, DataLoader
import os, pdf2image, numpy as np, PIL.Image as Image
from PIL import ImageOps
from torch import from_numpy, FloatTensor as FT, LongTensor as LT
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import random, math, collections
from shutil import copyfile, rmtree
from random import shuffle

np.random.seed()
HEIGHT = 1100
WIDTH = 850

## UTILITY FUNCTIONS ##############################################################################################################################

def imageFromTensor(tensor): 
    """Convert tensor to PILimage with one channel """
    return Image.fromarray(np.asarray((tensor * 127.5)+127.5).astype('uint8'))


def splitTrainVal(inPath):
    """
    This function hhelps split data insto training and val in 70:30 ratio
    """
    folders = ['train/', 'val/', 'train/change/', 'train/no_change/', 
               'val/change/', 'val/no_change/']
    if os.path.exists(inPath + 'train/'): rmtree(inPath + 'train/')
    if os.path.exists(inPath + 'val/'): rmtree(inPath + 'val/')
        
    for folder in folders: os.mkdir(os.path.join(inPath,folder))
    
    print('Spliting data into training and cross-validation in a 70:30 ratio...')
    for c in ['/change/', '/no_change/']:
        fnames = os.listdir(inPath + c)
        shuffle(fnames)
        nTrain = int(len(fnames) * 0.7)
        
        for name in fnames[:nTrain]: copyfile(inPath + c + name, inPath + 'train/' + c + name)
        for name in fnames[nTrain:]: copyfile(inPath + c + name, inPath + 'val/' + c + name)
            

## DATALOADER ENITITES ######################################################################################################################
class PDFDataset(Dataset):
    """
    This class load and preprocess and augmenting data for training for one particular 
    """
    def __init__(self, dataPath, dataTransform):
        changePaths = [dataPath + 'change/' + p for p in os.listdir(dataPath + 'change/')]
        noChangePaths = [dataPath + 'no_change/' + p for p in os.listdir(dataPath + 'no_change/')]
        changeLabels = [1 for _ in range(len(changePaths))]
        noChangeLabels = [0 for _ in range(len(noChangePaths))]
        
        self.pdfPaths =  changePaths + noChangePaths
        self.dataTransform = dataTransform
        self.labels = changeLabels +  noChangeLabels
        
    def __len__(self):
        return len(self.pdfPaths)
    
    def __getitem__(self, idx):
        pdfSheetImages = pdf2image.convert_from_path(self.pdfPaths[idx], dpi=100)[:4]
        pdfSheetImages = [p.convert(mode='L') for p in pdfSheetImages]
        pdfSheetImages = [self.dataTransform(p) for p in pdfSheetImages]
        array = np.stack([np.asarray(p) for p in pdfSheetImages], 0)
        tensor = from_numpy((array.astype('float32')-127.5)/127.5)
        label = LT([self.labels[idx]])
        return tensor, label

    
class RandomPad(object):
    """
    Pad the given PIL Image on all sides with the random "pad" value within the window.
    """
    def __init__(self, padWindow, fill=255):
        self.padWindow = padWindow
        self.fill = fill

    def __call__(self, img):
        if self.padWindow != 0:
            padding1 = np.random.randint(0, self.padWindow + 1)
            padding2 = np.random.randint(0, self.padWindow + 1)
            img = ImageOps.expand(img, border=(padding1, padding2), fill=self.fill)
        return img

    
class RandomRotate(object):
    """
    Rotate the given PIL Image within the +/-window.
    """
    def __init__(self, angleWindow):
        self.angleWindow = angleWindow

    def __call__(self, img):
        if self.angleWindow !=0:
            angle = np.random.randint(-self.angleWindow, self.angleWindow + 1)
            img = img.rotate(angle=angle)
        return img


def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.
    Notably used in RandomResizedCrop.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

    
class RandomResizedCropModified(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
    

def _is_pil_image(img):
    return isinstance(img, Image.Image)
    
    
def loadPdfData(dataPath, batchSize=16, rotationAngle=0, shear=0, scalePerc=0, translatePerc=0, brightness=0, contrast=0, saturation=0, hue=0):
    """
    Function to load, preprocess and augment data from path
    """
    """
    pdfTransform = transforms.Compose([transforms.Resize((HEIGHT, WIDTH)),
                                       transforms.Pad(50, fill=255), RandomRotate(rotationAngle), 
                                       transforms.CenterCrop((HEIGHT, WIDTH)),
                                       RandomPad(padWindow=padWindow), transforms.RandomCrop((HEIGHT, WIDTH)),
                                       RandomPad(padWindow=padWindow), transforms.Resize((HEIGHT, WIDTH)),
                                       transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
                                      ])
    
    pdfTransform = transforms.Compose([transforms.Resize((HEIGHT, WIDTH)), 
                                       transforms.RandomAffine(degrees=1, translate=(0.1, 0.1), shear=1, fillcolor=255)]) # ,  scale=0)
    """
    pdfTransform = transforms.Compose([transforms.Resize((HEIGHT, WIDTH)),
                                       transforms.RandomAffine(degrees=rotationAngle, shear=shear, scale=(1 - scalePerc, 1 + scalePerc), 
                                                               translate=(translatePerc, translatePerc), fillcolor=255),
                                       transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)])
    
    dataset = PDFDataset(dataPath=dataPath, dataTransform=pdfTransform)
    dataLoader = DataLoader(dataset, batch_size=batchSize, num_workers=16, shuffle=True, drop_last=True)
    
    image, label =  iter(dataLoader).next()
    print(f'Data Loaded - # Samples: {str(len(dataLoader) * image.size(0))} | Image Shape: {str(image.size())} | Label Shape: {str(label.size())}')
    return dataLoader
