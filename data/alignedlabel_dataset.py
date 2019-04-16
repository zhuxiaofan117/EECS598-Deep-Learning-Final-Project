import os
import os.path

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    labels = []
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    i = -1
    for root, subdir, fnames in sorted(os.walk(dir)):
        i += 1
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                labels.append(i)
    print(type(labels[0]))
    return (images[:min(max_dataset_size, len(images))], labels[:min(max_dataset_size, len(images))])


class AlignedlabelDataset(BaseDataset):

    def __init__(self, opt):
        
        BaseDataset.__init__(self, opt)

        self.root = os.path.join(opt.dataroot, opt.phase)  # get the image directory

        imgs, labels  = make_dataset(self.root)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + self.root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.AB_paths = imgs
        self.labels = labels

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        labels = self.labels[index]
        # print(type(labels))
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B,'labels':labels, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)




