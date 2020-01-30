from PIL import Image
import torch
import os

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class ScenicScoreDataset(torch.utils.data.Dataset):
  """ Scenic Score Dataset. """
  def __init__(self, images, scores, transform=None, resizeImage=False, root_dir=''):
    'Initialization'
    self.images = images
    self.scores = scores
    self.root_dir = root_dir
    self.transform = transform
    self.resizeImage = resizeImage

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.images)

  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()

    # get image
    img_name = os.path.join(self.root_dir,
                            self.images.iloc[index])
    image = Image.open(img_name)
    if self.resizeImage:
        # 256x256 according to http://places2.csail.mit.edu/download.html
        image = image.resize((256, 256))
    image.convert('RGB')

    # get score
    score = self.scores.iloc[index]

    if self.transform:
      image = self.transform(image)

    return image,score

