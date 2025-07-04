import os 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FrameSequenceDataset(Dataset):
    def __init__(self, data_dir, split='train', sequence_length=16, transform=None):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.transform = transform


        # here we eill defin path
        self.root_dir = os.path.join(data_dir,split)
        self.classes = ['NonFight','Fight'] # 0 - Nonfight , 1 - Fight

        #Traverse each class folder and collect clip paths
        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir,class_name)
            for clip_folder in os.listdir(class_path):
                clip_path  = os.path.join(class_path,clip_folder)
                self.data.append(clip_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        clip_path = self.data[idx]
        label = self.labels[idx]
        frames = []

        # load and transform 16 frmaes
        for i in range(self.sequence_length):
            frame_path = os.path.join(clip_path,f'frame_{i}.jpg')
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        

        # stack frames into tensor  : shape( 16,3,h,w)

        clip_tensor = torch.stack(frames)
        return clip_tensor,label
def get_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)), # smaller size = faster training
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ]
    )