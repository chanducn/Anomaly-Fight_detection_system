from dataset_loader import FrameSequenceDataset,get_transform
from torch.utils.data import DataLoader

train_dataset = FrameSequenceDataset('processed_data',split='train',transform=get_transform())
val_dataset = FrameSequenceDataset('processed_data',split = 'val',transform =get_transform())


train_loader = DataLoader(train_dataset,batch_size=8,shuffle =True)
val_loader = DataLoader(val_dataset,batch_size=8)