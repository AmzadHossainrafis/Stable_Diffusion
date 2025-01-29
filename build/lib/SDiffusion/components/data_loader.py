from torch.utils.data import Dataset
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, image_paths, texts, tokenizer, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        
        if self.transform:
            image = self.transform(image)
        
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return image, input_ids, attention_mask
    
