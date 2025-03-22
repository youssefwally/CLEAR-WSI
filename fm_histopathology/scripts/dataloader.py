#imports
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk


class BasicDataset(Dataset):
    def __init__(self, file_list, path, label="DX", transform=None):
        self.file_list = file_list
        self.transform = transform
        self.path = path
        self.chosen_label = label

    def __len__(self):
        self.length = len(self.file_list)
        return self.length

    def __getitem__(self, idx):
        path = f"{path}/{self.file_list[idx]}/mwp1orig_out.nii.gz"
        
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        img_transformed = self.transform(img)

        label = self.original_tabular_data.loc[self.original_tabular_data['IMAGEUID'] == int(self.file_list[idx])][self.chosen_label].item()
        label = 0.0 if label == "CN" else 1.0

        return img, label