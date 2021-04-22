import os
import random
from DatasetMedical import DatasetCAMUS, DatasetCAMUS_r, DatasetTEE
from pathlib import Path

def batch_to_img(xb, idx):
    img = np.array(xb[idx, 0:3].cpu())
    return img.transpose((1, 2, 0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def get_random_folder_split(path):
    x_path = path / 'train_gray'
    gray_files = os.listdir(x_path)
    no_files = len(gray_files)
    indices = list(range(no_files))
    random.Random(1).shuffle(indices)
    train_files = [gray_files[i] for i in indices[:int(np.floor(0.7*no_files))]]
    val_files = [gray_files[i] for i in indices[int(np.floor(0.7*no_files)): int(np.floor(0.85*no_files))]]
    test_files = [gray_files[i] for i in indices[int(np.floor(0.85*no_files)):]]
    return train_files, val_files, test_files

def get_train_val_set(dataset, is_local, pre_process, transform, isotropic, include_es, include_2ch, include_4ch):
    base_path = Path('/work/datasets/medical_project')/dataset
    
    if dataset == 'CAMUS_resized': 
        train_files, val_files, _ = get_random_folder_split(base_path)
        train_dataset = DatasetCAMUS_r(base_path / 'train_gray', train_files,
                                        base_path / 'train_gt', pre_processing=pre_process, transform=transform)
        valid_dataset = DatasetCAMUS_r(base_path / 'train_gray', val_files,
                                        base_path / 'train_gt', pre_processing=pre_process, transform=transform)
    elif dataset == 'CAMUS':
        base_path = Path('data') if is_local else base_path
        train_dataset = DatasetCAMUS(
            base_path,
            pre_processing=pre_process,
            transform=transform,
            isotropic=isotropic,
            include_es=include_es,
            include_2ch=include_2ch,
            include_4ch=include_4ch,
            start=1,
            stop=300)
        valid_dataset = DatasetCAMUS(base_path, isotropic=isotropic, include_es=include_es, include_2ch=include_2ch, include_4ch=include_4ch, start=301, stop=400)
        test_dataset = DatasetCAMUS(base_path, isotropic=isotropic, include_es=include_es, include_2ch=include_2ch, include_4ch=include_4ch, start=401, stop=450)
        print(len(train_dataset), len(valid_dataset), len(test_dataset))
    elif dataset == 'TEE':
        train_files, val_files, _ = get_random_folder_split(base_path)
        train_dataset = DatasetTEE(base_path / 'train_gray', train_files,
                                   base_path / 'train_gt', pre_processing=pre_process, transform=transform)
        valid_dataset = DatasetTEE(base_path / 'train_gray', val_files,
                                    base_path / 'train_gt', pre_processing=pre_process, transform=transform)

    return train_dataset, valid_dataset, test_dataset