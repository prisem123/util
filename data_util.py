from torch.utils.data import DataLoader, sampler
import torchvision.datasets as dset
import torchvision.transforms as T

def get_Data(dataset, transform=None):
    # load train/val/test set from given dataset.
    
    data_train_val = dataset(root=f'./{dataset.__name__}/data', train=True, 
                             download=True, transform=T.ToTensor())
    data_test = dataset(root=f'./{dataset.__name__}/data', train=False, 
                        download=True, transform=T.ToTensor())
    
    if transform:
        data_train_val = transform(data_train_val)
        data_test = transform(data_test)
        
    print()
    print('- Dataset loaded -')
    print('Train data shape:', data_train_val.data.shape)
    print('Test data shape:', data_test.data.shape)
    print()
    
    return data_train_val, data_test


def get_DataLoader(data_train_val, data_test, val_ratio=0.1, batch_size=64):
    # get DataLoader from given dataset.
    
    data_len = len(data_train_val.data)
    train_len = int(data_len * (1 - val_ratio))

    loader_train = DataLoader(data_train_val, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(train_len)))
    loader_val = DataLoader(data_train_val, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(train_len, data_len)))
    loader_test = DataLoader(data_test, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    print('- DataLoader loaded -')
    print('batch size:', batch_size)
    print('Validation ragio:', val_ratio)
    print()
    
    return loader_train, loader_val, loader_test