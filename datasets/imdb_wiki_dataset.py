import argparse
import pandas as pd
from IMDBWIKI import IMDBWIKI
import os
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='imdb_wiki', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='number of workers')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()


###
#
# agrs:
#       leave_out_train : if train the imbalance
#       train_age: the range of class for train
#       test_group : the group of ages for test ( the left code should be updated )
#
####
def get_dataset(args, leave_out_train = False, train_age = [], test_group = []):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    leave_list = [i for i in range(0,101)]
    train_list = []
    test_list = []
    val_list = []
    # train the range of ages like : [20 to 30]
    for i in train_age:
        age_is = i
        df_train_cur, df_val_cur, df_test_cur = df_train[df_train['age'] == \
            age_is], df_val[df_val['age'] == age_is], df_test[df_test['age'] == age_is]
        df_train_cur, df_val_cur, df_test_cur = shuffle(df_train_cur), shuffle(df_val_cur), shuffle(df_test_cur)
        # train the imbalance, the first class is : len(train_number) others are both 1000
        if leave_out_train:
            if i == 0:
                df_train_cur = df_train_cur[:args.train_number]
                train_list.append(df_train_cur)
            else:
                train_list.append(df_train_cur[:1000])
                test_list.append(df_test_cur[:1000])
                val_list.append(df_val_cur[:1000])
        else:
            train_list.append(df_train_cur)
            test_list.append(df_test_cur)
            val_list.append(df_val_cur)
        ####
    df_train = pd.concat(train_list)
    df_test = pd.concat(test_list)
    df_val = pd.concat(val_list)

    train_labels = df_train['age']

    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train')
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val')
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, test_loader, val_loader