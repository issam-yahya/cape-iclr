"""
Dataset registry + DataLoader construction.

Self-contained subset of the Time-Series-Library data factory: long-term
forecasting only (the anomaly-detection / classification / M4 branches and
their sktime dependency are not needed to reproduce the paper).
"""

from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
)

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = flag not in ('test', 'TEST')

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=getattr(args, 'seasonal_patterns', None),
    )
    if getattr(args, 'verbose', False):
        print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_set, data_loader
