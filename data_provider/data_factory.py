from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from data_provider import Dataset_Weather_Stations, Dataset_Weather_Stations_ALL
from data_provider import Dataset_Yahoo_Finance, Dataset_AG_Delhi, Dataset_NIFTY
from data_provider import Dataset_Weather, Dataset_PEMS_Traffic 

from data_provider.sampler import MyRandomSampler

from torch.utils.data import DataLoader

data_dict = {
	'ETTh1': Dataset_ETT_hour,
	'ETTh2': Dataset_ETT_hour,
	'ETTm1': Dataset_ETT_minute,
	'ETTm2': Dataset_ETT_minute,

	'Dataset_Weather_Station': Dataset_Weather_Stations,
	'Dataset_Weather_Stations_ALL': Dataset_Weather_Stations_ALL,
	
    'JenaWeather': Dataset_Weather,

    'CATraffic': Dataset_PEMS_Traffic,

    'YahooFinance': Dataset_Yahoo_Finance,

    'AGDelhiWeather': Dataset_AG_Delhi,

    'NIFTYStocks': Dataset_NIFTY,

	'custom': Dataset_Custom,
}


def data_provider(args, flag):
	Data = data_dict[args.data]
	timeenc = 0 if args.embed != 'timeF' else 1

	if flag == 'test':
		shuffle_flag = False
		drop_last = False
		batch_size = args.batch_size
		freq = args.freq
	elif flag == 'pred':
		shuffle_flag = False
		drop_last = False
		batch_size = 1
		freq = args.detail_freq
		Data = Dataset_Pred
	else:
		shuffle_flag = True
		drop_last = False
		batch_size = args.batch_size
		freq = args.freq
	
	args.scale = None if args.scale=="none" else args.scale
	
	if args.data == "Dataset_Weather_Stations_ALL":
		data_set = Data(
			root_path=args.root_path,
			data_path=args.data_path,
			flag=flag,
			size=[args.seq_len, args.label_len, args.pred_len],
			features=args.features,
			target=args.target,
			timeenc=timeenc,
			freq=freq,
			seasonal_patterns="Hourly" if args.freq == 'h' else None
		)
		print(flag, len(data_set), 'self')
		data_loader = DataLoader(
			data_set,
			num_workers=args.num_workers,
			sampler = MyRandomSampler(
									data_set,
									batch_size=batch_size if flag == 'train' else batch_size*10,
									drop_last=drop_last  if flag == 'train' else False,
									warm_batch_size = True if flag == 'train' else False,
									shuffle = True if  flag == 'train' else False,
									infinite = True if  flag == 'train' else False
									),
			persistent_workers=True,
			prefetch_factor=3,
			pin_memory = False,
			)
		return data_set, data_loader

	data_set = Data(
		root_path=args.root_path,
		data_path=args.data_path,
		flag=flag,
		size=[args.seq_len, args.label_len, args.pred_len],
		features=args.features,
		target=args.target,
		timeenc=timeenc,
		freq=freq,
		cycle=args.factor,
		scale=args.scale
	)
	data_loader = DataLoader(
		data_set,
		batch_size=batch_size,
		shuffle=shuffle_flag,
		num_workers=args.num_workers,
		drop_last=drop_last)
	return data_set, data_loader
