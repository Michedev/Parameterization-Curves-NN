
train-paper-experiments-1:
	python train.py -d 2 -n 100_000 --n-eval 1_000 -b 16 --device cpu
	python train.py -d 3 -n 100_000 --n-eval 1_000 -b 16 --device cpu
	python train.py -d 4 -n 100_000 --n-eval 1_000 -b 16 --device cpu
	python train.py -d 5 -n 100_000 --n-eval 1_000 -b 16 --device cpu