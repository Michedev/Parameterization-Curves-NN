
train-paper-experiments-1:
	python train.py -d 2 -n 1_600_000 --n-eval 1_000 -b 32 --device cpu
	python train.py -d 3 -n 1_600_000 --n-eval 1_000 -b 32 --device cpu
	python train.py -d 4 -n 1_600_000 --n-eval 1_000 -b 32 --device cpu
	python train.py -d 5 -n 1_600_000 --n-eval 1_000 -b 32 --device cpu
train-paper-experiments-1-cuda0:
	python train.py -d 2 -n 1_600_000 --n-eval 1_000 -b 32 --device cuda:0
	python train.py -d 3 -n 1_600_000 --n-eval 1_000 -b 32 --device cuda:0
	python train.py -d 4 -n 1_600_000 --n-eval 1_000 -b 32 --device cuda:0
	python train.py -d 5 -n 1_600_000 --n-eval 1_000 -b 32 --device cuda:0
train-paper-experiments-1-cuda1:
	python train.py -d 2 -n 1_600_000 --n-eval 1_000 -b 32 --device cuda:1
	python train.py -d 3 -n 1_600_000 --n-eval 1_000 -b 32 --device cuda:1
	python train.py -d 4 -n 1_600_000 --n-eval 1_000 -b 32 --device cuda:1
	python train.py -d 5 -n 1_600_000 --n-eval 1_000 -b 32 --device cuda:1

eval-runs:
	python test.py --test-size 1000 -f run
	python test.py --test-size 1000 -f run -t
	python run_analysis.py
	python run_analysis.py -t