
train-paper-experiments-1:
	python train.py -d 2 -n 1_600_000 --n-eval 1_000 -b 16 --device cpu
	python train.py -d 3 -n 1_600_000 --n-eval 1_000 -b 16 --device cpu
	python train.py -d 4 -n 1_600_000 --n-eval 1_000 -b 16 --device cpu
	python train.py -d 5 -n 1_600_000 --n-eval 1_000 -b 16 --device cpu
eval-runs:
	python test.py --test-size 50 -f run
	python test.py --test-size 50 -f runv2
	python test.py --test-size 50 -f runv3
	python test.py --test-size 50 -t -f run
	python test.py --test-size 50 -t -f runv2
	python test.py --test-size 50 -t -f runv3
	python run_analysis.py
	python run_analysis.py -t