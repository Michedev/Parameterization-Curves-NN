import matplotlib.pyplot as plt
import pandas as pd
from path import Path

TARGET_FILE = Path(__file__).parent / 'train_running_avg_loss.csv'
ROOT = TARGET_FILE.parent.parent
PLOTS_FOLDER = ROOT / 'plots'
SLIDES_FOLDER = ROOT / 'slides'


def main():
    data = pd.read_csv(TARGET_FILE)
    plt.plot(data.Step, data.Value, c='blue')
    plt.scatter([6000], data.Value[data.Step == 6000], c='red')

    fname = 'train_running_avg_loss.png'
    plt.savefig(PLOTS_FOLDER / fname)
    if SLIDES_FOLDER.exists():
        plt.savefig(SLIDES_FOLDER / fname)


if __name__ == '__main__':
    main()
