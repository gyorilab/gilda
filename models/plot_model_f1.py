import matplotlib.pyplot as plt
from gilda.grounder import load_gilda_models


if __name__ == '__main__':
    models = load_gilda_models(-1)
    f1s = [m.stats['f1']['mean'] for m in models.values()]
    plt.ion()
    plt.figure(figsize=(8, 4))
    plt.hist(f1s, 100, color='gray')
    plt.xlabel('Mean F1 of disambiguation model', fontsize=18)
    plt.ylabel('Number of models', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axvline(x=0.7, linestyle='--', color='gray')
    plt.subplots_adjust(left=0.114, right=0.98, bottom=0.157, top=0.97)
    plt.show()
    plt.savefig('gilda_model_f1_plots.pdf')