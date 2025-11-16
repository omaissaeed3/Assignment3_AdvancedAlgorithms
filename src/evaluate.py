# Optional evaluation helpers
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

def plot_and_save_confusion(cm, labels, fname):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(fname)
    plt.close()

def plot_and_save_roc(y_true, y_proba, fname):
    disp = RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.savefig(fname)
    plt.close()
