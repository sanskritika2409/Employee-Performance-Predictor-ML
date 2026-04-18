import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_

    plt.bar(feature_names, importance)
    plt.xticks(rotation=45)
    plt.title("Feature Importance")
    plt.savefig("outputs/feature_importance.png")
    plt.show()