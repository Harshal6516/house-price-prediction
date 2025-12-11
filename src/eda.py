import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(df["SalePrice"], kde=True)
    plt.title("SalePrice Distribution")
    plt.show()

def plot_correlations(df):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()
