from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Arial"
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
# sns.set_style("whitegrid")
sns.set_context("talk")  # paper, notebook, talk, poster
centers = [[0, 0], ]

plt.figure()
X, labels_true = make_blobs(n_samples=200, centers=centers, cluster_std=1)
plt.scatter(X[:, 0], X[:, 1], c=sns.color_palette("dark")[0], s=5)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.savefig("exps_final/figures/test/pc200.pdf", bbox_inches="tight", dpi=300)
# plt.show()
plt.figure()
X, labels_true = make_blobs(n_samples=2000, centers=centers, cluster_std=2.5)
plt.scatter(X[:, 0], X[:, 1], c=sns.color_palette("dark")[1], s=5)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.savefig("exps_final/figures/test/pc1000.pdf", bbox_inches="tight", dpi=300)
x = 1
