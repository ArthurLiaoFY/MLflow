# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from sklearn import datasets

from ml_models.kernel_density import epanechnikov
from ml_models.linear_models.models.local_model import LocalLinearModel

iris = datasets.load_iris()["data"]
y = iris[:, 2]
X = iris[:, 0]

lpm = LocalLinearModel(
    bandwidth=0.5,
    kernel_func=epanechnikov,
    equal_space_knots=False,
)
lpm.fit(X=X, y=y)
y_hat = lpm.predict(X=X)


f1 = lpm.plot_residual(X=X, y=y)
f2 = lpm.plot_fitted_value(X=X, y=y)
f3 = lpm.plot_fitted_line(X=X, y=y)

# %%
