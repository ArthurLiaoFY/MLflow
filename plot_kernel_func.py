import plotly
import plotly.graph_objects as go

from ml_models.density_estimate.kernel_density import *

x_span = np.linspace(start=-1.5, stop=1.5, num=401)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([uniform(x, h=1) for x in x_span]),
        name="uniform",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([triangular(x, h=1) for x in x_span]),
        name="triangular",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([epanechnikov(x, h=1) for x in x_span]),
        name="epanechnikov",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([quartic(x, h=1) for x in x_span]),
        name="quartic",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([triweight(x, h=1) for x in x_span]),
        name="triweight",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([tricube(x, h=1) for x in x_span]),
        name="tricube",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([cosine(x, h=1) for x in x_span]),
        name="cosine",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([gaussian(x, h=1) for x in x_span]),
        name="gaussian",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([logistic(x, h=1) for x in x_span]),
        name="logistic",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([sigmoid(x, h=1) for x in x_span]),
        name="sigmoid",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([silverman(x, h=1) for x in x_span]),
        name="silverman",
    )
)

fig.update_layout(title_text="Kernel Functions", title_x=0.5)
plotly.offline.plot(fig, filename="kernel_density.html")
