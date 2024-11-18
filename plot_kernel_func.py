import plotly
import plotly.graph_objects as go

from models.linear_models.kernel_functions import *

x_span = np.linspace(start=-1.5, stop=1.5, num=401)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([uniform_kernel(x, h=1) for x in x_span]),
        name="uniform",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([triangular_kernel(x, h=1) for x in x_span]),
        name="triangular",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([epanechnikov_kernel(x, h=1) for x in x_span]),
        name="epanechnikov",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([quartic_kernel(x, h=1) for x in x_span]),
        name="quartic",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([triweight_kernel(x, h=1) for x in x_span]),
        name="triweight",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([tricube_kernel(x, h=1) for x in x_span]),
        name="tricube",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([cosine_kernel(x, h=1) for x in x_span]),
        name="cosine",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([gaussian_kernel(x, h=1) for x in x_span]),
        name="gaussian",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([logistic_kernel(x, h=1) for x in x_span]),
        name="logistic",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([sigmoid_kernel(x, h=1) for x in x_span]),
        name="sigmoid",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_span,
        y=np.array([silverman_kernel(x, h=1) for x in x_span]),
        name="silverman",
    )
)

fig.update_layout(title_text="Kernel Functions", title_x=0.5)
plotly.offline.plot(fig, filename="kernel_density.html")
