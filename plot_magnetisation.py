import numpy as np
import plotly.graph_objects as go


if __name__ == "__main__":
    magnetisation = np.loadtxt("magnetisation.dat")
    data = [{"type": "scatter",
             "mode": "lines",
             "x": np.arange(0, len(magnetisation)),
             "y": magnetisation}]
    layout = {"title": {"text": "Relative Magnetisation"},
              "template": "plotly",
              "annotations": [{"name": "draft watermark",
                               "text": "DRAFT",
                               "textangle": -30,
                               "opacity": 0.1,
                               "font": {"color": "black", "size": 100},
                               "xref": "paper",
                               "yref": "paper",
                               "x": 0.5,
                               "y": 0.5,
                               "showarrow":False}]
             }
    fig = go.Figure({"data": data, "layout": layout})

    fig.write_image("magnetisation.png")
