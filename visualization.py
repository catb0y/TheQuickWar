from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd


if __name__ == '__main__':
    df = pd.read_pickle("res.pkl")
    x_lables = [str(dt[0]) + "-" + str(dt[1]) for dt in df.index.values]
    df.topic_3 = df.topic_3 + df.topic_5

    trace0 = dict(
        x=x_lables,
        y=df.topic_0.values,
        mode='lines',
        line=dict(width=0.5, color='rgb(106, 162, 106)'),
        stackgroup='one',
        groupnorm='percent',
        name="Wirtschaft"
    )
    trace1 = dict(
        x=x_lables,
        y=df.topic_1.values,
        mode='lines',
        line=dict(width=0.5, color='rgb(152, 139, 170)'),
        stackgroup='one',
        name="Gesellschaft & Politik"
    )
    trace2 = dict(
        x=x_lables,
        y=df.topic_2.values,
        mode='lines',
        line=dict(width=0.5, color='rgb(202, 154, 107)'),
        stackgroup='one',
        name="Krieg"
    )
    trace3 = dict(
        x=x_lables,
        y=df.topic_3.values,
        mode='lines',
        line=dict(width=0.5, color='rgb(204, 204, 122)'),
        stackgroup='one',
        name="Schiffahrt & Finanzen"
    )
    trace4 = dict(
        x=x_lables,
        y=df.topic_4.values,
        mode='lines',
        line=dict(width=0.5, color='rgb(102, 138, 171)'),
        stackgroup='one',
        name="Inserate"
    )

    data = [trace0, trace1, trace2, trace3, trace4]
    layout = go.Layout(
        showlegend=True,
        xaxis=dict(
            type='category',
        ),
        yaxis=dict(
            type='linear',
            range=[1, 100],
            dtick=20,
            ticksuffix='%'
        )
    )
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='stacked-area-plot-norm')
