import os
import base64
import subprocess
import io
from AIDetector_pytorch import Detector
import imutils
import cv2
import pandas as pd
from textwrap import dedent
import dash
from dash import dcc
from dash import html
import dash_player as player
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pathlib
from datetime import date

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Lost and Found Detection Explorer"



def markdown_popup():
    return html.Div(
        id="markdown",
        className="modal",
        style={"display": "none"},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=[
                            dcc.Markdown(
                                children=dedent(
                                    """
                                ##### What am I looking at?

                                This app enhances visualization of objects detected using state-of-the-art Mobile Vision Neural Networks.
                                Most user generated videos are dynamic and fast-paced, which might be hard to interpret. A confidence
                                heatmap stays consistent through the video and intuitively displays the model predictions. The pie chart
                                lets you interpret how the object classes are divided, which is useful when analyzing videos with numerous
                                and differing objects.

                                """
                                )
                            )
                        ],
                    ),
                ],
            )
        ),
    )


# Main App
app.layout = html.Div(
    children=[
        dcc.Interval(id="interval-updating-graphs", interval=1000, n_intervals=0),
        html.Div(id="top-bar", className="row"),
        html.Div(
            className="container",
            children=[
                html.Div(
                    id="left-side-column",
                    className="eight columns",
                    children=[
                        html.Img(
                            id="logo-mobile", src=app.get_asset_url("cat-logo.png")
                        ),
                        html.Div(
                            id="header-section",
                            children=[
                                html.H4("Lost and Found Detection Explorer"),
                                html.P(
                                    "This is a test :)"
                                ),
                                html.Button(
                                    "Learn More", id="learn-more-button", n_clicks=0, style={'display': 'inline-block', 'margin-right': '20px'}
                                ),
                                html.Button('Run backend', id='run-backend-btn'),
                                html.Div(id='backend-output'),
                            ],
                        ),
                        html.Div(
                            className="control-section",
                            children=[

                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(children=["Footage Selection:"]),
                                        dcc.Dropdown(
                                            id="dropdown-footage-selection",
                                            options=[
                                                {
                                                    "label": "test video 1",
                                                    "value": "test1.mp4",
                                                },
                                                {
                                                    "label": "test video 2",
                                                    "value": "test2.mp4",
                                                },
                                                {
                                                    "label": "test video 3",
                                                    "value": "test3.mp4",
                                                },

                                            ],
                                            value="test3.mp4",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(id='footage-output', style={'display': 'inline-block', 'margin-top': '10px','margin-bottom': '10px',  'text-align': 'right'}),
                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(children=["Category of Lost Items:"]),
                                        dcc.Dropdown(
                                            id="items-dropdown",
                                            options=[
                                                {
                                                    "label": "suitcase",
                                                    "value": "suitcase",
                                                },
                                                {
                                                    "label": "backpack",
                                                    "value": "backpack",
                                                },
                                                {
                                                    "label": "laptop",
                                                    "value": "laptop",
                                                },
                                            ],
                                            value="suitcase",
                                            searchable=False,
                                            clearable=False,
                                        ),

                                    ],
                                ),
                                # 创建输出文本
                                html.Div(id='output', style={'display': 'inline-block', 'margin-top': '10px','margin-bottom': '10px',  'text-align': 'right'}),
                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(children=["Graph View Mode:"]),
                                        dcc.Dropdown(
                                            id="dropdown-graph-view-mode",
                                            options=[
                                                {
                                                    "label": "Visual Mode",
                                                    "value": "visual",
                                                },
                                                {
                                                    "label": "Detection Mode",
                                                    "value": "detection",
                                                },
                                            ],
                                            value="visual",
                                            searchable=False,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="right-side-column",
                    className="four columns",
                    children=[
                        html.Div(
                            className="img-container",
                            children=html.Img(
                                id="logo-web", src=app.get_asset_url("cat-logo.png")
                            ),
                        ),
                        html.Div(id="div-visual-mode"),
                        html.Div(id="div-detection-mode"),
                    ],
                ),
            ],
        ),
        markdown_popup(),
    ]
)


def list_txt(path, list=None):

    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist


def _nn_euclidean_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return np.maximum(0.0, r2.min(axis=0))


def _pdist(a, b, single_embedding):
    new = np.asarray(a)
    known = np.asarray(b)
    if len(new) == 0 or len(known) == 0:
        return np.zeros((len(new), len(known)))
    new2, known2 = np.square(new).sum(axis=1), np.square(known).sum(axis=1)
    r2 = -2. * np.dot(new, known.T) + new2[:, None] + known2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    # import pytest
    # pytest.set_trace()
    return r2


def _nn_euclidean_distance(x, y, single_embedding):
    distances = _pdist(x, y, single_embedding)
    return np.maximum(0.0, distances.min(axis=0))


def run_backend():
    # Run the backend program and capture its output
    my_variable = open('my_item.txt').read()
    if my_variable is None:
        my_variable = 'suitcase'
    item_to_detect = ['person', my_variable]
    det = Detector(item_to_detect)
    my_selected_value = open('my_footage.txt').read()
    if my_selected_value is None:
        my_selected_value = 'videos/test3.mp4'

    name_list = []
    known_embedding = []
    name_list, known_embedding = det.loadIDFeats()
    print(name_list, known_embedding)
    list_txt(path='name_list.txt', list=name_list)

    fw = open('known_embedding.txt', 'w')
    for line in known_embedding:
        for a in line:
            fw.write(str(a))
            fw.write('\t')
        fw.write('\n')
    fw.close()

    cap = cv2.VideoCapture(my_selected_value)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(100 / fps)
    framecounter = 0
    videoWriter = None
    targetLocked = False
    minIndex = None
    trackId = None
    image_paths = []
    while True:

        success, im = cap.read()
        if im is None:
            break
        DetFeatures = []
        DetFeatures, img_input, box_input = det.loadDetFeats(im)
        # detFeatures = np.array(DetFeatures)
        if len(DetFeatures) > 0 and not targetLocked:
            dist_matrix = _nn_euclidean_distance(known_embedding, DetFeatures, known_embedding[0])
            minIndex = dist_matrix.argmin()
            if trackId is None:
                trackId = minIndex + 1
            det.targetTrackId = trackId
            targetLocked = True

        result = det.feedCap(im)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        if det.isLost is True:
            framecounter = framecounter + 1
            print('lost')

        if framecounter == 15:
            cv2.imwrite(f'./test-{det.frameCounter / fps}-second.png', result)
            image_path = f'./test-{det.frameCounter / fps}-second.png'
            image_paths.append(image_path)
            framecounter = 0

    # Parse the output to get the video and images
    video_path = 'result.mp4'

    return video_path, image_paths


@app.callback(Output('backend-output', 'children'),
              [Input('run-backend-btn', 'n_clicks')])
def update_backend_output(n_clicks):
    if n_clicks:
        video_path, image_paths = run_backend()
        video_content = open(video_path, 'rb').read()
        image_contents = [open(path, 'rb').read() for path in image_paths]
        children = [
            html.Video(src=f'data:video/mp4;base64,{base64.b64encode(video_content).decode()}',
                       style={'position': 'absolute',
                              'top': '30px', 'right': '50px'},
                       controls=True),
            *[html.Img(src=f'data:image/png;base64,{base64.b64encode(content).decode()}',
                       style={'width': '20%', 'height': '20%',
                              'display': 'inline-block', 'float': 'right',
                              'position': 'absolute',
                              'top': f'{(i+1)*180}px', 'right': '120px'}) for i, content in enumerate(image_contents)]
        ]
        return children


@app.callback(
    Output('output', 'children'),
    Input('items-dropdown', 'value')
)
def update_output(value):
    my_variable = None
    if value == 'suitcase':
        my_variable = 'suitcase'
    elif value == 'backpack':
        my_variable = 'backpack'
    elif value == 'laptop':
        my_variable = 'laptop'
    list_txt(path='my_item.txt', list=my_variable)
    return f'You have selected {value}!'


# Learn more popup
@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context
    prop_id = ""
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if prop_id == "learn-more-button":
        return {"display": "block"}
    else:
        return {"display": "none"}


# 定义回调函数，当下拉菜单选项改变时，更新后端变量和输出文本
@app.callback(
    Output('footage-output', 'children'),
    Input('dropdown-footage-selection', 'value')
)
def update_footage_output(selected_value):
    my_selected_value = None
    if selected_value == 'test1.mp4':
        my_selected_value = 'videos/test1.mp4'
    elif selected_value == 'test2.mp4':
        my_selected_value = 'videos/.mp4'
    elif selected_value == 'test3.mp4':
        my_selected_value = 'videos/test3.mp4'
    list_txt(path='my_footage.txt', list=my_selected_value)
    return f'You have selected {selected_value}!'


if __name__ == '__main__':
    app.run_server(debug=True)
