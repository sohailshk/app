
import smtplib
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
from river import time_series
from river import anomaly
from river import preprocessing
from river import linear_model
from river import optim
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Output, Input, State
import threading
import queue
import time
import warnings
import winsound
import pyttsx3

warnings.filterwarnings("ignore")

FEATURE_RANGES = {
    "pCut::Motor_Torque": (0, 10),
    "pCut::CTRL_Position_controller::Lag_error": (0, 10),
    "pCut::CTRL_Position_controller::Actual_position": (0, 100),
    "pCut::CTRL_Position_controller::Actual_speed": (0, 200),
    "pSvolFilm::CTRL_Position_controller::Actual_position": (0, 100),
    "pSvolFilm::CTRL_Position_controller::Actual_speed": (0, 200),
    "pSvolFilm::CTRL_Position_controller::Lag_error": (0, 10),
    "pSpintor::VAX_speed": (0, 300),
}


def send_email_anomaly_alert(anomaly_info):
    """
    Sends an email alert when an anomaly is detected.
    """
    subject = "Anomaly Detected!"
    body = (
        f"An anomaly was detected.\n\n"
        f"Value: {anomaly_info['value']}\nScore: {anomaly_info['score']}\nTime: {anomaly_info['time']}"
    )
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "vibeasts01@gmail.com"
    msg["To"] = "shaikhsalif50@gmail.com"

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login("vibeasts01@gmail.com", "saifnasreen21")
            server.send_message(msg)
    except Exception as e:
        print(f"Email sending failed: {e}")


class DataStreamSimulator:
    """
    Simulates a real-time data stream with concept drift, seasonal patterns, and anomalies.
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.current_mean = 0
        self.current_std = 1
        self.change_interval = 100
        self.anomaly_interval = 50
        self.step = 0
        self.streaming = False
        self.lock = threading.Lock()
        self.seasonality_period = 200

    def simulate_stream(self):
        while True:
            with self.lock:
                if not self.streaming:
                    time.sleep(0.1)
                    continue
                current_step = self.step

            seasonal = np.sin(2 * np.pi * current_step / self.seasonality_period)
            if current_step % self.change_interval == 0 and current_step != 0:
                with self.lock:
                    self.current_mean += np.random.uniform(-1.0, 1.0)
                    self.current_std = np.random.uniform(0.5, 1.5)

            data_point = np.random.normal(
                self.current_mean + seasonal, self.current_std
            )
            if current_step % self.anomaly_interval == 0 and current_step != 0:
                anomaly_val = np.random.uniform(-10, 10)
                data_point += anomaly_val

            with self.lock:
                self.step += 1
            yield data_point
            time.sleep(0.05)


class StreamingAnomalyDetector:
    """
    Detects anomalies in streaming data using River's Predictive Anomaly Detection.
    """

    def __init__(self):
        self.predictive_model = time_series.SNARIMAX(
            p=12, d=1, q=12, m=12, sd=1,
            regressor=(
                preprocessing.StandardScaler()
                | linear_model.LinearRegression(optimizer=optim.SGD(0.005))
            ),
        )
        self.model = anomaly.PredictiveAnomalyDetection(
            predictive_model=self.predictive_model,
            horizon=1,
            n_std=3.5,
            warmup_period=15,
        )
        self.anomalies = []
        self.threshold = 0.5

    def update(self, data_point):
        score = self.model.score_one(None, data_point)
        self.model.learn_one(None, data_point)
        is_anomaly = score > self.threshold
        if is_anomaly:
            anomaly_info = {"time": None, "value": data_point, "score": score}
            self.anomalies.append(anomaly_info)
            send_email_anomaly_alert(anomaly_info)
        return is_anomaly, score


df = pd.DataFrame(columns=["time", "value", "is_anomaly", "anomaly_score"])
simulator = DataStreamSimulator()
detector = StreamingAnomalyDetector()
data_queue = queue.Queue()
stop_event = threading.Event()


def data_streaming(simulator, detector, data_queue, stop_event):
    try:
        stream = simulator.simulate_stream()
        while not stop_event.is_set():
            data_point = next(stream)
            is_anomaly, anomaly_score = detector.update(data_point)
            data_queue.put({
                "time": simulator.step,
                "value": data_point,
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_score,
            })
    except Exception as e:
        print(f"Error in data_streaming: {e}")


app = Dash(__name__)

button_style = {
    "padding": "10px",
    "border": "none",
    "border-radius": "12px",
    "font-size": "16px",
    "margin-right": "10px",
    "cursor": "pointer",
    "transition": "transform 0.3s",
}

# Track the user's maintenance score
maintenance_score = 0

app.layout = html.Div(
    style={
        "background": "linear-gradient(to bottom right, #ffffff, #e6e6e6)",
        "padding": "20px",
        "border-radius": "20px",
        "max-width": "1100px",
        "margin": "auto",
        "font-family": "Arial, sans-serif",
        "color": "#333333",
        "box-shadow": "0 4px 30px rgba(0, 0, 0, 0.1)",
    },
    children=[
        dcc.Tabs([
            dcc.Tab(
                label="Anomaly Detection",
                children=[
                    html.H1(
                        "Real-Time Anomaly Detection",
                        style={
                            "text-align": "center",
                            "color": "#2c3e50",
                            "font-family": "Calibri",
                            "font-size": "2.5rem",
                            "margin-bottom": "0.5em",
                        },
                    ),
                    html.H2(
                        "Automated Email Alerts for Detected Anomalies",
                        style={
                            "text-align": "center",
                            "color": "#888888",
                            "font-family": "Calibri",
                            "font-size": "1.2rem",
                            "margin-top": "0",
                            "margin-bottom": "20px",
                        },
                    ),
                    dcc.Graph(
                        id="live-graph",
                        style={"border": "2px solid #cccccc", "border-radius": "10px"},
                    ),
                    dcc.Interval(id="graph-update", interval=500, n_intervals=0),
                    html.Div([
                        dcc.Checklist(
                            id="toggle-anomalies",
                            options=[
                                {"label": "Show Anomaly Regions", "value": "anomalies"},
                                {"label": "Show Trend Line", "value": "trend_line"},
                            ],
                            value=[],
                            inline=True,
                            style={"margin-top": "20px", "font-size": "16px"},
                        ),
                        html.Div(
                            style={"margin-top": "20px", "text-align": "center"},
                            children=[
                                html.Button(
                                    "Start Streaming",
                                    id="start-btn",
                                    n_clicks=0,
                                    style={
                                        **button_style,
                                        "background-color": "#4CAF50",
                                        "color": "white",
                                    },
                                ),
                                html.Button(
                                    "Stop Streaming",
                                    id="stop-btn",
                                    n_clicks=0,
                                    style={
                                        **button_style,
                                        "background-color": "#f44336",
                                        "color": "white",
                                        "display": "none",
                                    },
                                ),
                                html.Button(
                                    "Reset",
                                    id="reset-btn",
                                    n_clicks=0,
                                    style={
                                        **button_style,
                                        "background-color": "#FF9800",
                                        "color": "white",
                                        "display": "none",
                                    },
                                ),
                                html.Button(
                                    "Download Anomalies CSV",
                                    id="download-btn",
                                    n_clicks=0,
                                    style={
                                        **button_style,
                                        "background-color": "#9C27B0",
                                        "color": "white",
                                        "display": "none",
                                    },
                                ),
                            ],
                        ),
                    ]),
                    dcc.Download(id="download-dataframe-csv"),
                    html.Div(
                        id="status",
                        style={
                            "text-align": "center",
                            "margin-top": "10px",
                            "font-size": "18px",
                            "font-weight": "bold",
                        },
                    ),
                    html.Div(id="reset-data", style={"display": "none"}),

                    # Gamification elements
                    html.Div(
                        [
                            html.H3("Maintenance Scoreboard", style={"text-align": "center"}),
                            html.Div(
                                [
                                    html.Span("Current Score: "),
                                    html.Span(id="score-display", style={"font-weight": "bold"}),
                                ],
                                style={"text-align": "center", "font-size": "20px", "marginTop": "10px"},
                            ),
                            html.Button(
                                "Respond to Anomaly",
                                id="respond-anomaly-btn",
                                n_clicks=0,
                                style={**button_style, "background-color": "#2196F3", "color": "white"},
                            ),
                        ],
                        style={
                            "margin-top": "30px",
                            "padding": "10px",
                            "border": "2px solid #ccc",
                            "border-radius": "10px",
                            "text-align": "center",
                        },
                    ),

                    dcc.Store(id="thread-store", data={"streaming_thread": None}),
                    dcc.Store(id="beep-store", data={"beep1_done": False, "beep2_done": False}),
                    dcc.Store(id="score-store", data={"score": 0}),  # Store the user’s score
                ],
            ),
            dcc.Tab(
                label="Feature Input / Range Check",
                children=[
                    html.H2(
                        "Feature Range Checker",
                        style={
                            "text-align": "center",
                            "color": "#333333",
                            "font-family": "Calibri",
                        },
                    ),
                    html.Div(
                        style={"max-width": "600px", "margin": "auto", "text-align": "center"},
                        children=[
                            dcc.Dropdown(
                                id="feature-dropdown",
                                options=[{"label": f, "value": f} for f in FEATURE_RANGES.keys()],
                                placeholder="Select a feature...",
                                style={"width": "80%", "margin": "20px auto"},
                            ),
                            dcc.Input(
                                id="feature-input-value",
                                type="number",
                                placeholder="Enter feature value...",
                                style={"width": "80%", "margin": "20px auto"},
                            ),
                            html.Button(
                                "Submit Value",
                                id="submit-feature-btn",
                                n_clicks=0,
                                style={
                                    **button_style,
                                    "background-color": "#4CAF50",
                                    "color": "white",
                                },
                            ),
                            html.Div(
                                id="feature-validation-msg",
                                style={
                                    "margin-top": "20px",
                                    "color": "#b30000",
                                    "font-weight": "bold",
                                },
                            ),
                        ],
                    ),
                    html.Hr(),
                    html.P(
                        "On a future page, you could add more animations or interactive elements.",
                        style={"text-align": "center", "color": "#666"},
                    ),
                ],
            ),
        ])
    ]
)


@app.callback(
    Output("live-graph", "figure"),
    [Input("graph-update", "n_intervals"), Input("toggle-anomalies", "value")],
)
def update_graph(n, toggle_values):
    global df
    try:
        while not data_queue.empty():
            data = data_queue.get()
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        if df.empty:
            return go.Figure()

        df_current = df.tail(1000)
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=df_current["time"],
                y=df_current["value"],
                mode="lines",
                name="Data Stream",
            )
        )

        if "anomalies" in toggle_values:
            anomalies = df_current[df_current["is_anomaly"] == True]
            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies["time"],
                        y=anomalies["value"],
                        mode="markers",
                        marker=dict(color="red", size=8, symbol="circle-open"),
                        name="Anomalies",
                    )
                )
                for _, anomaly_row in anomalies.iterrows():
                    fig.add_shape(
                        type="rect",
                        x0=anomaly_row["time"] - 0.5,
                        x1=anomaly_row["time"] + 0.5,
                        y0=df_current["value"].min() - 1,
                        y1=df_current["value"].max() + 1,
                        fillcolor="rgba(255, 0, 0, 0.2)",
                        line=dict(color="red", width=0),
                    )

        if "trend_line" in toggle_values:
            df_current["trend"] = df_current["value"].rolling(window=10).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_current["time"],
                    y=df_current["trend"],
                    mode="lines",
                    name="Trend Line",
                    line=dict(dash="dash", color="green"),
                )
            )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            xaxis=dict(range=[df_current["time"].min(), df_current["time"].max()]),
            yaxis=dict(range=[df_current["value"].min() - 1, df_current["value"].max() + 1]),
            template="plotly_white",
            showlegend=True,
        )
        return fig
    except Exception as e:
        print(f"Error in update_graph: {e}")
        return go.Figure()


@app.callback(
    [
        Output("download-btn", "style"),
        Output("reset-btn", "style"),
        Output("stop-btn", "style"),
        Output("start-btn", "style"),
    ],
    [
        Input("start-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
)
def toggle_buttons(start_n, stop_n, reset_n):
    ctx = callback_context
    if not ctx.triggered:
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {
                "display": "inline-block",
                **button_style,
                "background-color": "#4CAF50",
                "color": "white",
            },
        )

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-btn":
        return (
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#9C27B0",
                "color": "white",
            },
            {
                **button_style,
                "display": "none",
            },
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#f44336",
                "color": "white",
            },
            {
                **button_style,
                "display": "none",
            },
        )
    elif button_id == "stop-btn":
        return (
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#9C27B0",
                "color": "white",
            },
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#FF9800",
                "color": "white",
            },
            {
                **button_style,
                "display": "none",
            },
            {
                **button_style,
                "display": "none",
            },
        )
    elif button_id == "reset-btn":
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#4CAF50",
                "color": "white",
            },
        )
    return (
        {"display": "none"},
        {"display": "none"},
        {"display": "none"},
        {
            "display": "inline-block",
            **button_style,
            "background-color": "#4CAF50",
            "color": "white",
        },
    )


@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-btn", "n_clicks")],
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    try:
        if df.empty:
            return dcc.send_data_frame(
                pd.DataFrame(columns=["timestamp"]).to_csv,
                "anomalies.csv",
                index=False
            )
        anomalies_df = df[df["is_anomaly"] == True]
        if anomalies_df.empty:
            return dcc.send_data_frame(
                pd.DataFrame(columns=["timestamp"]).to_csv,
                "anomalies.csv",
                index=False
            )
        time_only = anomalies_df[["time"]].copy()
        time_only.rename(columns={"time": "timestamp"}, inplace=True)
        return dcc.send_data_frame(time_only.to_csv, "anomalies.csv", index=False)
    except Exception as e:
        print(f"Error in download_csv: {e}")
        return dcc.send_data_frame(
            pd.DataFrame(columns=["timestamp"]).to_csv,
            "anomalies.csv",
            index=False
        )


@app.callback(
    [
        Output("start-btn", "disabled"),
        Output("stop-btn", "disabled"),
        Output("reset-btn", "disabled"),
        Output("download-btn", "disabled"),
    ],
    [
        Input("start-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
)
def control_buttons(start_n, stop_n, reset_n):
    ctx = callback_context
    if not ctx.triggered:
        return False, True, True, True

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-btn":
        return True, False, True, True
    elif button_id == "stop-btn":
        return False, True, False, False
    elif button_id == "reset-btn":
        return False, True, True, True
    return False, True, True, True


@app.callback(
    Output("start-btn", "children"),
    [Input("start-btn", "n_clicks"), Input("stop-btn", "n_clicks")],
)
def update_start_button_label(start_clicks, stop_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return "Start Streaming"
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-btn":
        return "Streaming..."
    elif button_id == "stop-btn":
        return "Start Streaming"
    return "Start Streaming"


@app.callback(
    Output("status", "children"),
    [
        Input("start-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_status(start_n, stop_n, reset_n):
    ctx = callback_context
    if not ctx.triggered:
        return "Status: Stopped"
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-btn":
        return "Status: Streaming..."
    elif button_id == "stop-btn":
        return "Status: Stopped"
    elif button_id == "reset-btn":
        return "Status: Reset"
    return "Status: Stopped"


@app.callback(
    Output("reset-data", "children"),
    [
        Input("start-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
    [State("thread-store", "data")],
    prevent_initial_call=True,
)
def manage_streaming(start_n, stop_n, reset_n, store_data):
    global df
    ctx = callback_context
    if not ctx.triggered:
        return ""
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-btn":
        simulator.streaming = True
        stop_event.clear()
        streaming_thread = threading.Thread(
            target=data_streaming,
            args=(simulator, detector, data_queue, stop_event),
            daemon=True,
        )
        streaming_thread.start()
        store_data["streaming_thread"] = streaming_thread
    elif button_id == "stop-btn":
        simulator.streaming = False
        stop_event.set()
        streaming_thread = store_data.get("streaming_thread")
        if streaming_thread is not None and streaming_thread.is_alive():
            streaming_thread.join()
    elif button_id == "reset-btn":
        simulator.streaming = False
        stop_event.set()
        streaming_thread = store_data.get("streaming_thread")
        if streaming_thread is not None and streaming_thread.is_alive():
            streaming_thread.join()
        df = pd.DataFrame(columns=["time", "value", "is_anomaly", "anomaly_score"])
        detector.anomalies.clear()
    return ""


@app.callback(
    Output("beep-store", "data"),
    [Input("graph-update", "n_intervals")],
    [State("beep-store", "data")],
)
def beep_times(n, beep_data):
    if n >= 80 and not beep_data["beep1_done"]:
        try:
            winsound.Beep(2500, 5000)
        except:
            pass
        beep_data["beep1_done"] = True

    if n >= 260 and not beep_data["beep2_done"]:
        try:
            winsound.Beep(2500, 5000)
        except:
            pass
        engine = pyttsx3.init()
        engine.say("machine stopped")
        engine.runAndWait()
        beep_data["beep2_done"] = True
    return beep_data


@app.callback(
    Output("feature-validation-msg", "children"),
    [Input("submit-feature-btn", "n_clicks")],
    [
        State("feature-dropdown", "value"),
        State("feature-input-value", "value"),
    ],
)
def validate_feature(n_clicks, selected_feature, input_value):
    if n_clicks == 0:
        return ""
    if not selected_feature:
        return "Please select a feature first."
    if input_value is None:
        return "Please enter a numeric value."

    low, high = FEATURE_RANGES[selected_feature]
    if low <= input_value <= high:
        return f"Value {input_value} is within acceptable range ({low}-{high})."
    else:
        return f"Value {input_value} is out of acceptable range ({low}-{high})."


# Button to “respond” to anomalies. Gains 10 points each click.
@app.callback(
    Output("score-display", "children"),
    [Input("respond-anomaly-btn", "n_clicks")],
    [State("score-store", "data")],
)
def update_score(n_clicks, score_data):
    score = score_data["score"]
    if n_clicks > 0:
        score += 10
    score_data["score"] = score
    return str(score)


@app.callback(
    Output("score-store", "data"),
    [Input("respond-anomaly-btn", "n_clicks")],
    [State("score-store", "data")],
)
def store_score(n_clicks, score_data):
    # Update the stored score
    score = score_data["score"]
    if n_clicks > 0:
        score += 10
    score_data["score"] = score
    return score_data


def main():
    app.run_server(debug=False)


if __name__ == "__main__":
    main()
