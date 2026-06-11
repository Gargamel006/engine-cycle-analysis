from copy import deepcopy
import importlib

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import turfan

turfan = importlib.reload(turfan)
from turfan import (
    BASE_INPUTS,
    TURBOFAN_DEFAULTS,
    TURBOJET_DEFAULTS,
    calculate_engine,
    standard_atmosphere,
)


st.set_page_config(page_title="航空发动机性能分析", page_icon="✈️", layout="wide")


ENGINE_OPTIONS = {
    "turbofan": {
        "name": "涡扇发动机",
        "defaults": TURBOFAN_DEFAULTS,
        "x_params": ["H", "M0", "alpha", "pi_f", "pi_c", "Tt4"],
    },
    "turbojet": {
        "name": "涡喷发动机",
        "defaults": TURBOJET_DEFAULTS,
        "x_params": ["H", "M0", "pi_c", "Tt4"],
    },
}


PARAM_CONFIG = {
    "H": {"label": "巡航高度 H", "unit": "m", "min": 0.0, "max": 15000.0, "step": 10.0},
    "M0": {"label": "巡航马赫数 M₀", "unit": "", "min": 0.1, "max": 1.5, "step": 0.01},
    "alpha": {"label": "涵道比 α", "unit": "", "min": 0.0, "max": 18.0, "step": 0.1},
    "pi_f": {"label": "风扇压缩比 π𝒻", "unit": "", "min": 1.05, "max": 2.4, "step": 0.01},
    "pi_c": {"label": "压气机压缩比 π꜀", "unit": "", "min": 2.0, "max": 60.0, "step": 0.1},
    "Tt4": {"label": "涡轮前总温 Tₜ₄", "unit": "K", "min": 1100.0, "max": 2300.0, "step": 10.0},
}


METRIC_CONFIG = {
    "F_m0": {"label": "单位推力 F/m₀", "unit": "N·s/kg", "color": "#2563eb"},
    "SFC": {"label": "单位燃油消耗率 SFC", "unit": "mg/(N·s)", "color": "#dc2626"},
    "eta_th": {"label": "热效率 ηₜₕ", "unit": "%", "color": "#16a34a"},
    "eta_p": {"label": "推进效率 ηₚ", "unit": "%", "color": "#9333ea"},
    "eta_o": {"label": "总效率 ηₒ", "unit": "%", "color": "#ea580c"},
    "f": {"label": "燃油空气比 f", "unit": "", "color": "#0891b2"},
    "u9": {"label": "喷管出口速度 u₉", "unit": "m/s", "color": "#475569"},
    "u0": {"label": "飞行速度 u₀", "unit": "m/s", "color": "#0f766e"},
}


DUAL_AXIS_PRESETS = {
    "经典对决：动力性 vs 经济性": {
        "left": ["F_m0"],
        "right": ["SFC"],
    },
    "机理剖析：油耗 vs 效率": {
        "left": ["SFC"],
        "right": ["eta_th", "eta_p", "eta_o"],
    },
    "热力循环监控：速度 vs 燃油空气比": {
        "left": ["u9", "u0"],
        "right": ["f"],
    },
}


EFFICIENCY_KEYS = {"eta_th", "eta_p", "eta_o"}


def axis_label(key, config):
    unit = config[key]["unit"]
    return f"{config[key]['label']} [{unit}]" if unit else config[key]["label"]


def metric_label(key):
    unit = METRIC_CONFIG[key]["unit"]
    return f"{METRIC_CONFIG[key]['label']} [{unit}]" if unit else METRIC_CONFIG[key]["label"]


def axis_title_from_metrics(metric_keys, fallback):
    if metric_keys and all(metric in EFFICIENCY_KEYS for metric in metric_keys):
        return "效率 [%]"
    if len(metric_keys) == 1:
        return metric_label(metric_keys[0])
    if metric_keys:
        return fallback
    return ""


def unique_metrics(*metric_groups):
    metrics = []
    for group in metric_groups:
        for metric in group:
            if metric not in metrics:
                metrics.append(metric)
    return metrics


def format_value_with_unit(value, unit):
    formatted = f"{value:.6g}"
    return f"{formatted} {unit}" if unit else formatted


def finite_min_max(values):
    finite_values = [value for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return None, None
    return float(min(finite_values)), float(max(finite_values))


def calc_padded_range(metric_keys, y_data, bottom_padding=0.05, top_padding=0.32):
    values = []
    for metric in metric_keys:
        values.extend([value for value in y_data[metric] if value is not None and np.isfinite(value)])
    if not values:
        return None

    actual_min = min(values)
    actual_max = max(values)
    data_range = actual_max - actual_min if actual_max != actual_min else 1.0
    return [
        actual_min - data_range * bottom_padding,
        actual_max + data_range * top_padding,
    ]


def build_series(engine_type, x_key, x_values, base_inputs, metric_keys):
    data = {metric: [] for metric in metric_keys}
    invalid_count = 0

    for x_value in x_values:
        inputs = deepcopy(base_inputs)
        inputs[x_key] = float(x_value)
        try:
            result = calculate_engine(engine_type, inputs)
        except ValueError:
            invalid_count += 1
            for metric in metric_keys:
                data[metric].append(None)
            continue

        for metric in metric_keys:
            data[metric].append(result.get(metric))

    return data, invalid_count


def atmosphere_state_from_result(result, inputs):
    atmosphere = standard_atmosphere(inputs["H"], gamma=inputs["gamma"], R=inputs["R"])
    return {
        "T0": result.get("T0", atmosphere["T0"]),
        "P0": result.get("P0", atmosphere["P0"]),
        "rho0": result.get("rho0", atmosphere["rho0"]),
        "a0": result.get("a0", atmosphere["a0"]),
    }


def slider_with_number(label, minimum, maximum, default, step, key):
    slider_col, input_col = st.columns([0.68, 0.32], gap="small")
    value_key = f"{key}_value"
    if value_key not in st.session_state:
        st.session_state[value_key] = float(default)

    def sync_from_slider():
        st.session_state[value_key] = st.session_state[f"{key}_slider"]

    def sync_from_input():
        st.session_state[value_key] = st.session_state[f"{key}_input"]

    with slider_col:
        st.slider(
            label,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(st.session_state[value_key]),
            step=float(step),
            format="%g",
            key=f"{key}_slider",
            on_change=sync_from_slider,
        )
    with input_col:
        st.number_input(
            "输入值",
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(st.session_state[value_key]),
            step=float(step),
            format="%g",
            key=f"{key}_input",
            on_change=sync_from_input,
            label_visibility="collapsed",
        )

    return float(st.session_state[value_key])


def range_slider_with_numbers(label, minimum, maximum, default_range, step, key):
    range_key = f"{key}_range"
    if range_key not in st.session_state:
        st.session_state[range_key] = (float(default_range[0]), float(default_range[1]))

    start, end = st.session_state[range_key]
    start = max(float(minimum), min(float(start), float(maximum)))
    end = max(float(minimum), min(float(end), float(maximum)))
    if start > end:
        start, end = end, start
    st.session_state[range_key] = (start, end)

    def sync_from_range_slider():
        st.session_state[range_key] = tuple(st.session_state[f"{key}_range_slider"])

    def sync_from_range_inputs():
        input_start = st.session_state[f"{key}_start_input"]
        input_end = st.session_state[f"{key}_end_input"]
        if input_start > input_end:
            input_start, input_end = input_end, input_start
        st.session_state[range_key] = (float(input_start), float(input_end))

    st.slider(
        label,
        min_value=float(minimum),
        max_value=float(maximum),
        value=st.session_state[range_key],
        step=float(step),
        format="%g",
        key=f"{key}_range_slider",
        on_change=sync_from_range_slider,
    )

    start_col, end_col = st.columns(2, gap="small")
    with start_col:
        st.number_input(
            "起点",
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(st.session_state[range_key][0]),
            step=float(step),
            format="%g",
            key=f"{key}_start_input",
            on_change=sync_from_range_inputs,
        )
    with end_col:
        st.number_input(
            "终点",
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(st.session_state[range_key][1]),
            step=float(step),
            format="%g",
            key=f"{key}_end_input",
            on_change=sync_from_range_inputs,
        )

    return st.session_state[range_key]


st.title("航空发动机理想循环交互分析")

engine_type = st.segmented_control(
    "发动机类型",
    options=list(ENGINE_OPTIONS.keys()),
    format_func=lambda key: ENGINE_OPTIONS[key]["name"],
    default="turbofan",
)

engine = ENGINE_OPTIONS[engine_type]
defaults = engine["defaults"]

left, right = st.columns([0.72, 0.28], gap="large")

with right:
    st.subheader("控制参数")

    x_key = st.selectbox(
        "X 轴变量",
        options=engine["x_params"],
        format_func=lambda key: axis_label(key, PARAM_CONFIG),
    )

    preset_name = st.selectbox(
        "双 Y 轴推荐组合",
        options=list(DUAL_AXIS_PRESETS.keys()),
        index=0,
    )
    preset = DUAL_AXIS_PRESETS[preset_name]

    left_y_keys = st.multiselect(
        "左 Y 轴参数",
        options=list(METRIC_CONFIG.keys()),
        default=preset["left"],
        format_func=metric_label,
        key=f"{engine_type}_left_y_{preset_name}",
    )
    right_y_keys = st.multiselect(
        "右 Y 轴参数",
        options=list(METRIC_CONFIG.keys()),
        default=preset["right"],
        format_func=metric_label,
        key=f"{engine_type}_right_y_{preset_name}",
    )
    selected_y_keys = unique_metrics(left_y_keys, right_y_keys)
    if not selected_y_keys:
        st.info("至少在左 Y 轴或右 Y 轴选择一个性能参数后即可绘图。")

    x_conf = PARAM_CONFIG[x_key]
    x_range = range_slider_with_numbers(
        "X 轴计算范围",
        x_conf["min"],
        x_conf["max"],
        (x_conf["min"], x_conf["max"]),
        x_conf["step"],
        f"{engine_type}_{x_key}_x_range",
    )
    point_count = st.slider("曲线采样点数", 30, 300, 120, 10)

    st.divider()
    st.caption("基准工况")

    current_inputs = dict(BASE_INPUTS)
    for key in engine["x_params"]:
        if key == x_key:
            continue
        conf = PARAM_CONFIG[key]
        current_inputs[key] = slider_with_number(
            axis_label(key, PARAM_CONFIG),
            conf["min"],
            conf["max"],
            float(defaults[key]),
            conf["step"],
            f"{engine_type}_{x_key}_{key}",
        )

with left:
    current_inputs[x_key] = float(defaults[x_key])
    try:
        design_result = calculate_engine(engine_type, current_inputs)
    except ValueError as exc:
        design_result = None
        st.warning(f"当前基准工况不可计算：{exc}")

    if design_result:
        x_default_text = format_value_with_unit(defaults[x_key], PARAM_CONFIG[x_key]["unit"])
        st.caption(f"下方性能指标对应的默认横坐标：{axis_label(x_key, PARAM_CONFIG)} = {x_default_text}")
        cards = st.columns(5)
        for column, metric in zip(cards, ["F_m0", "SFC", "eta_th", "eta_p", "eta_o"]):
            with column:
                st.metric(metric_label(metric), f"{design_result[metric]:.4g}")

        with st.expander("当前设计点大气状态", expanded=False):
            atmosphere_state = atmosphere_state_from_result(design_result, current_inputs)
            atmosphere_cols = st.columns(4)
            atmosphere_metrics = [
                ("大气静温 T₀", atmosphere_state["T0"], "K"),
                ("大气静压 P₀", atmosphere_state["P0"], "Pa"),
                ("大气密度 ρ₀", atmosphere_state["rho0"], "kg/m³"),
                ("声速 a₀", atmosphere_state["a0"], "m/s"),

            ]
            for column, (label, value, unit) in zip(atmosphere_cols, atmosphere_metrics):
                with column:
                    st.metric(f"{label} [{unit}]", f"{value:.4g}")

    if selected_y_keys:
        x_values = np.linspace(x_range[0], x_range[1], point_count)
        y_data, invalid_count = build_series(engine_type, x_key, x_values, current_inputs, selected_y_keys)
        left_axis_title = axis_title_from_metrics(left_y_keys, "左轴性能参数")
        right_axis_title = axis_title_from_metrics(right_y_keys, "右轴性能参数")
        left_range = calc_padded_range(left_y_keys, y_data)
        right_range = calc_padded_range(right_y_keys, y_data)
        dual_axis_active = bool(left_y_keys and right_y_keys)

        fig = go.Figure()
        # 定义学术图表常用的线型、颜色和标记
        academic_colors = ["#000000", "#D62728", "#1F77B4", "#2CA02C", "#FF7F0E"] # 黑, 深红, 深蓝, 深绿, 暗橙
        academic_dashes = ["solid", "dash", "dot", "dashdot", "longdash"]
        academic_symbols = ["circle", "square", "triangle-up", "diamond", "x"]

        plotted_metrics = []
        for axis_name, metric_group in [("左轴", left_y_keys), ("右轴", right_y_keys)]:
            for metric in metric_group:
                if metric in plotted_metrics:
                    continue
                plotted_metrics.append(metric)
                i = len(plotted_metrics) - 1
                yaxis_name = "y2" if axis_name == "右轴" else "y"
                legend_suffix = f" ({axis_name})" if dual_axis_active else ""
                legend_name = f"{metric_label(metric)}{legend_suffix}"
                axis_hover_line = f"<br>坐标轴: {axis_name}" if dual_axis_active else ""
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_data[metric],
                        yaxis=yaxis_name,
                        mode="lines+markers",  
                        name=legend_name,
                        line=dict(
                            width=1.5,  # 线条调细，显得更严谨
                            color=academic_colors[i % len(academic_colors)],
                            dash=academic_dashes[i % len(academic_dashes)]
                        ),
                        marker=dict(
                            size=6,
                            symbol=academic_symbols[i % len(academic_symbols)],
                            color="white", 
                            line=dict(width=1.5, color=academic_colors[i % len(academic_colors)])
                        ),
                        connectgaps=False,
                        hovertemplate=(
                            f"{axis_label(x_key, PARAM_CONFIG)}: %{{x:.4g}}<br>"
                            f"{metric_label(metric)}: %{{y:.4g}}"
                            f"{axis_hover_line}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            template="simple_white",  
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(
                family="Times New Roman, SimSun, serif", # 使用罗马字体/宋体
                size=14, 
                color="black"
            ),
            height=560,
            margin={"l": 70, "r": 30, "t": 40, "b": 70},
            hovermode="x unified",
            # 将图例放到图表内部右上角，并加上黑框
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.96,
                xanchor="right",
                x=0.96,
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            ),
            xaxis=dict(
                title=axis_label(x_key, PARAM_CONFIG),
                range=[x_range[0], x_range[1]],
                showgrid=False,           
                zeroline=False,          
                showline=True,            
                linewidth=1.5,            
                linecolor='black',        
                mirror=True,              
                ticks='inside',          
                tickwidth=1.5,
                tickcolor='black',
                ticklen=6,
                minor=dict(ticks='inside', ticklen=3, tickwidth=1, tickcolor='black', showgrid=False)
            ),
            yaxis=dict(
                title=left_axis_title,
                range=left_range,
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='black',
                mirror=not bool(right_y_keys),
                ticks='inside',
                tickwidth=1.5,
                tickcolor='black',
                ticklen=6,
                minor=dict(ticks='inside', ticklen=3, tickwidth=1, tickcolor='black', showgrid=False)
            ),
            yaxis2=dict(
                title=right_axis_title,
                range=right_range,
                overlaying="y",
                side="right",
                visible=bool(right_y_keys),
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='black',
                ticks='inside',
                tickwidth=1.5,
                tickcolor='black',
                ticklen=6,
                minor=dict(ticks='inside', ticklen=3, tickwidth=1, tickcolor='black', showgrid=False)
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
            theme=None,  
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            },
        )

        y_min_values = {metric: finite_min_max(y_data[metric])[0] for metric in selected_y_keys}
        y_max_values = {metric: finite_min_max(y_data[metric])[1] for metric in selected_y_keys}
        table_rows = []
        for axis_name, metric_group in [("左 Y 轴", left_y_keys), ("右 Y 轴", right_y_keys)]:
            for metric in metric_group:
                if any(row["性能参数"] == metric_label(metric) for row in table_rows):
                    continue
                table_rows.append(
                    {
                        "坐标轴": axis_name,
                        "性能参数": metric_label(metric),
                        "最小值": y_min_values[metric],
                        "最大值": y_max_values[metric],
                    }
                )
        st.dataframe(table_rows, width="stretch", hide_index=True)

        if invalid_count:
            st.warning(f"有 {invalid_count} 个采样点因循环不成立而断开显示。")

        st.caption("图中可用鼠标框选放大、滚轮缩放、拖拽平移；双击图表可恢复自动视图。")
