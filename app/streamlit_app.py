"""Streamlit UI -- three functional areas required by task book section 7:

  1. 上传诊断 (upload diagnosis): upload a signal or sample from PTB-XL test,
     get classifier labels + confidence + QT/PR/QRS measurements.
  2. 可控生成 (controlled generation): choose route / diagnostic label / target HR,
     generate the waveform, show the classifier's prediction + interval measures.
  3. 多路线对比 (multi-route comparison): same condition side-by-side for
     A/B/C/D + real reference + FFT spectrum comparison, with intervals.

Run:
  powershell -File tools\run.ps1 -Module streamlit run app\streamlit_app.py
(or from the repo root:  python -m streamlit run app/streamlit_app.py)

The app degrades gracefully: routes whose checkpoints are not trained yet are
shown as unavailable with the reason, and the rest of the UI keeps working.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config
from app import inference

cfg = load_config()
CLASSES = cfg["superclasses"]
LEADS = cfg["lead_names"]

st.set_page_config(page_title="ECG 生成与诊断系统", layout="wide")
st.title("心电生成与诊断系统 — PTB-XL")
st.caption("路线 A 机理模型 / B 标准扩散 / C 规律注入扩散 / D GAN · 5 类诊断多标签分类器 · 统一评估")


# ------------------------------------------------------------------ helpers ---
@st.cache_resource(show_spinner=False)
def _classifier():
    from models.classifier import load_classifier
    return load_classifier()[0]


def _parse_uploaded(file) -> np.ndarray | None:
    name = file.name.lower()
    try:
        if name.endswith(".npy"):
            arr = np.load(file)
        elif name.endswith((".csv", ".txt")):
            arr = np.loadtxt(file, delimiter=",")
        else:
            st.error("仅支持 .npy / .csv / .txt")
            return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"文件读取失败：{exc}")
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.tile(arr[:, None], (1, 12))
    if arr.shape == (12, 1000):
        sig = arr
    elif arr.shape == (1000, 12):
        sig = arr.T
    elif arr.shape[1] == 12:
        sig = arr.T
    elif arr.shape[0] == 12:
        sig = arr
    else:
        st.error(f"无法识别的形状 {arr.shape}（需要 12×1000 或 1000×12）")
        return None
    if sig.shape[-1] != cfg["n_samples"]:
        st.warning(f"长度 {sig.shape[-1]} != {cfg['n_samples']}，将截断/补零到 1000")
        fixed = np.zeros((12, cfg["n_samples"]), dtype=np.float32)
        n = min(sig.shape[-1], cfg["n_samples"])
        fixed[:, :n] = sig[:, :n]
        sig = fixed
    mu = sig.mean(1, keepdims=True)
    sd = sig.std(1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((sig - mu) / sd).astype(np.float32)


def _show_diagnosis(sig: np.ndarray, key_prefix: str):
    probs = inference.classify(sig[None])[0]
    fig = inference.plot_12lead(sig, title="12 导联心电（预处理后 z-score）")
    st.pyplot(fig, width="stretch")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("分类器输出（5 类多标签）")
        import pandas as pd
        df = pd.DataFrame({"诊断超类": CLASSES, "概率": probs.round(4)})
        df["判定"] = np.where(probs >= 0.5, "阳性", "-")
        st.dataframe(df, width="stretch", hide_index=True)
    with c2:
        st.subheader("关键参数（Lead II）")
        rows = inference.measure(sig[None])
        if rows:
            r = rows[0]

            def _f(key, nd=1):
                v = r.get(key, np.nan)
                return f"{v:.{nd}f}" if np.isfinite(v) else "提取失败"

            st.dataframe(pd.DataFrame([{
                "QT (ms)": _f("QT_ms"),
                "PR (ms)": _f("PR_ms"),
                "QRS (ms)": _f("QRS_ms"),
                "R-R CV": _f("rr_cv", 4),
            }]), width="stretch", hide_index=True)
            if r.get("failure"):
                st.caption(f"delineate 失败原因：{r['failure']}（已记录，不隐藏）")
        else:
            st.warning("未能提取到有效心动周期（记录为失败样本）")


def _cond_inputs(key_prefix: str):
    col1, col2 = st.columns(2)
    with col1:
        labels = np.zeros(len(CLASSES), dtype=np.float32)
        chosen = st.multiselect("诊断超类（可多选，留空=无条件）", CLASSES, key=f"{key_prefix}_labels")
        for c in chosen:
            labels[CLASSES.index(c)] = 1.0
    with col2:
        hr = st.slider("目标心率 (bpm)", 40, 180, 70, key=f"{key_prefix}_hr")
    return labels, float(hr)


# ------------------------------------------------------------------- tab 1 ---
tab1, tab2, tab3 = st.tabs(["① 上传诊断", "② 可控生成", "③ 多路线对比"])

with tab1:
    st.header("上传信号或从测试集抽样 → 分类 + 参数测量")
    mode = st.radio("信号来源", ["从 PTB-XL 测试集抽样", "上传文件 (.npy/.csv/.txt)"],
                    horizontal=True, key="t1_mode")
    sig = None
    if mode.startswith("从 PTB"):
        c1, c2 = st.columns([1, 2])
        with c1:
            sample_idx = st.number_input("测试集样本序号", 0, 5000, 0, step=1, key="t1_idx")
            if st.button("抽取并诊断", key="t1_sample_btn"):
                sig, meta = inference.load_real_sample(int(sample_idx))
                st.session_state["t1_sig"] = sig
                st.session_state["t1_meta"] = meta
                st.rerun()
        with c2:
            if "t1_meta" in st.session_state:
                m = st.session_state["t1_meta"]
                true_lab = ", ".join(c for c, v in zip(CLASSES, m["labels"]) if v) or "无"
                st.info(f"ecg_id={m['ecg_id']} · 真实诊断超类：{true_lab} · 真实 HR≈{m['hr']:.1f} bpm")
    else:
        file = st.file_uploader("选择文件", type=["npy", "csv", "txt"], key="t1_file")
        if file is not None:
            sig = _parse_uploaded(file)
    if sig is None and "t1_sig" in st.session_state:
        sig = st.session_state["t1_sig"]
    if sig is not None:
        _show_diagnosis(sig, "t1")
    st.caption("提示：上传信号按逐导联 z-score 归一化后送入分类器；测量使用 NeuroKit2 ecg_delineate，失败样本会明确标注。")


# ------------------------------------------------------------------- tab 2 ---
with tab2:
    st.header("选择路线 + 条件 → 生成波形 → 分类器预测 + 参数")
    routes = inference.available_routes()
    available = [r for r, v in routes.items() if v["available"]]
    if not available:
        st.error("没有任何可用路线（checkpoint 尚未训练）")
    else:
        col1, col2 = st.columns(2)
        with col1:
            route = st.selectbox("生成路线", available, key="t2_route")
            st.caption(f"条件支持：{routes[route]['cond']}" +
                       (f" · {routes[route]['note']}" if routes[route]["note"] else ""))
        with col2:
            st.write("")
        labels, hr = _cond_inputs("t2")
        if st.button("生成波形", key="t2_gen_btn"):
            with st.spinner(f"路线 {route} 生成中..."):
                try:
                    sig_g = inference.generate(route, labels if labels.any() else None, hr)[0]
                    st.session_state["t2_sig"] = sig_g
                except Exception as exc:  # noqa: BLE001
                    st.error(f"生成失败：{exc}")
        if "t2_sig" in st.session_state:
            _show_diagnosis(st.session_state["t2_sig"], "t2")
            st.caption("注意：路线 A 不使用诊断标签；B_uncond / D 为无条件模型，标签条件对其无效。")


# ------------------------------------------------------------------- tab 3 ---
with tab3:
    st.header("同一条件下的多路线对比 + 真实参考 + FFT 频谱")
    routes = inference.available_routes()
    available = [r for r, v in routes.items() if v["available"]]
    default_routes = [r for r in inference.MATURE_ROUTES if r in available]
    chosen_routes = st.multiselect("对比路线", available, default=default_routes, key="t3_routes")
    labels, hr = _cond_inputs("t3")
    use_real = st.checkbox("加入真实参考（PTB-XL 测试集抽样）", value=True, key="t3_real")
    if st.button("生成并对比", key="t3_btn"):
        sigs = {}
        with st.spinner("生成中..."):
            for r in chosen_routes:
                try:
                    sigs[r] = inference.generate(r, labels if labels.any() else None, hr)[0]
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"{r} 生成失败：{exc}")
            if use_real:
                real_sig, real_meta = inference.load_real_sample(seed=7)
                sigs["REAL"] = real_sig
                st.session_state["t3_real_meta"] = real_meta
        st.session_state["t3_sigs"] = sigs
    if "t3_sigs" in st.session_state and st.session_state["t3_sigs"]:
        sigs = st.session_state["t3_sigs"]
        st.subheader("Lead II 波形对比")
        st.pyplot(inference.plot_lead_ii_compare(sigs), width="stretch")
        st.subheader("FFT 频谱对比（Lead II）")
        st.pyplot(inference.plot_fft(sigs), width="stretch")
        st.subheader("各路线参数与分类器预测")
        import pandas as pd
        rows = []
        for r, s in sigs.items():
            try:
                probs = inference.classify(s[None])[0]
                mrows = inference.measure(s[None])
                m = mrows[0] if mrows else {}
                rows.append({
                    "路线": r,
                    "QT (ms)": f"{m.get('QT_ms', np.nan):.1f}" if np.isfinite(m.get("QT_ms", np.nan)) else "失败",
                    "PR (ms)": f"{m.get('PR_ms', np.nan):.1f}" if np.isfinite(m.get("PR_ms", np.nan)) else "失败",
                    "QRS (ms)": f"{m.get('QRS_ms', np.nan):.1f}" if np.isfinite(m.get("QRS_ms", np.nan)) else "失败",
                    "预测诊断": " / ".join(c for c, p in zip(CLASSES, probs) if p >= 0.5) or "（无）",
                    "最高概率": f"{CLASSES[int(np.argmax(probs))]} ({probs.max():.3f})",
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({"路线": r, "QT (ms)": "-", "PR (ms)": "-", "QRS (ms)": "-",
                             "预测诊断": f"失败：{exc}", "最高概率": "-"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if "t3_real_meta" in st.session_state:
            m = st.session_state["t3_real_meta"]
            st.caption(f"真实参考：PTB-XL 测试集 ecg_id={m['ecg_id']}，"
                       f"真实诊断超类={[c for c, v in zip(CLASSES, m['labels']) if v]}，HR≈{m['hr']:.1f}")
    else:
        st.info("设置条件后点击“生成并对比”。四种路线条件能力不同：A 仅心率，B_uncond/D 无条件，B_cond 仅标签，C 标签+心率。")

st.divider()
st.caption("项目根目录：ECG/ · 结果：results/metrics.csv · 生成信号：results/generated/*.npy · 训练日志：results/logs/")
