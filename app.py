"""
HW4 · Modelling and Forecasting Realized Variance
Financial Econometrics – ESSEC Centrale 2026
G. O'Brien · C. Desurmont · M. Lafon
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HW4 · Realized Variance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
TICKERS = [
    'MMM','AXP','AMGN','AAPL','BA','CAT','CVX','CSCO','KO','DIS',
    'DOW','GS','HD','HON','IBM','INTC','JNJ','JPM','MCD','MRK',
    'MSFT','NKE','PG','CRM','TRV','UNH','VZ','V','WBA','WMT',
]
RES  = Path('HW4_results')
CMAP = plt.cm.tab20(np.linspace(0, 1, 30))

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #94a3b8 !important; }

.q-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0e7490 100%);
    color: white !important;
    padding: 12px 20px;
    border-radius: 10px;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    letter-spacing: 0.3px;
}
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 10px 15px;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    margin: 0.6rem 0;
}
.warn-box {
    background: #fff7ed;
    border-left: 4px solid #f97316;
    padding: 10px 15px;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    margin: 0.6rem 0;
}
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 8px 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def qheader(text):
    st.markdown(f'<div class="q-header">{text}</div>', unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box">ℹ️&nbsp; {text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">⚠️&nbsp; {text}</div>', unsafe_allow_html=True)

def missing(fname):
    warn(
        f"<code>{fname}</code> not found. Add the export cell to your notebook "
        f"(see the <b>Overview</b> section for the full export snippet)."
    )

def hbar(series, title, xlabel, highlight_min=True):
    """Horizontal bar chart, optionally highlighting the minimum bar in green."""
    s = series.sort_values()
    colours = ['#16a34a' if (highlight_min and i == 0) else '#94a3b8'
               for i in range(len(s))]
    fig, ax = plt.subplots(figsize=(8, max(2.5, len(s) * 0.45)))
    s.plot(kind='barh', ax=ax, color=colours, edgecolor='white')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25, axis='x')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def grouped_bar(df_, cols, title, ylabel):
    """Grouped bar chart (stocks on x-axis, models as groups)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    x  = np.arange(len(df_))
    w  = 0.75 / max(len(cols), 1)
    palette = ['#2563eb','#16a34a','#dc2626','#7c3aed','#d97706','#0891b2']
    for i, col in enumerate(cols):
        ax.bar(x + i*w, df_[col], width=w, label=col,
               color=palette[i % len(palette)], alpha=0.85, edgecolor='white')
    ax.set_xticks(x + w*(len(cols)-1)/2)
    ax.set_xticklabels(df_.index, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(fname, parse_dates=False):
    p = RES / fname
    if not p.exists():
        return None
    kw = {'index_col': 0}
    if parse_dates:
        kw['parse_dates'] = True
    try:
        return pd.read_csv(p, **kw)
    except Exception:
        return None


@st.cache_data
def load_all():
    parts = {
        'HAR': load_csv('har_forecast.csv'),
        'AR1': load_csv('ar1_forecast.csv'),
        'RW':  load_csv('rw_forecast.csv'),
        'EN':  load_csv('ml_fs1_ElasticNet.csv'),
        'RF':  load_csv('ml_fs1_RF.csv'),
        'GBM': load_csv('ml_fs1_GBM.csv'),
    }
    combined = [df.add_suffix(f'_{k}') for k, df in parts.items() if df is not None]
    forecast  = pd.concat(combined, axis=1) if combined else None
    return {
        'lnRV':    load_csv('lnRV.csv',           parse_dates=True),
        'RV':      load_csv('RV.csv',              parse_dates=True),
        'adf':     load_csv('adf_results.csv'),
        'har':     load_csv('har_params.csv'),
        'forecast': forecast,
        'granger': load_csv('granger_results.csv'),
        'gc_fore': load_csv('granger_forecast.csv'),
        **{f'_{k}_f': v for k, v in parts.items()},
    }


D = load_all()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 HW4 Dashboard")
    st.markdown("**Modelling & Forecasting**  \nRealized Variance · DJI Stocks")
    st.divider()

    section = st.radio("", [
        "🏠  Overview",
        "📈  Log Realized Variance",
        "⚙️  HAR Model",
        "🎯  Linear Forecasts",
        "🤖  ML Models",
        "🔗  Granger Causality",
    ], label_visibility="collapsed")

    st.divider()
    ticker = st.selectbox("Selected stock", TICKERS, index=TICKERS.index("AAPL"))

    status = {
        'lnRV.csv':              D['lnRV'],
        'RV.csv':                D['RV'],
        'adf_results.csv':       D['adf'],
        'har_params.csv':        D['har'],
        'har_forecast.csv':      D['_HAR_f'],
        'ar1_forecast.csv':      D['_AR1_f'],
        'rw_forecast.csv':       D['_RW_f'],
        'ml_fs1_ElasticNet.csv': D['_EN_f'],
        'ml_fs1_RF.csv':         D['_RF_f'],
        'ml_fs1_GBM.csv':        D['_GBM_f'],
        'granger_results.csv':   D['granger'],
        'granger_forecast.csv':  D['gc_fore'],
    }
    with st.expander("Data status"):
        for fname, df in status.items():
            st.markdown(f"{'🟢' if df is not None else '🔴'} `{fname}`")

    st.divider()
    st.caption("G. O'Brien · C. Desurmont · M. Lafon  \n*ESSEC Centrale · 2026*")


# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in section:
    st.title("📊 HW4 · Realized Variance Dashboard")
    st.markdown(
        "**Financial Econometrics · ESSEC Centrale · 2026**  \n"
        "G. O'Brien &nbsp;·&nbsp; C. Desurmont &nbsp;·&nbsp; M. Lafon"
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks", "30 DJI")
    c2.metric("Trading Days", "5 087")
    c3.metric("Period", "2003–2022")
    c4.metric("Intraday Freq.", "5 min")

    st.divider()
    st.markdown("""
### About this app
This dashboard reproduces every step of HW4: *modelling and forecasting realized variance
for the 30 Dow Jones Index stocks*, organised by analytical theme.

| Section | Content |
|---------|---------|
| **Log Realized Variance** | lnRV time series, moments, ACF, stationarity (ADF) |
| **HAR Model** | Corsi (2009) estimation, coefficient heatmap, 3D scatter |
| **Linear Forecasts** | OOS: HAR vs AR(1) vs Random Walk — RMSE and MFE |
| **ML Models** | ElasticNet, Random Forest, Gradient Boosting vs HAR baseline |
| **Granger Causality** | Cross-stock causality tests + HAR augmentation |

Use the **sidebar** to navigate. The **stock dropdown** personalises single-stock views throughout.
""")

    st.divider()
    st.markdown("### Notebook export snippet")
    st.code("""
import os
os.makedirs('HW4_results', exist_ok=True)

RV.to_csv('HW4_results/RV.csv')
lnRV.to_csv('HW4_results/lnRV.csv')
adf.to_csv('HW4_results/adf_results.csv')
granger_df.to_csv('HW4_results/granger_results.csv')
pd.DataFrame(rows).to_csv('HW4_results/granger_forecast.csv', index=False)

# Already auto-loaded:
# har_params.csv, har_forecast.csv, ar1_forecast.csv, rw_forecast.csv,
# ml_fs1_ElasticNet.csv, ml_fs1_RF.csv, ml_fs1_GBM.csv
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
#  LOG REALIZED VARIANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Log Realized" in section:
    qheader("Log Realized Variance (lnRV)")
    st.markdown(
        r"The log transformation $y_t = \ln(RV_t)$ reduces skewness and brings the "
        r"distribution closer to Gaussian, enabling valid OLS inference in the HAR model."
    )

    tab_all, tab_single, tab_stats, tab_adf = st.tabs([
        "All 30 Stocks", f"Single Stock ({ticker})", "Statistics", "Stationarity (ADF)"
    ])

    # ── All 30 Stocks: time series + average ACF ───────────────────────────────
    with tab_all:
        if D['lnRV'] is None:
            missing('lnRV.csv')
        else:
            lnRV = D['lnRV']
            fig, ax = plt.subplots(figsize=(14, 4))
            for i, t in enumerate(TICKERS):
                if t in lnRV.columns:
                    ax.plot(lnRV.index, lnRV[t], lw=0.3, alpha=0.45, color=CMAP[i], label=t)
            ax.set_title("Daily log(RV) — All 30 DJI Stocks", fontsize=12)
            ax.set_ylabel("lnRV")
            ax.legend(fontsize=5.5, ncol=6, loc='lower right', framealpha=0.7)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            info("GFC (2008–09) and COVID (2020) spikes are clearly visible across all stocks.")

            st.divider()
            st.markdown("#### Average Sample ACF — All 30 Stocks")
            n_lags_all = st.slider("Number of lags", 10, 60, 40, key="acf_all")
            acf_mat = np.array([
                acf(lnRV[t].dropna(), nlags=n_lags_all, fft=True)[1:]
                for t in TICKERS if t in lnRV.columns
            ])
            mean_acf = acf_mat.mean(axis=0)
            fig2, ax2 = plt.subplots(figsize=(13, 3))
            ax2.bar(range(1, n_lags_all + 1), mean_acf, color='steelblue', alpha=0.8, width=0.7)
            ax2.axhline(0, color='black', lw=0.8)
            ax2.set_title(f"Average Sample ACF of lnRV — All 30 Stocks ({n_lags_all} lags)", fontsize=12)
            ax2.set_xlabel("Lag (days)")
            ax2.set_ylabel("ACF")
            ax2.grid(True, alpha=0.25, axis='y')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            info(
                "Autocorrelations decay slowly and remain large (≈ 0.4–0.5 at lag 40). "
                "This long memory is the core motivation for the HAR model's weekly and monthly components."
            )

    # ── Single Stock: time series + ACF with lag slider ────────────────────────
    with tab_single:
        if D['lnRV'] is None:
            missing('lnRV.csv')
        elif ticker not in D['lnRV'].columns:
            warn(f"{ticker} not found in lnRV.csv.")
        else:
            lnRV = D['lnRV']
            s = lnRV[ticker].dropna()

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(s.index, s.values, lw=0.7, color='#2563eb')
            ax.fill_between(s.index, s.values, alpha=0.15, color='#2563eb')
            ax.set_title(f"Daily log(RV) — {ticker}", fontsize=12)
            ax.set_ylabel("lnRV")
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.divider()
            st.markdown(f"#### Sample ACF — {ticker}")
            n_lags_s = st.slider("Number of lags", 10, 60, 40, key="acf_single")
            fig3, ax3 = plt.subplots(figsize=(13, 4))
            plot_acf(s, lags=n_lags_s, ax=ax3, zero=False, alpha=0.05)
            ax3.set_title(f"Sample ACF of lnRV — {ticker} ({n_lags_s} lags)", fontsize=12)
            ax3.set_xlabel("Lag (days)")
            ax3.grid(True, alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    # ── Statistics: 4 moments, all vs single stock ─────────────────────────────
    with tab_stats:
        if D['lnRV'] is None:
            missing('lnRV.csv')
        else:
            lnRV = D['lnRV']
            stats_ln = pd.DataFrame({
                'Mean':            lnRV.mean(),
                'Variance':        lnRV.var(),
                'Skewness':        lnRV.skew(),
                'Excess Kurtosis': lnRV.kurtosis(),
            }).round(4)

            view_stats = st.radio(
                "View", ["All 30 stocks", f"{ticker} only"],
                horizontal=True, key="stats_view"
            )

            if "All" in view_stats:
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown("#### First Four Moments — All Stocks")
                    st.dataframe(
                        stats_ln.style
                        .format(precision=4)
                        .background_gradient(subset=['Skewness', 'Excess Kurtosis'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                with c2:
                    st.markdown("#### Cross-Sectional Summary")
                    st.metric("Avg Excess Kurtosis", f"{stats_ln['Excess Kurtosis'].mean():.2f}")
                    st.metric("Max Excess Kurtosis", f"{stats_ln['Excess Kurtosis'].max():.2f}",
                              delta=stats_ln['Excess Kurtosis'].idxmax())
                    st.metric("Most right-skewed",   stats_ln['Skewness'].idxmax(),
                              delta=f"{stats_ln['Skewness'].max():.2f}")
                    info(
                        "After log-transform: skewness drops from 6–25 to 0.3–1.4 and "
                        "excess kurtosis from 60–850 to 0.2–3.8. "
                        "lnRV is near-Gaussian — suitable for OLS (HAR)."
                    )

                st.divider()
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                stats_ln['Skewness'].sort_values().plot(
                    kind='barh', ax=ax1, color='#3b82f6', edgecolor='white')
                ax1.axvline(0, color='red', lw=0.8, ls='--')
                ax1.set_title('Skewness by Stock')
                ax1.grid(True, alpha=0.25, axis='x')

                stats_ln['Excess Kurtosis'].sort_values().plot(
                    kind='barh', ax=ax2, color='#f97316', edgecolor='white')
                ax2.axvline(0, color='red', lw=0.8, ls='--')
                ax2.set_title('Excess Kurtosis by Stock')
                ax2.grid(True, alpha=0.25, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            else:
                if ticker in stats_ln.index:
                    row = stats_ln.loc[ticker]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mean",             f"{row['Mean']:.4f}")
                    c2.metric("Variance",         f"{row['Variance']:.4f}")
                    c3.metric("Skewness",         f"{row['Skewness']:.4f}")
                    c4.metric("Excess Kurtosis",  f"{row['Excess Kurtosis']:.4f}")

                    st.divider()
                    st.markdown("#### vs Cross-Sectional Average")
                    c1b, c2b, c3b, c4b = st.columns(4)
                    c1b.metric("Avg Mean",       f"{stats_ln['Mean'].mean():.4f}")
                    c2b.metric("Avg Variance",   f"{stats_ln['Variance'].mean():.4f}")
                    c3b.metric("Avg Skewness",   f"{stats_ln['Skewness'].mean():.4f}")
                    c4b.metric("Avg Ex. Kurt.",  f"{stats_ln['Excess Kurtosis'].mean():.4f}")
                    info(
                        f"{ticker} — Excess Kurtosis = {row['Excess Kurtosis']:.2f} "
                        f"vs DJI average = {stats_ln['Excess Kurtosis'].mean():.2f}."
                    )

    # ── Stationarity (ADF) ─────────────────────────────────────────────────────
    with tab_adf:
        if D['adf'] is None:
            missing('adf_results.csv')
        else:
            adf = D['adf'].reset_index()
            adf.columns = [str(c).strip() for c in adf.columns]
            ticker_col = adf.columns[0]
            stat_col   = next((c for c in adf.columns if 'ADF' in c or 'stat' in c.lower()), None)
            p_col      = next((c for c in adf.columns if 'p-value' in c.lower() or 'pval' in c.lower()), None)
            cv5_col    = next((c for c in adf.columns if '5%' in c or 'CV 5' in c), None)
            rej_col    = next((c for c in adf.columns
                               if 'Reject' in c or 'reject' in c or 'Stationary' in c), None)

            n_rej = int(adf[p_col].lt(0.05).sum()) if p_col else "N/A"
            m1, m2, m3 = st.columns(3)
            m1.metric("Stocks tested",     len(adf))
            m2.metric("Stationary (p<5%)", n_rej)
            m3.metric("Non-stationary",    len(adf) - n_rej if isinstance(n_rej, int) else "N/A")

            sub_chart, sub_roll, sub_tbl = st.tabs(
                ["ADF Chart", f"Rolling Stats ({ticker})", "Full Table"]
            )

            with sub_chart:
                if stat_col and cv5_col:
                    plot_df = adf.set_index(ticker_col).sort_values(stat_col)
                    cv5_val = plot_df[cv5_col].mean()
                    fig, ax = plt.subplots(figsize=(14, 5))
                    colours = ['#16a34a' if v < cv5_val else '#dc2626'
                               for v in plot_df[stat_col]]
                    ax.barh(plot_df.index, plot_df[stat_col],
                            color=colours, edgecolor='white', linewidth=0.4)
                    ax.axvline(cv5_val, color='red', lw=1.5, ls='--',
                               label=f'Critical value 5% ({cv5_val:.2f})')
                    ax.set_xlabel("ADF Statistic")
                    ax.set_title("ADF Test Statistics — All Stocks  (green = stationary)", fontsize=11)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.25, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    info("All bars to the left of the red dashed line reject the unit root at 5%.")

            with sub_roll:
                if D['lnRV'] is not None and ticker in D['lnRV'].columns:
                    s = D['lnRV'][ticker].dropna()
                    roll_mean = s.rolling(60).mean()
                    roll_std  = s.rolling(60).std()
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
                    ax1.plot(s.index, s.values, lw=0.5, color='#94a3b8', label='lnRV')
                    ax1.plot(roll_mean.index, roll_mean.values, lw=1.5,
                             color='#2563eb', label='60-day rolling mean')
                    ax1.set_ylabel("lnRV")
                    ax1.legend(fontsize=8)
                    ax1.grid(True, alpha=0.25)
                    ax1.set_title(f"Rolling Mean and Std — {ticker}", fontsize=11)
                    ax2.plot(roll_std.index, roll_std.values, lw=1.2,
                             color='#dc2626', label='60-day rolling std')
                    ax2.set_ylabel("Rolling Std")
                    ax2.legend(fontsize=8)
                    ax2.grid(True, alpha=0.25)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    info("A stationary series has a constant mean and variance over time. "
                         "Flat rolling mean and stable rolling std confirm stationarity.")

            with sub_tbl:
                styled = adf.style.format(precision=4)
                if rej_col:
                    styled = styled.map(
                        lambda v: 'background-color:#d4edda'
                                  if v is True or str(v) == 'True'
                                  else ('background-color:#f8d7da'
                                        if v is False or str(v) == 'False' else ''),
                        subset=[rej_col]
                    )
                st.dataframe(styled, use_container_width=True)

            info("All 30 lnRV series are stationary — a necessary condition for HAR estimation.")


# ══════════════════════════════════════════════════════════════════════════════
#  HAR MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif "HAR" in section:
    qheader("HAR Model Estimation (Corsi, 2009)")
    st.markdown("**Heterogeneous Autoregression** — restricted AR(22):")
    st.latex(
        r"y_t = \beta_0 + \beta_1\,y_{t-1} "
        r"+ \beta_2\,y^{(5)}_{t-1} "
        r"+ \beta_3\,y^{(22)}_{t-1} "
        r"+ \varepsilon_t"
    )

    with st.expander("Model intuition — why daily, weekly, monthly?", expanded=True):
        st.markdown(r"""
The HAR model is grounded in the **Heterogeneous Market Hypothesis**: different participants
operate at different horizons, each reacting to volatility over a cycle matching their investment strategy.

| Component | Variable | Horizon | Who drives it |
|-----------|----------|---------|---------------|
| **β₁ · Daily** | $y_{t-1}$ | 1 day | Market makers, day traders |
| **β₂ · Weekly** | $y^{(5)}_{t-1} = \tfrac{1}{5}\sum_{k=1}^{5}y_{t-k}$ | 5 days | Institutional desks, short-term funds |
| **β₃ · Monthly** | $y^{(22)}_{t-1} = \tfrac{1}{22}\sum_{k=1}^{22}y_{t-k}$ | 22 days | Long-only funds, macro investors |

**Why lag 22 for the monthly component?**
There are approximately **22 trading days per calendar month** (252 ÷ 12 ≈ 21).
Using $y^{(22)}_{t-1}$ aligns the monthly average with the natural economic cycle
and has become the field convention since Corsi (2009).

**Which component dominates?**
Empirically across DJI stocks: **β₁ > β₂ > β₃** in absolute magnitude.
Yesterday's volatility is the single strongest one-day predictor.
However, weekly and monthly lags are essential for capturing **long memory** — without them,
forecasts revert too quickly to the unconditional mean and perform poorly.

The sum **β₁ + β₂ + β₃ ≈ 0.95** reflects near-unit-root persistence,
consistent with well-documented long memory in realized variance.
""")

    if D['har'] is None:
        missing('har_params.csv')
    else:
        har = D['har']
        beta_cols = [c for c in har.columns if 'β' in c or 'beta' in c.lower() or 'Beta' in c]
        r2_col    = next(
            (c for c in har.columns
             if ('R' in c and '²' in c) or 'r2' in c.lower() or c == 'R2'),
            None
        )

        if not beta_cols:
            st.dataframe(har.style.format(precision=4), use_container_width=True)
        else:
            b1 = next((c for c in beta_cols if 'D' in c or '1' in c), beta_cols[0])
            b2 = next((c for c in beta_cols if 'W' in c or '2' in c),
                      beta_cols[1] if len(beta_cols) > 1 else beta_cols[0])
            b3 = next((c for c in beta_cols if 'M' in c or '3' in c),
                      beta_cols[2] if len(beta_cols) > 2 else beta_cols[0])
            persistence = har[b1] + har[b2] + har[b3]

            # ── Average summary on top ─────────────────────────────────────────
            st.markdown("#### Average Coefficients — All 30 Stocks")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Avg β₁  Daily",        f"{har[b1].mean():.4f}")
            mc2.metric("Avg β₂  Weekly",       f"{har[b2].mean():.4f}")
            mc3.metric("Avg β₃  Monthly",      f"{har[b3].mean():.4f}")
            mc4.metric("Avg Persistence (Σβ)", f"{persistence.mean():.4f}")
            info(
                f"Daily dominates: avg β₁ ≈ {har[b1].mean():.3f} "
                f"vs β₂ ≈ {har[b2].mean():.3f} vs β₃ ≈ {har[b3].mean():.3f}. "
                f"Avg persistence β₁+β₂+β₃ ≈ {persistence.mean():.3f} — "
                "near-unit-root long memory in lnRV."
            )

            st.divider()

            # ── Heatmap + per-stock detail panel ──────────────────────────────
            col_heat, col_stock = st.columns([3, 1])

            with col_heat:
                st.markdown("#### Coefficient Heatmap — All Stocks")
                filter_ticker = st.selectbox(
                    "Highlight / inspect a stock", ["None"] + list(har.index),
                    key="har_filter"
                )
                import plotly.graph_objects as go
                hmap_df = har[[b1, b2, b3]].copy()
                hmap_df.columns = ['β₁ Daily', 'β₂ Weekly', 'β₃ Monthly']

                fig_heat = go.Figure(data=go.Heatmap(
                    z=hmap_df.values.T,
                    x=hmap_df.index.tolist(),
                    y=['β₁ Daily', 'β₂ Weekly', 'β₃ Monthly'],
                    colorscale='Blues',
                    hoverongaps=False,
                    colorbar=dict(title='Coeff', thickness=12),
                    hovertemplate='<b>%{x}</b><br>%{y}: %{z:.4f}<extra></extra>',
                ))
                if filter_ticker != "None" and filter_ticker in hmap_df.index:
                    idx = hmap_df.index.tolist().index(filter_ticker)
                    fig_heat.add_vline(
                        x=idx, line_width=2.5,
                        line_color='crimson', opacity=0.85
                    )
                fig_heat.update_layout(
                    title='HAR Coefficients β₁ / β₂ / β₃ by Stock',
                    xaxis_title='Stock',
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                    height=310,
                    margin=dict(t=50, b=60, l=80, r=30),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            with col_stock:
                st.markdown("#### Stock Detail")
                if filter_ticker != "None" and filter_ticker in har.index:
                    row = har.loc[filter_ticker]
                    st.metric("β₁  Daily",   f"{row[b1]:.4f}")
                    st.metric("β₂  Weekly",  f"{row[b2]:.4f}")
                    st.metric("β₃  Monthly", f"{row[b3]:.4f}")
                    if r2_col and r2_col in har.columns:
                        st.metric("R²",      f"{row[r2_col]:.4f}")
                    st.metric("Persistence", f"{row[beta_cols].sum():.4f}")
                else:
                    st.caption("Select a stock to see its individual coefficients.")

            st.divider()

            # ── 3D Scatter ────────────────────────────────────────────────────
            st.markdown("#### 3D Coefficient Space")
            import plotly.express as px
            plot_df = har.reset_index()
            plot_df.columns = [str(c).strip() for c in plot_df.columns]
            idx_col = plot_df.columns[0]

            fig3d = px.scatter_3d(
                plot_df,
                x=b1, y=b2, z=b3,
                size=r2_col if r2_col else None,
                color=r2_col if r2_col else b1,
                text=idx_col,
                color_continuous_scale='viridis',
                size_max=25,
                title='HAR Coefficients — β₁ Daily / β₂ Weekly / β₃ Monthly',
                labels={b1: 'β₁ Daily', b2: 'β₂ Weekly',
                        b3: 'β₃ Monthly', idx_col: 'Ticker'}
            )
            fig3d.update_traces(textfont_size=9)
            st.plotly_chart(fig3d, use_container_width=True)
            info("Drag to rotate · scroll to zoom · hover for exact values.")


# ══════════════════════════════════════════════════════════════════════════════
#  LINEAR FORECASTS  (HAR / AR1 / RW)
# ══════════════════════════════════════════════════════════════════════════════
elif "Linear" in section:
    qheader("Linear Forecasts — HAR, AR(1), Random Walk")
    st.markdown(r"""
**Setup:** Last 50% of observations · horizon $h=1$ · **expanding window** (re-estimated each step).

| Model | Specification |
|-------|--------------|
| **HAR** | $y_{t+1} = \beta_0 + \beta_1 y_t + \beta_2 y^{(5)}_t + \beta_3 y^{(22)}_t$ |
| **AR(1)** | $y_{t+1} = \alpha_0 + \alpha_1 y_t$ |
| **Random Walk** | $\hat{y}_{t+1} = y_t$ |

Metrics: **MFE** (mean forecast error — bias) and **RMSE** (root mean squared error — accuracy).
""")

    if D['forecast'] is None:
        missing('har_forecast.csv / ar1_forecast.csv / rw_forecast.csv')
    else:
        df = D['forecast']
        lin_kw = ['HAR', 'AR1', 'AR(1)', 'RW', 'Random']
        ml_kw  = ['EN', 'RF', 'GBM', 'ElasticNet', 'Forest', 'Boosting']

        rmse_lin = [c for c in df.columns
                    if 'RMSE' in c
                    and any(k in c for k in lin_kw)
                    and not any(k in c for k in ml_kw)]
        mfe_lin  = [c for c in df.columns
                    if 'MFE' in c
                    and any(k in c for k in lin_kw)
                    and not any(k in c for k in ml_kw)]
        rmse_cols = rmse_lin or [c for c in df.columns if 'RMSE' in c]
        mfe_cols  = mfe_lin  or [c for c in df.columns if 'MFE'  in c]

        # ── Global summary metrics ─────────────────────────────────────────────
        if rmse_cols:
            avg_rmse = df[rmse_cols].mean().sort_values()
            st.markdown("#### Average RMSE — All 30 Stocks")
            met_cols = st.columns(len(avg_rmse))
            for i, (model, val) in enumerate(avg_rmse.items()):
                badge = "🏆 Best" if i == 0 else f"+{((val / avg_rmse.iloc[0]) - 1)*100:.1f}% vs best"
                met_cols[i].metric(model, f"{val:.6f}", delta=badge)
            st.divider()

        # ── Stock filter ───────────────────────────────────────────────────────
        sel_stock = st.selectbox(
            "Inspect a specific stock", ["All stocks"] + list(df.index),
            key="lin_stock"
        )

        tab_rmse_t, tab_mfe_t, tab_tbl_t = st.tabs(["RMSE", "MFE", "Full Table"])

        with tab_rmse_t:
            if rmse_cols:
                if sel_stock == "All stocks":
                    avg_r = df[rmse_cols].mean().sort_values()
                    col_l, col_r = st.columns([1, 2])
                    with col_l:
                        st.markdown("**Avg RMSE by Model**")
                        st.dataframe(
                            avg_r.rename("Avg RMSE").to_frame()
                            .style.format(precision=6)
                            .background_gradient(cmap='Reds'),
                            use_container_width=True
                        )
                        st.success(f"Best: **{avg_r.index[0]}**")
                    with col_r:
                        hbar(avg_r, "Avg RMSE — Linear Models", "RMSE")
                    st.divider()
                    st.markdown("**RMSE by Stock and Model**")
                    grouped_bar(df, rmse_cols, "RMSE by Stock and Model", "RMSE")

                else:
                    if sel_stock in df.index:
                        row_r = df.loc[sel_stock, rmse_cols].sort_values()
                        col_l, col_r = st.columns([1, 2])
                        with col_l:
                            st.markdown(f"**RMSE — {sel_stock}**")
                            for mdl, val in row_r.items():
                                st.metric(mdl, f"{val:.6f}")
                        with col_r:
                            hbar(row_r, f"RMSE — {sel_stock}", "RMSE")

        with tab_mfe_t:
            if mfe_cols:
                if sel_stock == "All stocks":
                    avg_m = df[mfe_cols].mean()
                    col_l, col_r = st.columns([1, 2])
                    with col_l:
                        st.markdown("**Avg MFE by Model**")
                        st.dataframe(
                            avg_m.rename("Avg MFE").to_frame()
                            .style.format(precision=6),
                            use_container_width=True
                        )
                    with col_r:
                        s_mfe = avg_m.sort_values()
                        fig, ax = plt.subplots(figsize=(8, max(2.5, len(s_mfe)*0.45)))
                        s_mfe.plot(kind='barh', ax=ax, color='#3b82f6', edgecolor='white')
                        ax.axvline(0, color='red', lw=1, ls='--')
                        ax.set_title("Avg MFE — Linear Models", fontsize=11)
                        ax.set_xlabel("MFE")
                        ax.grid(True, alpha=0.25, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    info("MFE ≈ 0 for all models: no systematic forecast bias.")
                    st.divider()
                    st.markdown("**MFE by Stock and Model**")
                    grouped_bar(df, mfe_cols, "MFE by Stock and Model", "MFE")

                else:
                    if sel_stock in df.index:
                        row_m = df.loc[sel_stock, mfe_cols]
                        col_l, col_r = st.columns([1, 2])
                        with col_l:
                            st.markdown(f"**MFE — {sel_stock}**")
                            for mdl, val in row_m.items():
                                st.metric(mdl, f"{val:.6f}")
                        with col_r:
                            fig, ax = plt.subplots(figsize=(8, max(2.5, len(row_m)*0.45)))
                            row_m.sort_values().plot(
                                kind='barh', ax=ax, color='#3b82f6', edgecolor='white')
                            ax.axvline(0, color='red', lw=1, ls='--')
                            ax.set_title(f"MFE — {sel_stock}", fontsize=11)
                            ax.grid(True, alpha=0.25, axis='x')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

        with tab_tbl_t:
            lin_all    = rmse_cols + mfe_cols
            display_df = df[lin_all].round(6) if lin_all else df.round(6)
            if sel_stock != "All stocks" and sel_stock in df.index:
                display_df = display_df.loc[[sel_stock]]
            styled = display_df.style.format(precision=6)
            if rmse_cols:
                try:
                    styled = styled.highlight_min(axis=1, color='#d4edda', subset=rmse_cols)
                except Exception:
                    pass
            st.dataframe(styled, use_container_width=True)

        st.divider()
        st.markdown("""
**Key findings:**
- HAR consistently outperforms AR(1) and Random Walk thanks to the weekly and monthly components.
- The Random Walk is a surprisingly competitive baseline, reflecting the high day-to-day persistence of volatility.
- MFE ≈ 0 for all models: no systematic bias in any direction.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif "ML" in section:
    qheader("Machine Learning Forecasts")

    col_en, col_rf, col_gbm = st.columns(3)
    with col_en:
        st.markdown("""<div class="info-box"><b>ElasticNet (EN)</b><br>
Regularised regression combining L1 (Lasso) + L2 (Ridge) penalties. Best for sparse
linear relationships. Same economic interpretation as OLS but with controlled overfitting.</div>""",
                    unsafe_allow_html=True)
    with col_rf:
        st.markdown("""<div class="info-box"><b>Random Forest (RF)</b><br>
Ensemble of 100 decision trees via bagging and random feature selection.
Captures nonlinear interactions robustly. Naturally resistant to overfitting.</div>""",
                    unsafe_allow_html=True)
    with col_gbm:
        st.markdown("""<div class="info-box"><b>Gradient Boosting (GBM)</b><br>
Sequential tree boosting — each tree corrects the previous residuals.
Often the most accurate ML model but sensitive to hyperparameters.</div>""",
                    unsafe_allow_html=True)

    st.markdown(
        r"Feature set: same HAR variables — $y_{t-1}$,  $y^{(5)}_{t-1}$,  $y^{(22)}_{t-1}$. "
        "Expanding-window OOS setup identical to the linear forecasts."
    )
    st.divider()

    if D['forecast'] is None:
        missing('ml_fs1_ElasticNet.csv / ml_fs1_RF.csv / ml_fs1_GBM.csv')
    else:
        df      = D['forecast']
        ml_kw   = ['EN', 'RF', 'GBM', 'ElasticNet', 'Forest', 'Boosting']
        har_kw  = ['HAR']
        rmse_ml  = [c for c in df.columns if 'RMSE' in c and any(k in c for k in ml_kw)]
        rmse_har = [c for c in df.columns if 'RMSE' in c and any(k in c for k in har_kw)
                    and not any(k in c for k in ml_kw)]
        all_rmse = [c for c in df.columns if 'RMSE' in c]
        compare  = rmse_har + rmse_ml if rmse_har else rmse_ml

        # ── Summary metrics ────────────────────────────────────────────────────
        if compare:
            avg_cmp = df[compare].mean().sort_values()
            har_val = df[rmse_har].mean().values[0] if rmse_har else None
            st.markdown("#### Average RMSE — ML Models vs HAR Baseline")
            met_cols = st.columns(len(avg_cmp))
            for i, (model, val) in enumerate(avg_cmp.items()):
                if har_val and any(k in model for k in har_kw) and not any(k in model for k in ml_kw):
                    met_cols[i].metric(model, f"{val:.6f}", delta="HAR baseline")
                elif har_val:
                    d = f"{((val / har_val) - 1)*100:+.1f}% vs HAR"
                    met_cols[i].metric(model, f"{val:.6f}", delta=d, delta_color="inverse")
                else:
                    met_cols[i].metric(model, f"{val:.6f}")
            st.divider()

        # ── Stock filter ───────────────────────────────────────────────────────
        sel_ml = st.selectbox(
            "Inspect a specific stock", ["All stocks"] + list(df.index),
            key="ml_stock"
        )

        tab_cmp_t, tab_full_t = st.tabs(["RMSE Comparison", "Full Results"])

        with tab_cmp_t:
            if compare:
                if sel_ml == "All stocks":
                    avg_c = df[compare].mean().sort_values()
                    col_l, col_r = st.columns([1, 2])
                    with col_l:
                        st.markdown("**Avg RMSE by Model**")
                        st.dataframe(
                            avg_c.rename("Avg RMSE").to_frame()
                            .style.format(precision=6)
                            .background_gradient(cmap='Reds'),
                            use_container_width=True
                        )
                        st.success(f"Best: **{avg_c.index[0]}**")
                    with col_r:
                        s_c = avg_c
                        colours_ml = []
                        for j, c in enumerate(s_c.index):
                            if j == 0:
                                colours_ml.append('#16a34a')
                            elif any(k in c for k in har_kw) and not any(k in c for k in ml_kw):
                                colours_ml.append('#1d4ed8')
                            else:
                                colours_ml.append('#94a3b8')
                        fig, ax = plt.subplots(figsize=(8, max(2.5, len(s_c)*0.45)))
                        s_c.plot(kind='barh', ax=ax, color=colours_ml, edgecolor='white')
                        ax.set_title("Avg RMSE — ML vs HAR Baseline", fontsize=11)
                        ax.set_xlabel("Avg RMSE")
                        ax.grid(True, alpha=0.25, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    st.divider()
                    st.markdown("**RMSE by Stock and Model**")
                    grouped_bar(df, compare, "RMSE by Stock — ML vs HAR", "RMSE")

                else:
                    if sel_ml in df.index:
                        row_ml = df.loc[sel_ml, compare].sort_values()
                        col_l, col_r = st.columns([1, 2])
                        with col_l:
                            st.markdown(f"**RMSE — {sel_ml}**")
                            for mdl, val in row_ml.items():
                                st.metric(mdl, f"{val:.6f}")
                        with col_r:
                            hbar(row_ml, f"RMSE — {sel_ml}", "RMSE")

        with tab_full_t:
            display_ml = df[all_rmse].round(6) if all_rmse else df.round(6)
            if sel_ml != "All stocks" and sel_ml in df.index:
                display_ml = display_ml.loc[[sel_ml]]
            styled_ml = display_ml.style.format(precision=6)
            if all_rmse:
                try:
                    styled_ml = styled_ml.highlight_min(axis=1, color='#d4edda', subset=all_rmse)
                except Exception:
                    pass
            st.dataframe(styled_ml, use_container_width=True)
            if all_rmse and sel_ml == "All stocks":
                best_all = df[all_rmse].mean().idxmin()
                st.success(f"Best overall model (lowest avg RMSE across all stocks): **{best_all}**")

        st.divider()
        st.markdown("""
**Discussion:** In line with the reference paper (Kilic, 2025), HAR proves a tough benchmark.
ML models capture some nonlinear patterns but rarely outperform HAR consistently
when the feature set is limited to past RV values. Model complexity is not always beneficial —
economic structure beats raw flexibility.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  GRANGER CAUSALITY
# ══════════════════════════════════════════════════════════════════════════════
elif "Granger" in section:
    qheader("Granger Causality and Extended Forecasting")
    st.markdown(
        r"We test $H_0$: past lnRV of stock **A** carries no additional information "
        r"for forecasting lnRV of stock **B**, beyond B's own history."
    )
    st.latex(
        r"y_{B,t} = \alpha + \sum_{k=1}^{p}\phi_k y_{B,t-k} "
        r"+ \sum_{k=1}^{p}\psi_k y_{A,t-k} + \varepsilon_t"
    )

    tab_gc, tab_har_gc = st.tabs(["Granger Tests", "HAR + Granger Forecasts"])

    # ── Granger Tests ──────────────────────────────────────────────────────────
    with tab_gc:
        if D['granger'] is None:
            missing('granger_results.csv')
        else:
            gc = D['granger']
            if gc.index.nlevels > 1:
                gc = gc.reset_index()
            gc.columns = [str(c).strip() for c in gc.columns]

            # Deduplicate column names if needed (Styler requires unique columns)
            if gc.columns.duplicated().any():
                seen = {}
                new_cols = []
                for c in gc.columns:
                    if c in seen:
                        seen[c] += 1
                        new_cols.append(f"{c}_{seen[c]}")
                    else:
                        seen[c] = 0
                        new_cols.append(c)
                gc.columns = new_cols

            # Always reset to clean integer index before styling
            gc = gc.reset_index(drop=True)

            pair_col  = next((c for c in gc.columns
                              if 'Cause' in c or 'Effect' in c or '→' in c), gc.columns[0])
            fstat_col = next((c for c in gc.columns
                              if 'F-stat' in c or 'F_stat' in c or 'F stat' in c), None)
            sig_col   = next((c for c in gc.columns
                              if 'Significant' in c or 'significant' in c), None)

            try:
                styled = gc.style.format(precision=4)
                if fstat_col:
                    styled = styled.background_gradient(cmap='RdYlGn', subset=[fstat_col])
                if sig_col:
                    styled = styled.map(
                        lambda v: 'background-color:#d4edda'
                                  if v is True or v == 'True'
                                  else ('background-color:#f8d7da'
                                        if v is False or v == 'False' else ''),
                        subset=[sig_col]
                    )
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(gc.round(4), use_container_width=True)

            if fstat_col and pair_col:
                gc_plot = gc.copy()
                fig, ax = plt.subplots(figsize=(14, 5))
                colours = ['#16a34a' if str(v) == 'True' else '#94a3b8'
                           for v in (gc_plot[sig_col] if sig_col
                                     else ['True'] * len(gc_plot))]
                ax.barh(gc_plot[pair_col].astype(str),
                        gc_plot[fstat_col].astype(float),
                        color=colours, edgecolor='white')
                ax.set_xlabel("Max F-statistic")
                ax.set_title(
                    "Granger Causality — Max F-Statistic by Pair  (green = significant at 5%)",
                    fontsize=11
                )
                ax.axvline(0, color='black', lw=0.8)
                ax.grid(True, alpha=0.25, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.divider()
            st.markdown("""
**Key findings:**
- Within-sector causality (e.g. JPM → GS, MCD → WMT) is significantly stronger than cross-sector.
- AAPL → MSFT (F ≈ 79.7) is more than twice as strong as MSFT → AAPL (F ≈ 32.9): AAPL is the dominant volatility source in tech.
- DOW pairs are the weakest due to its short sample (~750 overlapping days).
- Granger causality is pervasive but driven mainly by a common market factor; sector channels provide incremental content above that baseline.
""")

    # ── HAR + Granger Forecasts ────────────────────────────────────────────────
    with tab_har_gc:
        st.markdown(
            "Where Granger causality is detected, we augment HAR "
            "with the causing stock's lagged lnRV:"
        )
        st.latex(
            r"y_{B,t} = \beta_0 + \beta_1 y_{B,t-1} + \beta_2 y^{(5)}_{B,t-1} "
            r"+ \beta_3 y^{(22)}_{B,t-1} + \beta_4 y_{A,t-1} + \varepsilon_t"
        )
        st.markdown(
            "We use the same expanding window design (last 50%, horizon $h=1$, "
            "re-estimated at each step) and compare MFE and RMSE against the "
            "standard HAR to assess whether the additional information improves forecasts."
        )

        if D['gc_fore'] is None:
            missing('granger_forecast.csv')
        else:
            gc_f     = D['gc_fore']
            rmse_gc  = next((c for c in gc_f.columns if 'HAR+GC' in c and 'RMSE' in c), None)
            rmse_har = next((c for c in gc_f.columns
                             if 'HAR' in c and 'RMSE' in c and 'GC' not in c), None)
            pair_col = next((c for c in gc_f.columns
                             if 'pair' in c.lower() or 'Pair' in c), None)

            st.dataframe(gc_f.round(6), use_container_width=True)

            if rmse_gc and rmse_har:
                gc_f = gc_f.copy()
                gc_f['Improvement (%)'] = (1 - gc_f[rmse_gc] / gc_f[rmse_har]) * 100
                labels = gc_f[pair_col].values if pair_col else gc_f.index

                fig, ax = plt.subplots(figsize=(8, 3.5))
                colours_imp = ['#16a34a' if v > 0 else '#dc2626'
                               for v in gc_f['Improvement (%)']]
                ax.bar(range(len(labels)), gc_f['Improvement (%)'],
                       color=colours_imp, edgecolor='white', linewidth=0.5)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
                ax.axhline(0, color='black', lw=0.8)
                ax.set_title("RMSE Improvement of HAR+GC over HAR (%)", fontsize=11)
                ax.set_ylabel("Improvement (%)")
                ax.grid(True, alpha=0.25, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.divider()
            st.markdown("""
**Key finding:** Adding one carefully chosen variable — the lagged volatility of an
economically linked stock — outperforms more complex ML models.

**Why?** Economic reasoning guides *what* information to add. ML adds complexity;
Granger augmentation adds *relevance*. The lesson: knowing which extra variable to include,
guided by theory, is more valuable than making the model more complex.
""")
            info(
                "MCD → WMT achieves ≈ 1.3% RMSE reduction; AAPL → MSFT achieves ≈ 0.5%. "
                "The stronger the in-sample F-statistic, the larger the OOS improvement."
            )
