# ai_data_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import base64
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

# Optional OpenAI
try:
    import openai
except Exception:
    openai = None

# -------------------------
# Helper: AI Summary wrapper
# -------------------------
def generate_ai_summary(df, openai_key: str = None, max_cols=8):
    """
    If openai_key provided and openai package available -> call OpenAI ChatCompletion.
    Otherwise return heuristic summary.
    """
    rows, cols = df.shape
    sample_cols = list(df.columns[:max_cols])
    missing_summary = df.isna().sum().sort_values(ascending=False).head(6).to_dict()

    base_summary = (
        f"Dataset has {rows} rows and {cols} columns. "
        f"Sample columns: {', '.join(sample_cols)}. "
        f"Top missing counts: {missing_summary}."
    )

    if openai_key and openai:
        try:
            openai.api_key = openai_key
            prompt = (
                "You are a helpful data analyst. Summarize the dataset quickly and provide:\n"
                "1) 3 short insights\n"
                "2) 2 possible data quality issues\n"
                "3) 2 recommended next charts to visualize\n\n"
                f"DATA SUMMARY: {base_summary}\n"
                "Also add one suggestion for forecasting (if time series present)."
            )
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # fall back to heuristic
            return base_summary + f" (OpenAI call failed: {e})"
    else:
        # heuristic insights: numeric column top correlations and top categories
        txt = base_summary + " Heuristics:\n"
        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty:
            corr = numeric.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
            top = corr[corr < 1].head(5)
            txt += "Top numeric correlations: " + "; ".join([f"{idx}: {val:.2f}" for idx, val in top.items()]) + ".\n"
        # top categories
        cat = df.select_dtypes(exclude=[np.number])
        if not cat.empty:
            sample_cat = cat.columns[:3]
            for c in sample_cat:
                topvals = df[c].value_counts().head(3).to_dict()
                txt += f"Top values in {c}: {topvals}. "
        return txt

# --------------------------------
# Helper: forecasting (simple LR)
# --------------------------------
def linear_forecast(df, date_col, value_col, periods=12):
    """
    Simple forecasting: convert date_col to ordinal, fit linear regression on value_col,
    and forecast next `periods` equally spaced steps (based on median delta).
    Returns forecast_df with future dates and predicted values.
    """
    series = df[[date_col, value_col]].dropna()
    # parse dates
    try:
        series[date_col] = pd.to_datetime(series[date_col])
    except Exception:
        # try parsing each
        series[date_col] = series[date_col].apply(lambda x: parse_date(str(x), default=datetime.now()))
    series = series.sort_values(date_col)
    if series.shape[0] < 3:
        raise ValueError("Not enough points to forecast")

    X_dates = series[date_col].map(datetime.toordinal).values.reshape(-1, 1)
    y = series[value_col].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_dates, y)
    # compute next dates by median delta
    deltas = series[date_col].diff().dropna().map(lambda x: x.total_seconds()).astype(float)
    median_delta_seconds = np.median(deltas) if len(deltas) > 0 else 86400
    # build future dates
    last_date = series[date_col].iloc[-1]
    future_dates = [last_date + timedelta(seconds=int(median_delta_seconds * (i + 1))) for i in range(periods)]
    X_future = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    preds = model.predict(X_future).flatten()
    return pd.DataFrame({date_col: future_dates, value_col: preds})

# --------------------------
# Helper: create downloadable HTML report
# --------------------------
def create_html_report(title, summary_text, figs):
    """
    figs: list of (title, plotly_fig) tuples. We'll embed plotly HTML divs.
    Returns bytes of HTML file.
    """
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title></head><body style='background:#0e1117;color:#fff;font-family:Arial'>",
             f"<h1>{title}</h1>",
             f"<pre style='color:#ddd'>{summary_text}</pre>",
             "<hr/>"]
    for t, fig in figs:
        try:
            inner = fig.to_html(full_html=False, include_plotlyjs='cdn')
            parts.append(f"<h2 style='color:#fff'>{t}</h2>")
            parts.append(inner)
        except Exception:
            parts.append(f"<h2>{t}</h2><p>Failed to render chart</p>")
    parts.append("</body></html>")
    return "\n".join(parts).encode("utf-8")

# --------------------------
# Heuristic chatbot Q&A
# --------------------------
from difflib import get_close_matches
import plotly.express as px
import pandas as pd
import numpy as np
import re

def heuristic_qa(df, question: str):
    q = question.lower().strip()

    # 🔹 Synonyms (you can expand freely)
    synonym_map = {
        "revenue": "sales",
        "sale": "sales",
        "product": "product name",
        "item": "product name",
        "profit": "profit",
        "income": "profit",
        "category": "category",
        "customer": "customer name",
        "region": "region",
        "city": "city",
        "state": "state",
        "country": "country",
        "segment": "segment",
    }

    # 🔹 Helper: fuzzy match
    def find_col(keyword):
        keyword = keyword.lower().strip()
        if keyword in synonym_map:
            keyword = synonym_map[keyword]
        cols = [c.lower() for c in df.columns]
        match = get_close_matches(keyword, cols, n=1, cutoff=0.5)
        if match:
            for c in df.columns:
                if c.lower() == match[0]:
                    return c
        return None

    # 🔹 Extract number (Top N)
    n_match = re.search(r"top\\s*(\\d+)", q)
    n = int(n_match.group(1)) if n_match else 5

    # default response
    text_response = "Sorry, I couldn't interpret that question."
    chart = None

    # -------------------------------
    # Handle “Top N … by …”
    # -------------------------------
    if "top" in q and "by" in q:
        try:
            words = q.split()
            subject_keyword = None
            metric_keyword = None
            for i, w in enumerate(words):
                if w == "top" and i + 1 < len(words):
                    if words[i + 1].isdigit() and i + 2 < len(words):
                        subject_keyword = words[i + 2]
                    else:
                        subject_keyword = words[i + 1]
                if w == "by" and i + 1 < len(words):
                    metric_keyword = words[i + 1]

            subject_col = find_col(subject_keyword)
            metric_col = find_col(metric_keyword)

            if subject_col and metric_col:
                result = (
                    df.groupby(subject_col)[metric_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(n)
                )
                text_response = f"Top {n} {subject_col} by {metric_col}:\n{result.to_string()}"

                # 📊 Bar chart
                chart = px.bar(
                    result.reset_index(),
                    x=subject_col,
                    y=metric_col,
                    title=f"Top {n} {subject_col} by {metric_col}",
                    template="plotly_dark",
                    color_discrete_sequence=["#00c8ff"],
                )
            else:
                text_response = f"I couldn't find matching columns for '{subject_keyword}' or '{metric_keyword}'."
        except Exception as e:
            text_response = f"Error interpreting query: {e}"

    # -------------------------------
    # Handle “Average” or “Mean”
    # -------------------------------
    elif "average" in q or "mean" in q:
        for word in q.split():
            col = find_col(word)
            if col:
                avg_val = df[col].mean()
                text_response = f"Average {col}: {avg_val:.2f}"

                # 📈 Single bar chart
                chart = px.bar(
                    x=[col],
                    y=[avg_val],
                    title=f"Average of {col}",
                    template="plotly_dark",
                    color_discrete_sequence=["#2ecc71"],
                )
                break
        else:
            # show all numeric averages
            num_cols = df.select_dtypes(include=[np.number])
            means = num_cols.mean().sort_values(ascending=False)
            text_response = f"Numeric column averages:\n{means.round(2).to_string()}"
            chart = px.bar(
                x=means.index,
                y=means.values,
                title="Column Averages",
                template="plotly_dark",
                color_discrete_sequence=["#2ecc71"],
            )

    # -------------------------------
    # Handle “Correlation”
    # -------------------------------
    elif "correlation" in q:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            text_response = "Not enough numeric columns for correlation."
        else:
            corr = num.corr().round(3)
            text_response = f"Correlation matrix (top values):\n{corr.to_string()}"
            chart = px.imshow(
                corr,
                text_auto=True,
                title="Correlation Heatmap",
                template="plotly_dark",
                color_continuous_scale="Tealrose",
            )

    return text_response, chart

# --------------------------
# Streamlit UI starts here
# --------------------------

# --------------------------
# Helper: Generate narrative summary from result
# --------------------------
def generate_narrative_from_result(df_text, header):
    """
    Generate a human-readable summary from a Top N result DataFrame or text.
    """
    try:
        # Try to detect a simple table from string
        lines = df_text.split("\n")
        data_lines = [line.strip() for line in lines[1:] if line.strip()]
        if not data_lines:
            return ""

        # Parse into dataframe
        import io
        df_result = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\s{2,}", engine="python", header=None)
        if len(df_result.columns) == 2:
            df_result.columns = ["Label", "Value"]
        df_result["Value"] = pd.to_numeric(df_result["Value"], errors="coerce")
        df_result = df_result.dropna(subset=["Value"])

        if len(df_result) >= 2:
            top_item = df_result.iloc[0]
            second_item = df_result.iloc[1]
            ratio = top_item["Value"] / second_item["Value"] if second_item["Value"] else 1
            return f"📊 The top-performing **{header.split('by')[0].split()[-1]}** is **{top_item['Label']}**, with a total of **{top_item['Value']:.2f}**, approximately **{ratio:.1f}×** higher than the next entry."
        elif len(df_result) == 1:
            top_item = df_result.iloc[0]
            return f"📊 The highest value belongs to **{top_item['Label']}** with **{top_item['Value']:.2f}**."
    except Exception:
        pass
    return ""



st.set_page_config(page_title="AI Data Analyzer Pro", layout="wide", initial_sidebar_state="expanded")

# small custom CSS for dark card look
st.markdown("""
<style>
body { background-color: #0b0f14; color: #fff; }
.stApp { background-color: #0b0f14; }
.card { background:#0f1720; border-radius:10px; padding:12px; box-shadow: 0 2px 10px rgba(0,0,0,0.5); }
.small-muted { color:#9aa3b2; font-size:12px; }
h1, h2, h3 { color: #fff; }
</style>
""", unsafe_allow_html=True)

colL, colR = st.columns([3, 1])

with colL:
    st.title("📊 AI Data Analyzer Pro")
    st.write("Upload raw data, explore with powerful visuals, ask questions, forecast and export a storytelling report.")
with colR:
    st.write("")  # spacer
    st.markdown("**Theme:** Dark | Built-in visual mode")

# Sidebar: file upload and filter controls
st.sidebar.header("Upload & Filters")
uploaded = st.sidebar.file_uploader("Upload CSV file (or drop here)", type=["csv", "xlsx", "json"])

if uploaded:
    # read
    try:
        if str(uploaded.name).lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded)
        elif str(uploaded.name).lower().endswith(".json"):
            df = pd.read_json(uploaded)
        else:
            df = pd.read_csv(uploaded, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        # final fallback
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding="latin1", on_bad_lines="skip")

    # show basic preview and cards
    st.subheader("Preview & Metrics")
    preview_cols = st.multiselect("Columns to preview", options=list(df.columns), default=list(df.columns)[:6])
    st.dataframe(df[preview_cols].head(10))

    st.write("🧠 Available columns:", list(df.columns))

    rows, cols = df.shape
    mv = int(df.isna().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", cols)
    c3.metric("Missing values", f"{mv:,}")
    # optional numeric metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        c4.metric("Numeric cols", len(numeric_cols))

    st.markdown("---")
    # Live filtering section
    st.sidebar.header("Live filtering")
    filters = {}
    for col in st.sidebar.multiselect("Filter columns (choose to add filter widgets)", options=list(df.columns), default=[]):
        if pd.api.types.is_numeric_dtype(df[col]):
            vmin, vmax = float(df[col].min()), float(df[col].max())
            low, high = st.sidebar.slider(f"{col} range", vmin, vmax, (vmin, vmax))
            filters[col] = (low, high)
        else:
            vals = st.sidebar.multiselect(f"{col} values", options=list(df[col].dropna().unique()), default=list(df[col].dropna().unique())[:5])
            filters[col] = vals

    # apply filters
    df_filtered = df.copy()
    for k, v in filters.items():
        if isinstance(v, tuple):
            df_filtered = df_filtered[(df_filtered[k] >= v[0]) & (df_filtered[k] <= v[1])]
        elif isinstance(v, list):
            if v:
                df_filtered = df_filtered[df_filtered[k].isin(v)]

    st.subheader("Filtered preview")
    st.dataframe(df_filtered.head(12))

    st.markdown("---")
    # Visualization controls
    st.subheader("Visualizations")
    chart = st.selectbox("Choose chart", ["Histogram", "Line", "Bar", "Scatter", "Correlation heatmap", "Time series with forecast"])
    # columns selection for chart
    if chart in ["Histogram", "Line"]:
        col = st.selectbox("Column", df_filtered.select_dtypes(include=[np.number]).columns.tolist())
    elif chart == "Bar":
        cat = st.selectbox("Categorical Column", df_filtered.select_dtypes(exclude=[np.number]).columns.tolist())
    elif chart == "Scatter":
        num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        xcol = st.selectbox("X axis", num_cols)
        ycol = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
    elif chart == "Correlation heatmap":
        num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    elif chart == "Time series with forecast":
        # choose date col and value col
        date_col = st.selectbox("Date column", df_filtered.columns.tolist())
        value_col = st.selectbox("Value column (numeric)", df_filtered.select_dtypes(include=[np.number]).columns.tolist())
        periods = st.number_input("Forecast periods (steps)", min_value=1, max_value=365, value=12)

    # rendering charts
    if chart == "Histogram":
        fig = px.histogram(df_filtered, x=col, nbins=40, title=f"Distribution: {col}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Line":
        fig = px.line(df_filtered, y=col, title=f"Line: {col}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Bar":
        vc = df_filtered[cat].value_counts().reset_index()
        vc.columns = [cat, "count"]
        fig = px.bar(vc, x=cat, y="count", title=f"Counts: {cat}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Scatter":
        fig = px.scatter(df_filtered, x=xcol, y=ycol, trendline="ols", title=f"{ycol} vs {xcol}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Correlation heatmap":
        corr = df_filtered[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation heatmap", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Time series with forecast":
        # prepare series
        try:
            s = df_filtered[[date_col, value_col]].dropna()
            s[date_col] = pd.to_datetime(s[date_col])
            s = s.sort_values(date_col)
            fig = px.line(s, x=date_col, y=value_col, title=f"Time series of {value_col}", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            # forecast
            try:
                forecast_df = linear_forecast(s, date_col, value_col, periods=int(periods))
                fig2 = px.line(pd.concat([s.rename(columns={date_col:"date", value_col:"value"}),
                                         forecast_df.rename(columns={date_col:"date", value_col:"value"})]),
                               x="date", y="value", color_discrete_sequence=["#00c8ff", "#f39c12"],
                               title="Historical + Forecast (LR)", template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f"Forecast failed: {e}")
        except Exception as e:
            st.error(f"Time series plotting error: {e}")

    # --------------------------
    # AI Insights & Chatbot Section
    # --------------------------
    st.markdown("---")
    st.subheader("🤖 AI Insights & Chatbot")

    openai_key = st.text_input("OpenAI API key (optional — leave blank to use heuristics)", type="password")

    if st.button("Generate AI Summary"):
        with st.spinner("Generating summary..."):
            summary = generate_ai_summary(df_filtered, openai_key)
            st.markdown("**🧠 AI Summary:**")
            st.write(summary)

    st.markdown(
        "**Ask a question about the data** (try: 'Top 5 Product by Revenue', 'Average Sales', 'Show correlation')")
    question = st.text_input("Ask a question")

    if st.button("Answer question"):
        # --- Try OpenAI first ---
        if openai_key and openai:
            try:
                openai.api_key = openai_key
                context = f"DATA: rows={rows}, cols={cols}, columns={list(df_filtered.columns)[:12]}"
                prompt = f"You are a data analyst. Answer succinctly. {context}\nQUESTION: {question}"
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                )
                ans = resp.choices[0].message.content.strip()
                st.markdown("### 💬 AI Answer")
                st.write(ans)
            except Exception as e:
                st.error(f"OpenAI call error: {e}. Falling back to heuristic mode.")
                response_text, chart = heuristic_qa(df_filtered, question)
        else:
            # --- Use heuristic fallback ---
            response_text, chart = heuristic_qa(df_filtered, question)

            import re, io, pandas as pd

            st.markdown("### 💬 AI Insight")

            # 🧩 Detect and format “Top N … by …” results neatly as a table
            if re.search(r"Top\s*\d+", response_text, re.IGNORECASE) and "by" in response_text:
                lines = response_text.split("\n")
                header = lines[0]
                data_lines = [line.strip() for line in lines[1:] if line.strip()]

                try:
                    # Convert to DataFrame
                    df_str = "\n".join(data_lines)
                    df_result = pd.read_csv(io.StringIO(df_str), sep="\s{2,}", engine="python", header=None)
                    if len(df_result.columns) == 2:
                        df_result.columns = ["Label", "Value"]

                    st.markdown(f"**{header}**")
                    st.dataframe(df_result, use_container_width=True)

                    # 🧠 Generate AI-style narrative
                    summary_text = generate_narrative_from_result(response_text, header)
                    if summary_text:
                        st.info(summary_text)

                except Exception:
                    st.text(response_text)

            # 🌿 Highlight averages in green
            elif "Average" in response_text or "average" in response_text:
                st.success(response_text)

            # 🌐 Highlight correlation in blue
            elif "Correlation" in response_text or "correlation" in response_text:
                st.info(response_text)

            # 🪶 Default fallback
            else:
                st.write(response_text)

            # 🎨 Show chart if available
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)

        import re
        import pandas as pd
        import io
        response_text, chart = heuristic_qa(df_filtered, question)
        st.markdown("### 💬 AI Insight")
        # Detect and format “Top N … by …” results neatly as a table

        if re.search(r"Top\s*\d+", response_text, re.IGNORECASE) and "by" in response_text:
            lines = response_text.split("\n")
            header = lines[0]
            data_lines = [line.strip() for line in lines[1:] if line.strip()]
            # Try to parse text into table

            try:
                # convert lines into dataframe
                df_str = "\n".join(data_lines)
                df_result = pd.read_csv(io.StringIO(df_str), sep="\s{2,}", engine="python", header=None)

                # Set proper column names dynamically
                if len(df_result.columns) == 2:
                    df_result.columns = ["Label", "Value"]

                elif len(df_result.columns) == 3:
                    df_result.columns = ["Category", "Metric", "Value"]

                st.markdown(f"**{header}**")
                st.dataframe(df_result, use_container_width=True)

            except Exception:
                # fallback if parsing fails
                st.text(response_text)
        # Highlight averages with green box

        elif "Average" in response_text or "average" in response_text:
            st.success(response_text)
        # Highlight correlation results with blue info box

        elif "Correlation" in response_text or "correlation" in response_text:
            st.info(response_text)
        # Default display

        else:
            st.write(response_text)
        # Show chart if available

        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

    st.markdown("---")
    # Pivot table creator
    st.subheader("Pivot table (interactive)")
    pivot_index = st.selectbox("Index", options=[None] + list(df_filtered.columns), index=0)
    pivot_columns = st.selectbox("Columns", options=[None] + list(df_filtered.columns), index=0)
    pivot_values = st.selectbox("Values (agg)", options=[None] + list(df_filtered.columns), index=0)
    aggfunc = st.selectbox("Agg function", options=["sum","mean","count","median"], index=1)
    if st.button("Create Pivot"):
        try:
            if pivot_values is None:
                st.warning("Select a values column for aggregation.")
            else:
                pt = pd.pivot_table(df_filtered, index=pivot_index if pivot_index else None,
                                    columns=pivot_columns if pivot_columns else None,
                                    values=pivot_values, aggfunc=aggfunc)
                st.dataframe(pt.fillna("").head(200))
        except Exception as e:
            st.error(f"Pivot failed: {e}")

    st.markdown("---")
    # Storytelling dashboard + export
    st.subheader("Storytelling Report & Export")
    report_title = st.text_input("Report title", value="AI Data Report")
    make_figs = []
    st.write("Select up to 4 charts to include in the report:")
    # produce a few default figures to choose from
    # default: top numeric hist, correlation heatmap, top categorical bar
    if numeric_cols:
        f1 = px.histogram(df_filtered, x=numeric_cols[0], nbins=30, title=f"Distribution: {numeric_cols[0]}", template="plotly_dark")
        make_figs.append((f"Distribution: {numeric_cols[0]}", f1))
    if len(numeric_cols) > 1:
        f2 = px.scatter(df_filtered, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[1]} vs {numeric_cols[0]}", template="plotly_dark")
        make_figs.append((f"{numeric_cols[1]} vs {numeric_cols[0]}", f2))
    if df_filtered.select_dtypes(exclude=[np.number]).shape[1] > 0:
        catcol = df_filtered.select_dtypes(exclude=[np.number]).columns[0]
        vc = df_filtered[catcol].value_counts().reset_index()
        vc.columns = [catcol, "count"]
        f3 = px.bar(vc, x=catcol, y="count", title=f"Counts: {catcol}", template="plotly_dark")
        make_figs.append((f"Counts: {catcol}", f3))
    if len(numeric_cols) > 1:
        corr = df_filtered[numeric_cols].corr()
        f4 = px.imshow(corr, text_auto=True, title="Correlation heatmap", template="plotly_dark")
        make_figs.append(("Correlation heatmap", f4))

    # let user pick
    selected = st.multiselect("Include charts", options=[t for t, _ in make_figs], default=[t for t, _ in make_figs][:3])
    figs_to_include = [(t, fig) for t, fig in make_figs if t in selected]

    summary_text = generate_ai_summary(df_filtered, openai_key)
    if st.button("Generate and Download Report"):
        # build HTML and offer download
        html_bytes = create_html_report(report_title, summary_text, figs_to_include)
        st.download_button("Download HTML report", data=html_bytes, file_name="report.html", mime="text/html")
        st.success("Report ready for download.")

else:
    st.info("Upload a dataset to begin. Example: Global Superstore or e-commerce sales CSV.")
    st.write("I can: show interactive Plotly charts, forecast a numeric time series (simple linear), accept OpenAI API key for richer AI summaries and chat answers, create pivot tables, and generate a downloadable storytelling HTML report.")
