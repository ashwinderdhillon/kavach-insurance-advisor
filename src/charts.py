"""Plotly figures built straight from tool results (no numbers of their own)."""
import plotly.graph_objects as go

BLUE, GREY = "#2563eb", "#94a3b8"
BANDS = [(0, 40, "rgba(220, 38, 38, 0.25)"), (40, 70, "rgba(217, 119, 6, 0.25)"), (70, 100, "rgba(22, 163, 74, 0.25)")]


def _layout(fig, title, height=320):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.2),
                      hovermode="x unified")
    return fig


def cover_decay(r):
    s, sym = r["series"], r["symbol"]
    years = [p["year"] for p in s]
    fig = go.Figure([
        go.Scatter(x=years, y=[p["real_value_today_prices"] for p in s], name="What your cover is really worth",
                   mode="lines+markers", line=dict(color=BLUE, width=3)),
        go.Scatter(x=years, y=[p["cover_needed_to_match_today"] for p in s], name="Cover needed to keep pace",
                   mode="lines", line=dict(color=GREY, dash="dash")),
    ])
    fig.update_yaxes(tickprefix=sym, tickformat=",.0f", rangemode="tozero")
    fig.update_xaxes(dtick=max(1, len(years) // 6))
    return _layout(fig, f"Cover decay at {r['medical_inflation_pct']:.1f}% medical inflation: "
                        f"{sym}{r['cover']:,.0f} today")


def cost_of_delay(r):
    sym, d = r["symbol"], r["delay_years"]
    t = r["term_life"]
    groups = ["Term life"]
    now, later = [t["annual_premium_now"]], [t["annual_premium_if_delayed"]]
    if "health" in r:
        groups.append("Health")
        now.append(r["health"]["entry_premium_now"])
        later.append(r["health"]["entry_premium_if_delayed"])
    fig = go.Figure([
        go.Bar(x=groups, y=now, name="Buy now", marker_color=BLUE, text=[f"{sym}{v:,.0f}" for v in now]),
        go.Bar(x=groups, y=later, name=f"Buy in {d} years", marker_color=GREY, text=[f"{sym}{v:,.0f}" for v in later]),
    ])
    fig.update_yaxes(tickprefix=sym, tickformat=",.0f")
    fig.update_layout(barmode="group", hovermode="closest")
    either = r["risk_of_waiting"]["probability_of_developing"]["either"]
    return _layout(fig, f"Yearly premium: now vs in {d} years<br><sup>Chance of developing diabetes or high blood "
                        f"pressure while waiting: {either:.0%}</sup>")


def protection_gauge(r, height=260):
    today, y10 = r["score_today"], r["score_year_10"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=today, number=dict(suffix=" / 100"),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=BLUE, thickness=0.3),
                   steps=[dict(range=[a, b], color=c) for a, b, c in BANDS],
                   threshold=dict(line=dict(color="#475569", width=3), thickness=0.85, value=y10))))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=10),
                      title=dict(text=f"Protection score today<br><sup>Dark line: year 10 ({y10:.0f}) if nothing "
                                      f"changes · weakest area: {r['weakest_area'].replace('_', ' ')}</sup>",
                                 font=dict(size=15)))
    return fig


BUILDERS = {"project_cover_decay": cover_decay, "cost_of_delay": cost_of_delay, "protection_score": protection_gauge}


def figure_for(tool_name, result):
    """Figure for a tool result, or None if the tool has no chart or returned an error."""
    if tool_name not in BUILDERS or "error" in result:
        return None
    return BUILDERS[tool_name](result)
