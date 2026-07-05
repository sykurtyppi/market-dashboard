"""Economic Calendar page."""
from datetime import datetime

import pandas as pd
import streamlit as st

from data_collectors.economic_calendar_collector import EconomicCalendarCollector


def render(components):
    st.markdown("<h1 class='main-header'>Economic Calendar & Events</h1>", unsafe_allow_html=True)

    st.markdown("""
    Track upcoming economic events that can move markets. FOMC, CPI, NFP, and more.
    """)

    @st.cache_resource
    def get_calendar_collector():
        return EconomicCalendarCollector()

    calendar = get_calendar_collector()

    # --- NEXT EVENT COUNTDOWN ---
    countdown = calendar.get_countdown_to_next_event()

    if countdown:
        st.subheader("Next Major Event")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {countdown['event']}")
            st.markdown(f"*{countdown['description']}*")
        with col2:
            if countdown['is_today']:
                st.error("TODAY")
            elif countdown['is_tomorrow']:
                st.warning("TOMORROW")
            else:
                st.info(f"{countdown['days']} days away")
        with col3:
            st.markdown(f"**Date:** {countdown['date'].strftime('%b %d, %Y')}")
            st.markdown(f"**Category:** {countdown['category']}")

    st.divider()

    # --- CALENDAR SUMMARY ---
    summary = calendar.get_calendar_summary()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Events This Week", summary.get('this_week', 0))
    with col2:
        st.metric("High Impact (30d)", summary.get('high_importance', 0))
    with col3:
        next_fomc = summary.get('next_fomc')
        if next_fomc:
            days_to_fomc = (next_fomc - datetime.now()).days
            st.metric("Days to FOMC", days_to_fomc)
        else:
            st.metric("Days to FOMC", "N/A")
    with col4:
        next_cpi = summary.get('next_cpi')
        if next_cpi:
            days_to_cpi = (next_cpi - datetime.now()).days
            st.metric("Days to CPI", days_to_cpi)
        else:
            st.metric("Days to CPI", "N/A")

    st.divider()

    # --- UPCOMING EVENTS ---
    st.subheader("Upcoming Events (30 Days)")

    events = calendar.get_upcoming_events(days=30)

    if events:
        # Create DataFrame for display
        events_data = []
        for e in events:
            if e.date > datetime.now():
                days_away = (e.date - datetime.now()).days
                events_data.append({
                    'Date': e.date.strftime('%b %d'),
                    'Day': e.date.strftime('%a'),
                    'Event': e.name,
                    'Category': e.category,
                    'Importance': e.importance.value.upper(),
                    'Days Away': days_away,
                })

        events_df = pd.DataFrame(events_data)

        # Color code by importance with dark-mode friendly colors
        def highlight_importance(row):
            if row['Importance'] == 'HIGH':
                # Red tint with dark text
                return ['background-color: #5c2a2a; color: #ff8a80'] * len(row)
            elif row['Importance'] == 'MEDIUM':
                # Orange/amber tint with dark text
                return ['background-color: #4a3c1a; color: #ffd54f'] * len(row)
            # Low importance - subtle gray
            return ['background-color: #2d2d2d; color: #b0b0b0'] * len(row)

        styled_df = events_df.style.apply(highlight_importance, axis=1).set_properties(**{
            'font-size': '14px',
            'padding': '8px',
        })

        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            height=400
        )

        # This week detail
        this_week = calendar.get_events_this_week()
        if this_week:
            with st.expander("This Week's Events"):
                for e in this_week:
                    importance_color = '#F44336' if e.importance.value == 'high' else '#FF9800'
                    st.markdown(f"""
                    - **{e.date.strftime('%a %b %d')}**: {e.name}
                      <span style='color:{importance_color}'>({e.importance.value.upper()})</span>
                      - {e.description}
                    """, unsafe_allow_html=True)
    else:
        st.info("No upcoming events found")

    st.divider()

    # --- EVENT IMPACT GUIDE ---
    with st.expander("Event Impact Guide"):
        st.markdown("""
        ### High Impact Events

        | Event | Typical Impact | What to Watch |
        |-------|---------------|---------------|
        | **FOMC** | High volatility | Rate decision, dot plot, Powell comments |
        | **CPI** | Major moves | Core CPI vs expectations, shelter costs |
        | **NFP** | Morning volatility | Headline jobs, wage growth, revisions |
        | **GDP** | Moderate | Growth rate vs estimates, consumer spending |

        ### Trading Around Events

        **Pre-Event:**
        - VIX9D often spikes (near-term uncertainty)
        - Options premiums inflate
        - Consider reducing position size

        **Post-Event:**
        - Initial reaction often reversed
        - Wait 30-60 min for dust to settle
        - Watch for trend continuation or reversal

        ### Key Dates Pattern

        - **FOMC**: 8 meetings/year, decisions at 2:00 PM ET
        - **CPI**: Monthly, ~10th-14th, 8:30 AM ET
        - **NFP**: First Friday of month, 8:30 AM ET
        - **GDP**: Quarterly, end of month following quarter
        """)
