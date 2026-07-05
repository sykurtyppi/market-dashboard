"""Settings page utilities: .env persistence and API-settings lock."""
import hmac
from pathlib import Path

import streamlit as st


def save_to_env(key: str, value: str):
    """Save or update a key-value pair in .env file"""
    env_path = Path('.env')
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines_env = f.readlines()
        
        found = False
        for i, line in enumerate(lines_env):
            if line.startswith(f'{key}='):
                lines_env[i] = f'{key}={value}\n'
                found = True
                break
        
        if not found:
            lines_env.append(f'{key}={value}\n')
        
        with open(env_path, 'w') as f:
            f.writelines(lines_env)
    else:
        with open(env_path, 'w') as f:
            f.write(f'{key}={value}\n')
def render_api_settings_lock() -> bool:
    """
    Protect API key settings behind a password.

    Password source (in order):
    - Streamlit secrets: SETTINGS_PAGE_PASSWORD
    - Environment variable: SETTINGS_PAGE_PASSWORD
    """
    from utils.secrets_helper import get_secret

    settings_password = get_secret("SETTINGS_PAGE_PASSWORD")

    # Defer clearing widget-bound state to a new rerun before widget instantiation.
    if st.session_state.pop("clear_api_settings_password", False):
        st.session_state["api_settings_password_input"] = ""

    # No password configured: keep behavior open, but surface a reminder.
    if not settings_password:
        st.info(
            "API settings are currently open. "
            "Set `SETTINGS_PAGE_PASSWORD` in Streamlit Secrets to lock this section."
        )
        return True

    if st.session_state.get("api_settings_unlocked", False):
        col_info, col_action = st.columns([4, 1])
        with col_info:
            st.success("API settings are unlocked for this session.")
        with col_action:
            if st.button("Lock", key="lock_api_settings_btn"):
                st.session_state["api_settings_unlocked"] = False
                st.rerun()
        return True

    st.warning("API settings are locked.")
    entered = st.text_input(
        "Enter API settings password",
        type="password",
        key="api_settings_password_input",
    )

    if st.button("Unlock API Settings", key="unlock_api_settings_btn"):
        if entered and hmac.compare_digest(entered, settings_password):
            st.session_state["api_settings_unlocked"] = True
            st.session_state["clear_api_settings_password"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False
