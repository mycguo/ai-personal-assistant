import streamlit as st
from streamlit.errors import StreamlitAuthError, StreamlitSecretNotFoundError
from file_manager import clear_files, load_files


def _get_auth_state():
    """Return (auth_available, is_logged_in, display_name)."""
    if hasattr(st, "user"):
        user = st.user
        return True, bool(getattr(user, "is_logged_in", False)), getattr(user, "name", "User")

    if hasattr(st, "experimental_user"):
        user = st.experimental_user
        return True, bool(getattr(user, "is_logged_in", False)), getattr(user, "name", "User")

    return False, False, "Admin"


def _login_with_google():
    try:
        auth = st.secrets.get("auth", {})
        if "google" in auth:
            st.login("google")
        else:
            st.login()
    except (StreamlitAuthError, StreamlitSecretNotFoundError):
        st.session_state["admin_login_error"] = True


def login_screen():
    st.header("This is for system admin only. Please login first")
    st.subheader("Please log in.")
    if st.session_state.get("admin_login_error"):
        st.error(
            "Google login is not configured correctly. The app owner must check "
            "the authentication settings in Streamlit Cloud's Manage app → Settings → Secrets. "
            "See the README for the required configuration."
        )
    if hasattr(st, "login"):
        st.button("Log in with Google", on_click=_login_with_google)
    else:
        st.warning("Streamlit authentication is unavailable in this deployment.")



def main():
    auth_available, is_logged_in, display_name = _get_auth_state()

    if not auth_available or not is_logged_in:
        login_screen()
    else:
        if auth_available:
            st.header(f"Welcome, {display_name}!")

        st.title("Knowledge Assistant System Admin")
        st.header("System Admin Maintenance")
        st.info("This app now uses Gemini File API instead of local vector store/S3 sync.")

        current_files = load_files()
        st.write(f"Tracked uploaded files: {len(current_files)}")

        if current_files:
            with st.expander("View tracked files"):
                for f in current_files:
                    st.write(f"- {f.get('name', 'unknown')} ({f.get('mime_type', 'unknown')})")

        st.header("Danger Zone")
        if st.button("Clear Local File Registry"):
            clear_files()
            st.success("Local file registry cleared.")
            st.rerun()

        if auth_available and hasattr(st, "logout"):
            st.button("Log out", on_click=st.logout)

if __name__ == "__main__":
    main()
