import streamlit as st
from file_manager import clear_files, load_files

def login_screen():
    st.header("This is for system admin only. Please login first")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)



def main():
    if not st.experimental_user.is_logged_in:
        login_screen()
    else:
        st.header(f"Welcome, {st.experimental_user.name}!")
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

        st.button("Log out", on_click=st.logout)

if __name__ == "__main__":
    main()
