import streamlit as st

def login_screen():
    st.header("This is for system admin only. Please login first")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)



def main():
    if not st.experimental_user.is_logged_in:
        login_screen()
    else:
        st.header(f"Welcome, {st.experimental_user.name}!")
        st.title("Knowledge Assistant System Admin function")
        st.header("System Admin Only: Danger Zone")
        st.button("Reset the DB")
        st.button("Log out", on_click=st.logout)

if __name__ == "__main__":
    main()