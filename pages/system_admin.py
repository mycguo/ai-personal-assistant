import streamlit as st
from app import download_faiss_from_s3, download_s3_bucket   
from pages.app_admin import upload_vector_store_to_s3


bucket_name = st.secrets["BUCKET_NAME"]

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
        st.header("System Admin Only: Danger Zone")
        if st.button("Upload Vector Store to S3"):
            with st.spinner("Uploading to S3..."):
                upload_vector_store_to_s3()
                st.success("Uploaded to S3!")

        if st.button("Download Vector Store from S3"):
            with st.spinner("Downloading from S3..."):
                download_s3_bucket(bucket_name, "faiss_download")
                st.success("Downloaded file from S3!")

        if st.button("Download Vector Store from S3 and overwrite the local index"):
            with st.spinner("Downloading from S3..."):
                download_s3_bucket(bucket_name, "faiss_index")
                st.success("Downloaded file from S3!")


        st.button("Log out", on_click=st.logout)

if __name__ == "__main__":
    main()