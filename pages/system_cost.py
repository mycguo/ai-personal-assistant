import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import json
import streamlit as st


# Reusable function for retrieving paginated data from the API
# https://cookbook.openai.com/examples/completions_usage_api
def get_data(url, params):
    # Set up the API key and headers https://platform.openai.com/settings/organization/admin-keys
    OPENAI_ADMIN_KEY = st.secrets["OPENAI_ADMIN_KEY"]

    headers = {
        "Authorization": f"Bearer {OPENAI_ADMIN_KEY}",
        "Content-Type": "application/json",
    }

    # Initialize an empty list to store all data
    all_data = []

    # Initialize pagination cursor
    page_cursor = None

    # Loop to handle pagination
    while True:
        if page_cursor:
            params["page"] = page_cursor

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data_json = response.json()
            all_data.extend(data_json.get("data", []))

            page_cursor = data_json.get("next_page")
            if not page_cursor:
                break
        else:
            print(f"Error: {response.status_code}")
            break

    if all_data:
        print("Data retrieved successfully!")
    else:
        print("Issue: No data available to retrieve.")
    return all_data

# Define the API endpoint
url = "https://api.openai.com/v1/organization/usage/completions"

# Calculate start time: n days ago from now
days_ago = 30
start_time = int(time.time()) - (days_ago * 24 * 60 * 60)

# Define parameters with placeholders for all possible options
params = {
    "start_time": start_time,  # Required: Start time (Unix seconds)
    # "end_time": end_time,  # Optional: End time (Unix seconds)
    "bucket_width": "1d",  # Optional: '1m', '1h', or '1d' (default '1d')
    # "project_ids": ["proj_example"],  # Optional: List of project IDs
    # "user_ids": ["user_example"],     # Optional: List of user IDs
    # "api_key_ids": ["key_example"],   # Optional: List of API key IDs
    # "models": ["o1-2024-12-17", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"],  # Optional: List of models
    # "batch": False,             # Optional: True for batch jobs, False for non-batch
    # "group_by": ["model"],     # Optional: Fields to group by
    "limit": 7,  # Optional: Number of buckets to return, this will chunk the data into 7 buckets
    # "page": "cursor_string"   # Optional: Cursor for pagination
}

def plot_cost():
    usage_data = get_data(url, params)

    print(json.dumps(usage_data, indent=2))

    # Initialize a list to hold parsed records
    records = []

    # Iterate through the data to extract bucketed data
    for bucket in usage_data:
        start_time = bucket.get("start_time")
        end_time = bucket.get("end_time")
        for result in bucket.get("results", []):
            records.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "input_cached_tokens": result.get("input_cached_tokens", 0),
                    "input_audio_tokens": result.get("input_audio_tokens", 0),
                    "output_audio_tokens": result.get("output_audio_tokens", 0),
                    "num_model_requests": result.get("num_model_requests", 0),
                    "project_id": result.get("project_id"),
                    "user_id": result.get("user_id"),
                    "api_key_id": result.get("api_key_id"),
                    "model": result.get("model"),
                    "batch": result.get("batch"),
                }
            )

    # Create a DataFrame from the records
    df = pd.DataFrame(records)

    # Convert Unix timestamps to datetime for readability
    df["start_datetime"] = pd.to_datetime(df["start_time"], unit="s")
    df["end_datetime"] = pd.to_datetime(df["end_time"], unit="s")

    # Reorder columns for better readability
    df = df[
        [
            "start_datetime",
            "end_datetime",
            "start_time",
            "end_time",
            "input_tokens",
            "output_tokens",
            "input_cached_tokens",
            "input_audio_tokens",
            "output_audio_tokens",
            "num_model_requests",
            "project_id",
            "user_id",
            "api_key_id",
            "model",
            "batch",
        ]
    ]

    # Display the DataFrame
    print("displaying the dataframe")
    df.head()

    if not df.empty:
        st.write("### Daily Input vs Output Token Usage Last 30 Days")
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bar charts for input and output tokens
        width = 0.35  # width of the bars
        indices = range(len(df))

        ax.bar(indices, df["input_tokens"], width=width, label="Input Tokens", alpha=0.7)
        ax.bar(
            [i + width for i in indices],
            df["output_tokens"],
            width=width,
            label="Output Tokens",
            alpha=0.7,
        )

        # Set labels and ticks
        ax.set_xlabel("Time Bucket")
        ax.set_ylabel("Number of Tokens")
        ax.set_xticks([i + width / 2 for i in indices])
        ax.set_xticklabels([dt.strftime("%Y-%m-%d") for dt in df["start_datetime"]], rotation=45)
        ax.legend()
        
        # Display the plot using Streamlit
        st.pyplot(fig)
    else:
        st.warning("No data available to plot for this time period.")


    # Calculate start time: n days ago from now
    days_ago = 30
    start_time = int(time.time()) - (days_ago * 24 * 60 * 60)

    # Define the Costs API endpoint
    costs_url = "https://api.openai.com/v1/organization/costs"

    costs_params = {
        "start_time": start_time,  # Required: Start time (Unix seconds)
        "bucket_width": "1d",  # Optional: Currently only '1d' is supported
        "limit": 30,  # Optional: Number of buckets to return
    }

    # Initialize an empty list to store all data
    all_costs_data = get_data(costs_url, costs_params)
    print(json.dumps(all_costs_data, indent=2))


    # Initialize a list to hold parsed cost records
    cost_records = []

    # Extract bucketed cost data from all_costs_data
    for bucket in all_costs_data:
        start_time = bucket.get("start_time")
        end_time = bucket.get("end_time")
        for result in bucket.get("results", []):
            cost_records.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "amount_value": result.get("amount", {}).get("value", 0),
                    "currency": result.get("amount", {}).get("currency", "usd"),
                    "line_item": result.get("line_item"),
                    "project_id": result.get("project_id"),
                }
            )

    # Create a DataFrame from the cost records
    cost_df = pd.DataFrame(cost_records)

    # Convert Unix timestamps to datetime for readability
    cost_df["start_datetime"] = pd.to_datetime(cost_df["start_time"], unit="s")
    cost_df["end_datetime"] = pd.to_datetime(cost_df["end_time"], unit="s")

    st.write("### Cost of the System by Day Last 30 Days")
    if not cost_df.empty:
        # Ensure datetime conversion for 'start_datetime' column
        if (
            "start_datetime" not in cost_df.columns
            or not pd.api.types.is_datetime64_any_dtype(cost_df["start_datetime"])
        ):
            cost_df["start_datetime"] = pd.to_datetime(
                cost_df["start_time"], unit="s", errors="coerce"
            )

        # Create a new column for just the date part of 'start_datetime'
        cost_df["date"] = cost_df["start_datetime"].dt.date

        # Group by date and sum the amounts
        cost_per_day = cost_df.groupby("date")["amount_value"].sum().reset_index()

        # Plot the data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            cost_per_day["date"],
            cost_per_day["amount_value"],
            width=0.6,
            color="skyblue",
            alpha=0.8,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Cost (USD)")
        ax.set_title("Total Cost per Day (Last 30 Days)")
        ax.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No cost data available to plot.")

def main():
    st.title("Showing Cost of the System")
    plot_cost()

if __name__ == "__main__":
    main()