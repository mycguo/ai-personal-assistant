import requests
import pandas as pd
import random


# --------------------Transcription Code for English Language---------------------


def transcribe_english(file, token):
    headers = {"authorization": token}
    url_response = requests.post(
        "https://api.assemblyai.com/v2/upload", headers=headers, data=file
    )

    url = url_response.json()["upload_url"]

    id_endpoint = "https://api.assemblyai.com/v2/transcript"

    json = {
        "audio_url": url,
        "speaker_labels": True,
        "auto_highlights": True,
        "iab_categories": True,
        "sentiment_analysis": True,
        "auto_chapters": True,
        "entity_detection": True,
    }

    headers = {"authorization": token, "content-type": "application/json"}
    id_response = requests.post(id_endpoint, json=json, headers=headers)
    transcribe_id = id_response.json()["id"]
    text_endpoint = "https://api.assemblyai.com/v2/transcript/" + transcribe_id
    headers = {
        "authorization": token,
    }
    result = requests.get(text_endpoint, headers=headers).json()

    while result.get("status") != "completed":
        print("processing")
        if result.get("status") == "error":
            print("error")
            return "error"
        text_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcribe_id}"
        headers = {"authorization": token}
        result = requests.get(text_endpoint, headers=headers).json()
    print("completed")
    return result


# ------------------Extrcting Data from AssemblyAI's JSON response-------------------


def json_data_extraction(result):
    print("Starting Data Extraction")
    audindex = pd.json_normalize(result["words"])
    chapters = pd.json_normalize(result["chapters"])
    # topics = pd.json_normalize(result["iab_categories_result"]["results"])
    # topics["label_1"] = topics["labels"].apply(lambda x: x[0]["label"])
    # topics["label_2"] = topics["labels"].apply(
    #     lambda x: x[1]["label"] if len(x) > 1 else "none"
    # )
    # topics["label_3"] = topics["labels"].apply(
    #     lambda x: x[2]["label"] if len(x) > 2 else "none"
    # )
    highlights = pd.json_normalize(result["auto_highlights_result"]["results"])
    highlights = highlights.text.unique()
    audindex["summary"] = ""
    audindex["headline"] = ""
    audindex["gist"] = ""
    speakers = list(audindex.speaker)
    previous_speaker = "A"
    l = len(speakers)
    i = 0
    speaker_seq_list = list()
    for index, new_speaker in enumerate(speakers):
        if index > 0:
            previous_speaker = speakers[index - 1]
        if new_speaker != previous_speaker:
            i += 1
        speaker_seq_list.append(i)
    audindex["sequence"] = speaker_seq_list

    for j in range(0, len(chapters)):
        for i in range(0, len(audindex)):
            if (
                audindex.iloc[i]["start"] >= chapters.iloc[j]["start"]
                and audindex.iloc[i]["end"] <= chapters.iloc[j]["end"]
            ):
                audindex.loc[i, "summary"] = chapters.iloc[j]["summary"]
                audindex.loc[i, "headline"] = chapters.iloc[j]["headline"]
                audindex.loc[i, "gist"] = chapters.iloc[j]["gist"]

    # for j in range(0, len(topics)):
    #     try:
    #         for i in range(0, len(audindex)):
    #             if (
    #                 audindex.iloc[i]["start"] >= topics.iloc[j]["timestamp.start"]
    #                 and audindex.iloc[i]["end"] <= topics.iloc[j + 1]["timestamp.start"]
    #             ):
    #                 audindex.loc[i, "label_1"] = topics.iloc[j]["label_1"]
    #                 audindex.loc[i, "label_2"] = topics.iloc[j]["label_2"]
    #                 audindex.loc[i, "label_3"] = topics.iloc[j]["label_3"]
    #     except:
    #         for i in range(0, len(audindex)):
    #             if audindex.iloc[i]["start"] >= topics.iloc[j]["timestamp.start"]:
    #                 audindex.loc[i, "label_1"] = topics.iloc[j]["label_1"]
    #                 audindex.loc[i, "label_2"] = topics.iloc[j]["label_2"]
    #                 audindex.loc[i, "label_3"] = topics.iloc[j]["label_3"]

    group = [
        "speaker",
        "summary",
        "headline",
        "gist",
        "sequence",
        # "label_1",
        # "label_2",
        # "label_3",
    ]
    df = pd.DataFrame(
        audindex.groupby(group).agg(
            utter=("text", " ".join),
            start_time=("start", "min"),
            end_time=("end", "max"),
        )
    )
    df.reset_index(inplace=True)
    df["key_phrase"] = "none"
    for x in highlights:
        df.loc[(df.utter.str.contains(x)), "key_phrase"] = x
    df.sort_values(by=["start_time"], inplace=True)
    return df


def start_transcription(file, tokens):
    fname = file.name
    fileData = file.read()
    token = tokens[random.randint(0, len(tokens) - 1)]
    print(token)
    print("transcribing...")
    print(fname)
    result = transcribe_english(fileData, token)
    print("transcription completed")
    if result == "error":
        print("Transcription Error")
    print("transcription completed")
    df = json_data_extraction(result)
    data = df[
        [
            "speaker",
            "utter",
            "summary",
            "headline",
            "key_phrase",
            "start_time",
            "end_time",
        ]
    ]
    return data
