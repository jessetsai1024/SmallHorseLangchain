from youtube_transcript_api import YouTubeTranscriptApi

# 視頻ID
video_id = "cAOiiVYUAC8"

# 列出可用字幕
transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
print(transcript_list)

# 獲取字幕
transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
# print(transcript)
content = ""
for item in transcript:
    # print(item['text'])
    if item['text'] != "[Music]":
        content += item['text'] + " "

print(content)

# 翻譯字幕 
transcript = transcript_list.find_transcript(['en'])
translated_transcript = transcript.translate('zh-Hant')


content = ""
for item in translated_transcript.fetch():
    if item['text'] != "[音樂]":
        content += item['text'] + ""
        
print(content)