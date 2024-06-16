import pandas as pd
import google.generativeai as genai
file_path = r"D:\project\Pre-Election-Analysis-of-the-2024-Presidential-Election-using-the-PySpark-Framework\com_comment_df.csv"
df = pd.read_csv(file_path, encoding='utf-8')
df = df[df["content"].str.contains('柯文哲|賴清德|侯友宜')][["content"]]
df = df[~df["content"].isin(["侯友宜", "賴清德", "柯文哲"])]
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]


# 初始化你的模型
api_key = "AIzaSyCzZWfN9nNWWVPhcjmWaQ5sk3srdZBxkcY"
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)

# 假設你的DataFrame叫做df，並且包含評論的欄位叫做"Comment"
comments = df['content']

# 定義所有問題
questions = [
    "對此評論打一個情感分數正面、負面、中性:{}".format(comment) for comment in comments
]

# 發送每個問題的請求並獲取回答
#  genai限制每分鐘60 token
responses = []
batch_size = 50
import time
for i in range(0, len(questions), batch_size):
    batch_questions = questions[i:i + batch_size]
    batch_responses = [model.generate_content(question, safety_settings=safety_settings).text for question in batch_questions]
    responses.extend(batch_responses)
    

    time.sleep(60)

# 將回答新增到DataFrame中
df['Sentiment'] = responses
df.to_csv("Sentiment_df.csv", index=False)



df = pd.read_csv(file_path, encoding='utf-8')
df = df[df["content"].str.contains('柯文哲|賴清德|侯友宜')]
df = df[~df["content"].isin(["侯友宜", "賴清德", "柯文哲"])]
df.drop(columns=['content'], inplace=True)
sentiment_df=  pd.read_csv("D:\project\PreElectionWebSentimentAnalysis\Sentiment_df.csv")

sentiment_df.loc[sentiment_df['Sentiment'].str.contains('正面'), 'Sentiment'] = '正面'
sentiment_df.loc[sentiment_df['Sentiment'].str.contains('正向'), 'Sentiment'] = '正面'
sentiment_df.loc[sentiment_df['Sentiment'].str.contains('負面'), 'Sentiment'] = '負面'
sentiment_df.loc[sentiment_df['Sentiment'].str.contains('负面'), 'Sentiment'] = '負面'
sentiment_df.loc[sentiment_df['Sentiment'].str.contains('中性'), 'Sentiment'] = '中性'
sentiment_df.loc[sentiment_df['Sentiment'].str.contains('中立'), 'Sentiment'] = '中性'

df.reset_index(drop=True, inplace=True)
sentiment_df.reset_index(drop=True, inplace=True)
result = pd.concat([sentiment_df, df], axis=1)

invalid_sentiments = result[~result['Sentiment'].isin(['正面', '負面', '中性'])]

result = result[result['Sentiment'].isin(['正面', '負面', '中性'])]

# sentiment_df = sentiment_df[sentiment_df["Sentiment"] != 其他]
# 顯示符合條件的行


result.to_csv("com_sentiment_df.csv", index=False)

