import requests
import praw
from transformers import pipeline
from collections import defaultdict

# -------- CONFIG --------
NEWS_API_KEY = "YOUR_NEWS_API_KEY"
REDDIT_CLIENT_ID = "YOUR_REDDIT_ID"
REDDIT_SECRET = "YOUR_REDDIT_SECRET"
REDDIT_AGENT = "mood-mapper/1.0"

# -------- MODELS --------
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# -------- DATA SOURCES --------
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=10&apiKey={NEWS_API_KEY}"
    data = requests.get(url).json()
    return [a["title"] for a in data.get("articles", [])]

def fetch_reddit():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_AGENT
    )
    texts = []
    for post in reddit.subreddit("worldnews").hot(limit=10):
        texts.append(post.title)
    return texts

# -------- EMOTION ENGINE --------
def analyze_emotions(texts):
    scores = defaultdict(float)
    for text in texts:
        results = emotion_classifier(text)[0]
        for r in results:
            scores[r["label"]] += r["score"]
    total = sum(scores.values())
    return {k: round(v / total, 2) for k, v in scores.items()}

# -------- SIGNAL  --------
def generate_signal(emotions):
    if emotions.get("fear", 0) > 0.4:
        return "High anxiety detected. Avoid risky decisions."
    if emotions.get("joy", 0) > 0.35:
        return "Positive sentiment spike. Good time to launch."
    return "Neutral emotional state. Maintain steady operations."

# -------- MAIN --------
def run():
    texts = fetch_news() + fetch_reddit()
    emotions = analyze_emotions(texts)
    signal = generate_signal(emotions)

    print("\n🌍 Global Emotional Snapshot")
    print(emotions)
    print("\n📢 Decision Signal:")
    print(signal)

if __name__ == "__main__":
    run()
