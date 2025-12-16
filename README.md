# 🌍 Internet Mood → Real-World Signal Mapper

A Python tool that aggregates global internet sentiment from **news headlines** and **Reddit posts**, classifies emotions using a **transformer-based AI model**, and generates actionable **decision signals** for businesses or creators.  

Unlike traditional sentiment analyzers, this project synthesizes **multi-source emotional data** into a daily “emotional fingerprint,” helping you understand the **global mood** and make informed decisions.

---

## Features

- Fetches **top news headlines** via NewsAPI
- Scrapes **hot Reddit posts** from `r/worldnews` (or any subreddit)
- Uses **HuggingFace transformer model** for multi-emotion classification:
  - joy, fear, anger, sadness, surprise, trust, etc.
- Aggregates scores across sources into a **weighted emotional snapshot**
- Generates **decision signals** based on dominant emotions
- CLI-based, lightweight, and easily extendable
- Can be turned into **daily mood tracker or automated alerts**

---

## Tech Stack

- **Python 3.10+**
- **Requests** – for HTTP requests
- **PRAW** – Reddit API client
- **Transformers + PyTorch** – for emotion classification
- **NewsAPI** – for fetching headlines

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/rahimprz/Mood-Mapper.git
cd internet-mood-mapper

