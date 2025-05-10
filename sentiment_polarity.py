from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
sentence = "I love learning about NLP! It's fascinating."
score = sia.polarity_scores(sentence)
print(score)
