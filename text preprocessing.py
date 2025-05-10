import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

text = "Hello! This is an example sentence, showing off the stop words filtration."

# Lowercase
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Remove stopwords
stop_words = set(stopwords.words('english'))
words = text.split()
filtered_words = [word for word in words if word not in stop_words]

print(filtered_words)
