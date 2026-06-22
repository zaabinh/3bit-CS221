import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import contractions
except ImportError:
    contractions = None

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError as error:
    raise RuntimeError(
        "Missing NLTK stopwords data. Run: python -m nltk.downloader stopwords wordnet"
    ) from error

LEMMATIZER = WordNetLemmatizer()


def _expand_contractions(text: str) -> str:
    if contractions is not None:
        return contractions.fix(text)

    replacements = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    for contraction, expansion in replacements.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


def preprocess_review(review: str) -> str:
    if not isinstance(review, str):
        raise TypeError("Review must be a string")

    review = review.strip()
    if not review:
        raise ValueError("Review cannot be empty")

    review = _expand_contractions(review)
    review = review.lower()
    review = re.sub(r"[^a-zA-Z0-9\s]", " ", review)
    review = re.sub(r"\s+", " ", review).strip()
    review = " ".join(
        word for word in review.split()
        if word not in STOP_WORDS
    )
    review = " ".join(
        LEMMATIZER.lemmatize(word)
        for word in review.split()
    )

    return review
