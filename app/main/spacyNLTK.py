from spacy.tokens import Doc, Span, Token
from spacy.language import Language

from nltk.sentiment.vader import SentimentIntensityAnalyzer

@Language.factory("spacyVader")
class SpacyVader(object):
    """A spacy pipline for NLTK Vader Sentiment Analysis"""
    
    def __init__(self, nlp, name):
        extensions = ["polarity"]
        getters = [self.get_polarity]
        for ext, get in zip(extensions, getters):
            if not Doc.has_extension(ext):
                Doc.set_extension(ext, default=None)
            if not Span.has_extension(ext):
                Span.set_extension(ext, getter=get)
            if not Token.has_extension(ext):
                Token.set_extension(ext, getter=get)
    
    def __call__(self, doc):
        # Doc-level sentiment
        sentiment = self.get_sentiment(doc)
        doc._.set("polarity", sentiment['compound'])

        return doc

    def get_sentiment(self, doc):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(doc.text)
        return sentiment
    
    def get_polarity(self, doc):
        return self.get_sentiment(doc).polarity