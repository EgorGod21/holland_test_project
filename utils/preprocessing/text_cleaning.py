from spacy.lang.ru.stop_words import STOP_WORDS
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def text_preprocess_natasha(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    filtered_tokens = [
        token.text
        for token in doc.tokens
        if token.lemma not in STOP_WORDS and len(token.lemma) > 2
    ]
    return ' '.join(filtered_tokens)
