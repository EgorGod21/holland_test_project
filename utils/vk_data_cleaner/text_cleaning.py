import re

def delete_excess_whitespace(func):
    '''Заменяет множественные пробелы на одиночные'''
    def wrapper(text):
        return re.sub(r'\s+', ' ', func(text))
    return wrapper

@delete_excess_whitespace
def clean_text(text):
    patterns = [
        r'https?://\S+',  # ссылки
        r'[\w\.-]+@[a-zA-Z\d\.-]+\.[a-zA-Z]{2,}',  # email
        r'[a-zA-Z]+\.[a-zA-Z]+/\S+',
        r'[a-zA-Z\d]+\.[a-zA-Z]+',
        r'[a-zA-Z-]+\.[a-zA-Z]+',
        r'#\S+',  # хештеги
        r'[a-z]+\d+',
        r'@[a-zA-Z]+',
        r'\[\S+\]'
    ]

    words = [
        'сдэк', 'почта', 'watsapp', 'wildberries',
        'facebook', 'твиттер', 'twitter', 'телефон',
        'вконтакте', 'boxberry', 'fb', 'instagram', 'telegram'
    ]

    combined = re.compile('|'.join(patterns + words + ['[^а-яёА-ЯЁ]']))
    return re.sub(combined, ' ', text)
