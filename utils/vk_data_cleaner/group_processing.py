import re
import ast
from tqdm import tqdm
from .text_cleaning import clean_text

change_dict = {
    'DJ': 'Музыка',
    'R&B': 'Музыка',
    'Rap, Hip-Hop': 'Музыка'
}

def activity_change(activity):
    return change_dict.get(activity, activity)

def process_group_data(df):
    df = df.copy()

    cleaned_items = []

    for i in tqdm(range(len(df))):
        groups = ast.literal_eval(str(df.at[i, 'items']))
        res = []
        for group in groups:
            gr = {}
            name = clean_text(group.get('name', '').lower()).strip()
            activity = activity_change(group.get('activity', ''))
            description = clean_text(group.get('description', '').lower()).strip()
            status = clean_text(group.get('status', '').lower()).strip() if group.get('status') else ''

            if description == '':
                continue

            gr['id'] = group.get('id')
            gr['name'] = name
            gr['activity'] = '' if re.search(r'\d', activity) or activity.startswith('Данный материал заблокирован') else activity
            gr['description'] = description
            gr['status'] = status
            res.append(gr)

        cleaned_items.append(res)

    df['items'] = cleaned_items

    df = df[df['items'].apply(lambda x: len(x) > 0)]
    df = df[df['items'].notna()]
    df.reset_index(drop=True, inplace=True)

    return df
