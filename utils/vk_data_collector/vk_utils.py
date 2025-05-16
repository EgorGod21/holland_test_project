import requests
from vk_api import VkApi

def get_vk_session(token):
    session = VkApi(token=token)
    return session.get_api()

def get_user_info(user_id, vk):
    user = vk.users.get(user_ids=user_id, fields=['sex', 'is_closed', 'deactivated'])[0]
    return {
        'sex': user.get('sex'),
        'is_closed': user.get('is_closed'),
        'deactivated': user.get('deactivated', False)
    }

def get_groups_items(user_id, token, vk_api_version):
    url = f'https://api.vk.com/method/groups.get'
    params = {
        'user_id': user_id,
        'access_token': token,
        'v': vk_api_version,
        'fields': 'activity,name,screen_name,members_count,description,status,type,fixed_post',
        'extended': 1
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "response" in data:
        items = data["response"]["items"]
        return [
            {k: v for k, v in item.items() if k not in ['photo_50', 'photo_100', 'photo_200'] and item['is_closed'] == 0}
            for item in items if item['is_closed'] == 0
        ]
    elif "error" in data and data['error']['error_code'] == 29:
        print(f"Rate limit error for user {user_id}")
        return data
    return []
