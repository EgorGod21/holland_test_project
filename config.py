from dotenv import load_dotenv
import os

load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
GIGACHAT_TOKEN = os.getenv("GIGACHAT_TOKEN")
ACCESS_TOKENS = os.getenv("ACCESS_TOKENS").split(",")
VK_API_VERSION = os.getenv("VK_API_VERSION")
