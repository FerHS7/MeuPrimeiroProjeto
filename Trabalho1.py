import requests
import pandas as pd
from bs4 import BeautifulSoup
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
response = requests.get(static_url)
soup = BeautifulSoup(response.content, 'html.parser')
print(soup.title)