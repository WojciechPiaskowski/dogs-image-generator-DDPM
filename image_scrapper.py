from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests
import io
from PIL import Image
import time
import os
import pandas as pd

def load_create_status():

    try:
        df = pd.read_csv(os.getcwd() + '\\data\\download_status.csv')

    except:

        breeds = [
        'french bulldogs',
        'labrador retrievers',
        'golden retrievers',
        'german shepherd',
        'poodles',
        'bulldogs',
        'rottweilers',
        'beagles',
        'dachshunds',
        'german shorthaired pointers',
        'pembroke welsh corgis',
        'australian shepherds',
        'yorkshire terriers',
        'cavalier king charles spaniels',
        'doberman pinschers',
        'boxers',
        'miniature schnauzers',
        'cane corso',
        'great danes',
        'shih tzu',
        'siberian huskies',
        'bernese mountain dogs',
        'pomeranians',
        'boston terriers',
        'havanese']

        breeds = [breed + ' dog pictures' for breed in breeds]
        d = {'breed': breeds}
        df = pd.DataFrame(d)
        df['downloaded'] = 0
        df.to_csv(os.getcwd() + '\\data\\download_status.csv', index=False)

    return df

def scroll_to_bottom(wd):
    last_height = wd.execute_script('\
    return document.body.scrollHeight')

    while True:
        wd.execute_script('\
        window.scrollTo(0,document.body.scrollHeight)')

        # waiting for the results to load
        # Increase the sleep time if your internet is slow
        time.sleep(1)

        new_height = wd.execute_script('\
        return document.body.scrollHeight')

        # click on "Show more results" (if exists)
        try:
            wd.find_elements(By.CLASS_NAME, 'LZ4I')[0].click()

            # waiting for the results to load
            # Increase the sleep time if your internet is slow
            time.sleep(1)

        except:
            pass

        # checking if we have reached the bottom of the page
        if new_height == last_height:
            break

        last_height = new_height

    return

def get_google_images(delay, search_label):

    wd = webdriver.Chrome()
    wd.get('https://www.google.com/imghp?hl=en')

    time.sleep(1)
    wd.find_element(By.CLASS_NAME, 'sy4vM').click()

    time.sleep(1)
    input_bar = wd.find_element(By.ID, 'APjFqb')
    time.sleep(1)
    input_bar.send_keys(search_label)
    time.sleep(1)
    input_bar.send_keys(Keys.ENTER)

    image_urls = set()
    scroll_to_bottom(wd)

    thumbnails = wd.find_elements(By.CLASS_NAME, 'Q4LuWd')

    for img in thumbnails:
        try:
            img.click()
            time.sleep(delay)

        except:
            continue

        images = wd.find_elements(By.CLASS_NAME, 'iPVvYb')

        for image in images:
            if image.get_attribute('src') in image_urls:
                break

            if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                image_urls.add(image.get_attribute('src'))
                print(f'found {len(image_urls)}')

    wd.quit()

    return image_urls

def download_image(path, url, file_name):

    wd = webdriver.Chrome()

    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)

        image.save(path + file_name + '.jpg')

        print('success')

    except Exception as e:
        print('FAIL: ', e)

    wd.quit()

    return


# get status info
df = load_create_status()
search_labels = list(df[df['downloaded'] == 0]['breed'])

# setup arguments
path = os.getcwd() + '\\data\\dogs\\'

for breed in search_labels:

    st = time.time()
    urls = get_google_images(delay=1, search_label=breed)
    print(f'found {len(urls)} images in {(time.time() - st) / 60:.2f} min')

    st_global = time.time()
    for i, url in enumerate(urls):

        st_local = time.time()
        download_image(path, url, f'{breed}_{i}')
        print(f'downloaded image #{i} in {time.time() - st_local:.1f} s, total time {(time.time() - st_global) / 60:.2f} min')

    df[df['breed'] == breed]['downloaded'] = 1
    df.to_csv(os.getcwd() + '\\data\\download_status.csv', index=False)





