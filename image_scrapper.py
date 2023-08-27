from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time
import os

# scroll down function
def scroll_down(wd, delay):

    wd.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    time.sleep(delay)

    return

def get_google_images(delay, url, max_images):

    wd = webdriver.Chrome()
    wd.get(url)

    time.sleep(3)
    wd.find_element(By.XPATH, "//span[contains(text(), 'Accept all')]").click()

    image_urls = set()
    skips = 0

    while len(image_urls) + skips < max_images:
        scroll_down(wd, delay)
        thumbnails = wd.find_elements(By.CLASS_NAME, 'Q4LuWd')

        for img in thumbnails[len(image_urls) + skips: max_images]:
            try:
                img.click()
                time.sleep(delay)

            except:
                continue

            images = wd.find_elements(By.CLASS_NAME, 'iPVvYb')

            for image in images:
                if image.get_attribute('src') in image_urls:
                    max_images += 1
                    skips += 1
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
        file_path = path + file_name

        # with open(file_path, 'wb') as f:
        #     image.save(f, 'JPEG')

        image.save(path + file_name + '.jpg')

        print('success')

    except Exception as e:
        print('FAIL: ', e)

    wd.quit()

    return


# setup arguments
path = os.getcwd() + '\\data\\dogs\\'
url = 'https://www.google.com/search?q=dog+pictures&tbm=isch&ved=2ahUKEwiS3e-a4vWAAxVz-IsKHVOqCccQ2-cCegQIABAA&oq=dog+pictures&gs_lcp=CgNpbWcQAzIHCAAQigUQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABFAAWABguwZoAHAAeACAAVGIAVGSAQExmAEAqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=OIznZJLhKfPwrwTT1Ka4DA&bih=955&biw=1745'
max_images = 100

st = time.time()
urls = get_google_images(delay=1, url=url, max_images=max_images)
print(f'found {max_images} images in {(time.time() - st) / 60:.2f} min')

st_global = time.time()
for i, url in enumerate(urls):

    st_local = time.time()
    download_image(path, url, f'dog_{i}')
    print(f'downloaded image #{i} in {time.time() - st_local:.1f} s, total time {(time.time() - st_global) / 60:.2f} min')



# time needed for 10k images (assuming it wont break)
# finding 100 images 3.5 mins (need to fix load more images)
# finding 10k images ~~ 6h
# downloading an image ~ 8s
# downloading 10k images ~~ 22 - 23h
# total time around 30h ...

# fix show more results
# save links after finding them
# implement saving the state of downloading the links and resuming




