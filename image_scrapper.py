from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time
import os

# setup paths
os.getcwd()
path = os.getcwd() + '\\data\\dogs\\'
path_chromedriver = path + 'chromedriver-win64\\chromedriver.exe'

# webdriver
# wd = webdriver.Chrome(path_chromedriver)
wd = webdriver.Chrome()

# scroll down function
def scroll_down(wd, delay):

    wd.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    time.sleep(delay)

    return

def get_google_images(wd, delay, max_images):

    url = 'https://www.google.com/search?q=dog+pictures&tbm=isch&ved=2ahUKEwiS3e-a4vWAAxVz-IsKHVOqCccQ2-cCegQIABAA&oq=dog+pictures&gs_lcp=CgNpbWcQAzIHCAAQigUQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABFAAWABguwZoAHAAeACAAVGIAVGSAQExmAEAqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=OIznZJLhKfPwrwTT1Ka4DA&bih=955&biw=1745'
    wd.get(url)

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

            images = wd.find_elements(By.CLASS_NAME, 'n3VNCb')

            for image in images:
                if image.get_attribute('src') in image_urls:
                    max_images += 1
                    skips += 1
                    break

                if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                    image_urls.add(image.get_attribute('src'))
                    print(f'found {len(image_urls)}')

    return image_urls

def download_image(path, url, file_name):

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

# urls = get_google_images(wd, delay=1, max_images=max_images)

for i, url in enumerate(urls):
    download_images(path, url, str(i) + '.jpg', max_images)

url = 'https://www.google.com/search?newwindow=1&sca_esv=559765737&sxsrf=AB5stBjz837urzE56V9afUz3Fimc1LsdZw:1692896310583&q=dog+pictures&tbm=isch&source=lnms&sa=X&ved=2ahUKEwiDhO-Z4vWAAxU_yLsIHQJxBoAQ0pQJegQIDhAB&biw=1745&bih=955&dpr=1.1'
url = 'https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*'
urls = [url]


download_image(path, url, 'dog')

























delay = 1
max_images = 3
url = 'https://www.google.com/search?q=dog+pictures&tbm=isch&ved=2ahUKEwiS3e-a4vWAAxVz-IsKHVOqCccQ2-cCegQIABAA&oq=dog+pictures&gs_lcp=CgNpbWcQAzIHCAAQigUQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABFAAWABguwZoAHAAeACAAVGIAVGSAQExmAEAqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=OIznZJLhKfPwrwTT1Ka4DA&bih=955&biw=1745'
wd.get(url)

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

        images = wd.find_elements(By.CLASS_NAME, 'n3VNCb')

        for image in images:
            if image.get_attribute('src') in image_urls:
                max_images += 1
                skips += 1
                break

            if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                image_urls.add(image.get_attribute('src'))
                print(f'found {len(image_urls)}')






wd = webdriver.Chrome()



url = 'https://www.google.com/search?q=dog+pictures&tbm=isch&ved=2ahUKEwiS3e-a4vWAAxVz-IsKHVOqCccQ2-cCegQIABAA&oq=dog+pictures&gs_lcp=CgNpbWcQAzIHCAAQigUQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABFAAWABguwZoAHAAeACAAVGIAVGSAQExmAEAqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=OIznZJLhKfPwrwTT1Ka4DA&bih=955&biw=1745'
wd.get(url)
# notification = wd.find_elements(By.CLASS_NAME, 'VfPpkd-vQzf8d')
# notification = wd.find_elements(By.CLASS_NAME, 'vQzf8d')
wd.switchTo().frame('Jh91Gc')
wd.find_element(By.ID('id')).click()
# notification = wd.find_element(By.CLASS_NAME, 'VfPpkd-Jh91Gc')
# notification.click()




wd.quit()


len(image_urls) + skips < max_images
scroll_down(wd, delay)
thumbnails = wd.find_elements(By.CLASS_NAME, 'Q4LuWd')













