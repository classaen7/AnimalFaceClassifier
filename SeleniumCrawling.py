from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import time
import urllib.request
import os

dir = {
    "강아지" : ["박보영","아이유","강다니엘","송중기"],
    "고양이" : ["강동원", "이준기","한예슬","뉴진스 해린"],
    "토끼" : ["정국", "워너원 박지훈","트와이스 나연","장원영" ],
    "공룡" : ["김우빈", "공유","송지효","신민아"],
    "곰" : ["안재홍", "김대명","레드벨벳 슬기","엔믹스 오해원"]     
}



defualt_path = 'C:/Users/CHOI/OneDrive/바탕 화면/산학프로젝트 인공지능/데이터 크롤링/'

for animal in dir:
    animal_path = defualt_path + animal
    if not os.path.isdir(animal_path):
        os.mkdir(animal_path)
    
    for name in dir[animal]:
        name_path = animal_path + "/" + name
        if not os.path.isdir(name_path):
            os.mkdir(name_path)
        print(name_path)

        options = webdriver.ChromeOptions()
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
        elem = driver.find_element(By.NAME,"q")
        elem.send_keys(name+" 얼굴")
        elem.send_keys(Keys.RETURN)
        
        
        SCROLL_PAUSE_TIME = 1
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    driver.find_element(By.CSS_SELECTOR,".mye4qd").click()
                except: break
            last_height = new_height
        

        images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")
        count = 1
        #반복문으로 이미지요소 배열들 돌며 작업 
        print("**********"+name_path+"*************")
        for image in images:
            try:
                image.click()
                time.sleep(1)
                driver.find_element(By.CSS_SELECTOR,".n3VNCb")
                imgUrl = driver.find_element(By.XPATH,"//*[@id='Sva75c']/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div/div[1]/div[2]/div[2]/div/a/img").get_attribute('src')
                urllib.request.urlretrieve(imgUrl, name_path + "/" + str(count)+".jpg")
                count = count + 1
                
            except:
                    print("g")
                    pass
        
        driver.close()