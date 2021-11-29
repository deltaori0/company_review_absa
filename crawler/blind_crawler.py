from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import openpyxl

company = 'SK%ED%85%94%EB%A0%88%EC%BD%A4'
# 삼성전자 : '%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90'
# 네이버 : 'NAVER'
# 카카오 : '%EC%B9%B4%EC%B9%B4%EC%98%A4'
# 라인플러스 : '%EB%9D%BC%EC%9D%B8%ED%94%8C%EB%9F%AC%EC%8A%A4'
# 쿠팡 : 'COUPANG'
# 배민 (우아한형제들) : '%EC%9A%B0%EC%95%84%ED%95%9C%ED%98%95%EC%A0%9C%EB%93%A4'
# SKT : 'SK%ED%85%94%EB%A0%88%EC%BD%A4'
start_page = 1
end_page = 30

# 0. 엑셀 파일 준비
wb = openpyxl.Workbook()
sheet = wb.active
sheet.append(['sentence', 'polarity'])

# 1. 웹 드라이버 켜기
driver = webdriver.Chrome('./chromedriver')

# 2. 블라인드 접속하기
driver.get(f'https://www.teamblind.com/kr/company/{company}/reviews')

# 3. 로그인
driver.find_element(By.CLASS_NAME, 'btn_signin').click()
time.sleep(20)

# 4. 리뷰 수집

for page in range(start_page, end_page+1):
    driver.get(f'https://www.teamblind.com/kr/company/{company}/reviews?page={page}')

    reviews = driver.find_elements(By.CLASS_NAME, 'review_item_inr')
    print('number of reviews', len(reviews))    # 한 페이지에 30개 존재

    for i in range(1, len(reviews)+1):
        pos = driver.find_element(By.CSS_SELECTOR, f'div.review_all > div:nth-of-type({i})  p:nth-of-type(1) > span').text
        print('pos', pos)
        neg = driver.find_element(By.CSS_SELECTOR, f'div.review_all > div:nth-of-type({i})  p:nth-of-type(2) > span').text
        print('neg', neg)

        sheet.append([pos, 'positive'])
        sheet.append([neg, 'negative'])


# 5. 엑셀 저장
wb.save('skt_review.xlsx')