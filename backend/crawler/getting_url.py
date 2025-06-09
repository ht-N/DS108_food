from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

name = "Default"
profile_path = fr"C:/Users/hello/AppData/Local/Microsoft/Edge/User Data"

webdriver_path = fr"msedgedriver.exe"
service = Service(webdriver_path)
options = webdriver.EdgeOptions()
options.add_argument(f'--user-data-dir={profile_path}')
options.add_argument(f'--profile-directory={name}')
# options.add_argument("headless")

driver = webdriver.Edge(service=service, options=options)

# List of job categories
job_categories = [
    "tim-viec-lam-ke-toan-kiem-toan-thue-cr392",
    "tim-viec-lam-kinh-doanh-ban-hang-cr1",
    "tim-viec-lam-bat-dong-san-xay-dung-cr333",
    "tim-viec-lam-cong-nghe-thong-tin-cr257",
    "tim-viec-lam-tai-chinh-ngan-hang-bao-hiem-cr206",
    "tim-viec-lam-nhan-su-hanh-chinh-phap-che-cr177",
    "tim-viec-lam-cham-soc-khach-hang-customer-service-van-hanh-cr158",
    "tim-viec-lam-marketing-pr-quang-cao-cr92"
]

url_sp = []

# Loop through each job category
for category in job_categories:
    print(f"\nStarting category: {category}")
    
    # Loop through pages 1-2 for each category
    for i in range(1, 2):
        url = f"https://www.topcv.vn/{category}?type_keyword=0&page={i}"
        print(f"Try getting url for category: {category}, page: {i}")

        driver.get(url)

        # Check if captcha appears and wait for manual solving if needed
        try:
            # Wait up to 60 seconds for job listings to appear
            print("Waiting for page to load or captcha to be solved...")
            WebDriverWait(driver, 60).until(
                EC.presence_of_all_elements_located((By.XPATH, "//h3[@class='title ']//a"))
            )
            print("Page loaded successfully!")
        except:
            # If timeout occurs, ask for manual intervention
            print("Captcha might be present. Please solve it manually.")
            input("Press Enter after solving the captcha...")
            # Wait a bit more after manual intervention
            time.sleep(3)

        elements = driver.find_elements(By.XPATH, "//h3[@class='title ']//a")

        # If no elements found, we might have reached the end of available pages
        if not elements:
            print(f"No more jobs found for {category} at page {i}. Moving to next category.")
            break

        for element in elements:
            # Get href attribute value
            href_value = element.get_attribute("href")
            url_sp.append(href_value)

        print(f"Total URLs collected so far: {len(url_sp)}")
        
        # Optional: add a delay between page requests to avoid overwhelming the server
        time.sleep(2)

driver.quit()

df = pd.DataFrame(url_sp, columns=["URL"])

# Save DataFrame to CSV file
df.to_csv("urls_all.csv", index=False)