from selenium import webdriver
import time
from pathlib import Path
from typing import Optional

import common.mylogger


class WebContentFetcher:
    def __init__(self) -> None:
        self.logger = common.mylogger.setup_logger()

    def fetch_html(self, url: str) -> Optional[bytes]:
        """指定されたURLからデータを取得してbytes型で返す。データが取得できない場合はNoneを返す"""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")
            driver = webdriver.Chrome()
            # ページが完全に読み終わるまで待つ
            driver.set_page_load_timeout(300)

            # 最大5回繰り返す
            for i in range(5):
                try:
                    driver.get(url)
                    break
                except:
                    print(f"HTML取得に失敗（{i+1}回目）")
                    time.sleep(10)

            html = driver.page_source.encode("utf-8")

            driver.quit()
            return html
        except Exception as e:
            self.logger.error(f"url：{url}, error：{e}")
            return None
