import os
import time
import base64
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import urllib.parse

def wait_for_logbb_result_src(driver, timeout=60):
    def src_has_result(driver):
        img = driver.find_element(By.ID, "prop_LogBB")
        src = img.get_attribute("src")
        return "result=" in src

    WebDriverWait(driver, timeout).until(src_has_result)


def input_smiles_via_context_menu(driver, smiles):
    # Wait for the editor container div
    editor_div = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[jsname='jsa-resetDiv'], div[tabindex='0']"))
    )

    # Right-click on the editor div
    action = ActionChains(driver)
    action.context_click(editor_div).perform()

    # Wait for the context menu and click “Paste as SMILES” option
    paste_smiles_option = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='gwt-uid-6']"))
    )
    paste_smiles_option.click()

    # Wait for the popup input box to appear (assumed it's a text input inside a dialog)
    smiles_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div/div/table/tbody/tr[2]/td[2]/div/div/div/div[2]/table/tbody/tr[2]/td/textarea")) # /html/body/div/div/table/tbody/tr[2]/td[2]/div/div/div/div[2]/table/tbody/tr[2]/td/textarea
    )

    # Clear any existing text and enter your SMILES
    smiles_input.clear()
    smiles_input.send_keys(smiles)

    # Find and click the OK/Accept button in the dialog
    ok_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/div/div/table/tbody/tr[2]/td[2]/div/div/div/div[2]/table/tbody/tr[3]/td/table/tbody/tr/td[1]/button"))
    ) # 
    ok_button.click()

def extract_logbb_value(driver, smiles):
    url = "http://qsar.chem.msu.ru/admet/"
    driver.get(url)

    # Input the SMILES via context menu method
    input_smiles_via_context_menu(driver, smiles)

    time.sleep(1)  # wait for editor update

    # Click Calculate
    calculate_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/table[1]/tbody/tr[2]/td[1]/button"))
    )
    calculate_button.click()

    wait_for_logbb_result_src(driver, timeout=20)

    logbb_img = driver.find_element(By.ID, "prop_LogBB")
    src = logbb_img.get_attribute("src")

    parsed_url = urllib.parse.urlparse(src)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    result_param = query_params.get('result', [])

    if result_param:
        parts = result_param[0].split('|')
        if len(parts) >= 2:
            try:
                return float(parts[1])  # The logBB value
            except ValueError:
                return None
    return None


def run_batch(smiles_list):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    
    results = {}
    for smi in smiles_list:
        try:
            val = extract_logbb_value(driver, smi)
            results[smi] = val
            print(f"{smi} → logBB = {val}")
        except Exception as e:
            print(f"Error with {smi}: {e}")
            results[smi] = None
    driver.quit()
    return results

if __name__ == "__main__":
    drug_like_smiles = list(pd.read_csv('../datasets/b3db_drug_like.csv')['SMILES'])
    result = run_batch(drug_like_smiles)

    df = pd.DataFrame(list(result.items()), columns=["SMILES", "qsar_pred_logBB"])
    df.to_csv('./datasets/qsar_predictions.csv', index=False)