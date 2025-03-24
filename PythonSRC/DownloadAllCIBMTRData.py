import os
import csv
import re
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# =============================================================================
# Configuration
# =============================================================================
BASE_URL = "https://cibmtr.org"
DATASETS_PAGE = BASE_URL + "/CIBMTR/Resources/Publicly-Available-Datasets"
BASE_DOWNLOAD_DIR = "CIBMTR_Datasets"  # Base folder for downloads
METADATA_CSV_PATH = "cibmtr_datasets_metadata.csv"  # CSV to store metadata

# Create base download directory if not exists
os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)

# =============================================================================
# Selenium WebDriver Setup (Chrome in headless mode)
# =============================================================================
chrome_options = Options()
chrome_options.add_argument("--headless")
# If chromedriver is not in PATH, specify executable_path as below:
# driver = webdriver.Chrome(executable_path="/path/to/chromedriver", options=chrome_options)
driver = webdriver.Chrome(options=chrome_options)

print("Opening the datasets page...")
driver.get(DATASETS_PAGE)
# Wait to ensure the dynamic content loads; adjust the delay if needed
time.sleep(5)

# Retrieve the fully rendered HTML
html = driver.page_source
driver.quit()

# =============================================================================
# Parse the HTML with BeautifulSoup
# =============================================================================
soup = BeautifulSoup(html, "html.parser")

# -------------------------------------------------------------------
# IMPORTANT:
# The CIBMTR site uses <div class="item"> for each dataset card.
# The old script used <div class="dataset-item">, which does not exist.
# -------------------------------------------------------------------
dataset_items = soup.find_all("div", class_="item")
if not dataset_items:
    print("No dataset entries found using 'div.item'.")
    print("Please inspect the page structure again if this persists.")
    exit(1)

# Prepare list to store metadata
metadata_rows = []
csv_fields = ["Year", "Title", "Author", "Dataset_File", "DataDictionary_File"]

# =============================================================================
# Helper functions to extract desired text
# =============================================================================
def extract_year(item_soup):
    """
    Extracts the 'Start Year of Infusion: XXXX' from 
    <p class="Authors"><span class="label">Start Year of Infusion:</span>XXXX</p>.
    Returns a string or 'Unknown'.
    """
    year_para = item_soup.find(
        "p", class_="Authors", 
        string=re.compile("Start Year of Infusion:")
    )
    if year_para:
        # e.g. "Start Year of Infusion: 2003"
        text_full = year_para.get_text(strip=True)
        # Remove the label part
        return text_full.replace("Start Year of Infusion:", "").strip()
    return "Unknown"

def extract_author(item_soup):
    """
    Extracts the 'First Author: XXXX' from 
    <p class="Authors"><span class="label">First Author:</span>XXXX</p>.
    Returns a string or 'Unknown'.
    """
    author_para = item_soup.find(
        "p", class_="Authors", 
        string=re.compile("First Author:")
    )
    if author_para:
        text_full = author_para.get_text(strip=True)
        return text_full.replace("First Author:", "").strip()
    return "Unknown"

def extract_title(item_soup):
    """
    Extracts the dataset title from:
      <p class="teaser">
        <strong>Some Title</strong>
      </p>
    Returns a string or 'No Title'.
    """
    teaser = item_soup.find("p", class_="teaser")
    if teaser and teaser.find("strong"):
        return teaser.find("strong").get_text(strip=True)
    return "No Title"

def extract_links(item_soup):
    """
    Finds any <a> elements that contain 'Dataset' or 'Data Dictionary' in their text.
    Returns (dataset_url, data_dict_url).
    """
    dataset_url = ""
    data_dict_url = ""
    
    # Iterate over all <a> tags in this dataset "card"
    for link_tag in item_soup.find_all("a"):
        text = link_tag.get_text(strip=True)
        if "Dataset" in text:
            potential_url = link_tag.get("href", "")
            if potential_url.startswith("/"):
                potential_url = BASE_URL + potential_url
            dataset_url = potential_url
        elif "Data Dictionary" in text:
            potential_url = link_tag.get("href", "")
            if potential_url.startswith("/"):
                potential_url = BASE_URL + potential_url
            data_dict_url = potential_url
    
    return dataset_url, data_dict_url

# =============================================================================
# Process each dataset entry
# =============================================================================
for item in dataset_items:
    year = extract_year(item)
    title = extract_title(item)
    author = extract_author(item)
    
    dataset_url, data_dict_url = extract_links(item)

    # Create a folder for each year
    year_dir = os.path.join(BASE_DOWNLOAD_DIR, year)
    os.makedirs(year_dir, exist_ok=True)

    # Prepare local filenames
    dataset_filename = ""
    if dataset_url:
        dataset_filename = dataset_url.split("/")[-1].strip()
    data_dict_filename = ""
    if data_dict_url:
        data_dict_filename = data_dict_url.split("/")[-1].strip()
    
    dataset_filepath = os.path.join(year_dir, dataset_filename) if dataset_filename else ""
    data_dict_filepath = os.path.join(year_dir, data_dict_filename) if data_dict_filename else ""

    # ----------------------------
    # Download the dataset file
    # ----------------------------
    if dataset_url and dataset_filename:
        if not os.path.exists(dataset_filepath):
            print(f"Downloading dataset: {title} ({year}) -> {dataset_filename}")
            try:
                r = requests.get(dataset_url, timeout=30)
                r.raise_for_status()
                with open(dataset_filepath, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Failed to download dataset {dataset_filename}: {e}")
        else:
            print(f"Dataset file already exists: {dataset_filename}")
    else:
        print(f"No dataset link found for: '{title}'")

    # ----------------------------
    # Download the data dictionary
    # ----------------------------
    if data_dict_url and data_dict_filename:
        if not os.path.exists(data_dict_filepath):
            print(f"Downloading data dictionary: {title} ({year}) -> {data_dict_filename}")
            try:
                r = requests.get(data_dict_url, timeout=30)
                r.raise_for_status()
                with open(data_dict_filepath, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Failed to download data dictionary {data_dict_filename}: {e}")
        else:
            print(f"Data dictionary already exists: {data_dict_filename}")
    else:
        print(f"No data dictionary link found for: '{title}'")

    # ----------------------------
    # Save metadata for CSV
    # ----------------------------
    metadata_rows.append({
        "Year": year,
        "Title": title,
        "Author": author,
        "Dataset_File": os.path.join(year, dataset_filename) if dataset_filename else "",
        "DataDictionary_File": os.path.join(year, data_dict_filename) if data_dict_filename else "",
    })

# =============================================================================
# Save all metadata to a CSV file
# =============================================================================
with open(METADATA_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"\nDownloaded {len(metadata_rows)} dataset entries. Metadata saved to {METADATA_CSV_PATH}.")
