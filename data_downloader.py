
import os
import sys
import requests
import zipfile
import io
import time
from bs4 import BeautifulSoup
import pandas as pd

# Import shared medical content filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.medical_content_filter import (
    is_medical_content,
    get_medical_keyword_count,
    get_medical_content_score,
    analyze_text_medical_content
)


DATA_DIR = 'general_medical_misinformation_data'
os.makedirs(DATA_DIR, exist_ok=True)
RAW_DATA_OUTPUT = 'data/processed/raw_downloaded_data.csv'
os.makedirs(os.path.dirname(RAW_DATA_OUTPUT), exist_ok=True)
DELAY_SECONDS = 3

def is_url_accessible(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=20)
        if response.status_code < 400:
            return True
        print(f"      {url} returned status {response.status_code}. Skipping.")
        return False
    except Exception as exc:
        print(f"      Could not reach {url}: {exc}")
        return False


def download_kaggle_datasets(slugs):
    kaggle_config_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_config_path):
        print("   Kaggle API not configured. Skipping Kaggle datasets.")
        print("   (This is OK - other data sources will be used)")
        return
    
    kaggle_slugs_list = [
        'sudalairajkumar/novel-corona-virus-2019-dataset',
        'imdevskp/corona-virus-report',
        'kimjihoo/coronavirusdataset',
        'tanmoyx/covid19-patient-precondition-dataset',
        
        'clmentbisaillon/fake-and-real-news-dataset',
        'jillanisofttech/fake-or-real-news',
        'gpreda/covid19-tweets',
        'sophia742/covid19-misinformation-tweets',
    ]
    
    downloaded_count = 0
    total_slugs = len(kaggle_slugs_list)
    for idx, slug in enumerate(kaggle_slugs_list, 1):
        try:
            print(f"   [{idx}/{total_slugs}] Downloading Kaggle dataset: {slug}")
            
            result = os.system(f'kaggle datasets download -d {slug} -p {DATA_DIR} --unzip > /dev/null 2>&1')
            if result == 0:
                downloaded_count += 1
                print(f"      Successfully downloaded {slug}")
            
                filter_kaggle_dataset(slug)
            else:
                print(f"      Failed to download {slug} (API error or dataset not accessible)")
            time.sleep(DELAY_SECONDS) 
        except Exception as e:
            print(f"      Error downloading {slug}: {e}")
            pass
    
    print(f"   Successfully downloaded {downloaded_count}/{len(kaggle_slugs_list)} Kaggle datasets")
    if downloaded_count == 0:
        print("   (No Kaggle datasets downloaded - will use other data sources)")

def filter_kaggle_dataset(slug):
    """Filter Kaggle dataset to keep only medical content"""
    try:
        # Find CSV files in the dataset directory
        dataset_name = slug.split('/')[-1]
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path, nrows=1000, on_bad_lines='skip')  

                        text_cols = []
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                text_cols.append(col)
                        
                        if text_cols:
                            # Check if dataset is medical
                            medical_rows = 0
                            total_rows = min(100, len(df))
                            
                            for _, row in df.head(total_rows).iterrows():
                                for col in text_cols:
                                    if pd.notna(row[col]) and is_medical_content(str(row[col])):
                                        medical_rows += 1
                                        break
                            
                            medical_percentage = (medical_rows / total_rows) * 100
                            
                            if medical_percentage < 25:  # Less than 25% medical content
                                print(f"      Removing non-medical dataset: {file} ({medical_percentage:.0f}% medical)")
                                os.remove(file_path)
                            else:
                                print(f"      Keeping medical dataset: {file} ({medical_percentage:.0f}% medical)")
                    
                    except Exception as e:
                        print(f"      Could not filter {file}: {e}")
    except Exception as e:
        print(f"   Warning: Could not filter dataset {slug}: {e}")

def download_url_file(url, output_file_name):
    """Downloads a single file (CSV/JSON/ZIP) from a direct URL with progress."""
    output_path = os.path.join(DATA_DIR, output_file_name)
    try:
        print(f"      Starting download: {output_file_name}...")
        # Increased timeout
        with requests.get(url, stream=True, timeout=(10, 180)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            if url.endswith('.zip') or r.headers.get('Content-Type') == 'application/zip':
                print(f"      Downloading ZIP ({total_size / (1024*1024):.1f} MB)...")
                content = b''
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024*1024) == 0: 
                                print(f"      Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)", end='\r')
                
                print(f"      Extracting {output_file_name}...")
                z = zipfile.ZipFile(io.BytesIO(content))
                z.extractall(DATA_DIR)
                print(f"      Downloaded and extracted: {output_file_name}")
            else:
                print(f"      Downloading file ({total_size / (1024*1024):.1f} MB)...")
                downloaded = 0
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                if downloaded % (1024*1024) == 0:  # Print every MB
                                    print(f"      Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)", end='\r')
                print(f"      Downloaded: {output_file_name}")
    except requests.exceptions.RequestException as e:
        print(f"      Failed to download {output_file_name}: {e}")
    except Exception as e:
        print(f"      Error downloading {output_file_name}: {e}")
    time.sleep(DELAY_SECONDS)

def scrape_fact_check_page(url, output_file_name, source_name=None, expected_label='credible'):
    output_path = os.path.join(DATA_DIR, output_file_name)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        
        main_content = (soup.find('main') or 
                       soup.find('article') or 
                       soup.find('div', id='content') or
                       soup.find('div', class_='content') or
                       soup.find('div', class_='main-content') or
                       soup.find('body'))
        
        extracted_texts = []
        medical_filtered = 0
        non_medical_filtered = 0
        
        if main_content:
            # Extract headings and paragraphs
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'div']): 
                text = element.get_text(strip=True)
                
                # Filter out navigation, copyright, and very short text
                if (len(text) < 40 or 
                    '©' in text or 
                    'cookie' in text.lower() or
                    'privacy policy' in text.lower() or
                    'terms of service' in text.lower() or
                    'sign up' in text.lower() or
                    'subscribe' in text.lower() or
                    'follow us' in text.lower()):
                    continue
                
              
                if is_medical_content(text, min_medical_keywords=2):
                 
                    extracted_texts.append(f"\n--- CLAIM/FACT ---\nLABEL: {expected_label}\n{text}")
                    medical_filtered += 1
                else:
                    non_medical_filtered += 1
        
        if extracted_texts:
            metadata = f"SOURCE: {source_name or 'unknown'}\nURL: {url}\nLABEL: {expected_label}\n"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(metadata)
                f.write('\n'.join(extracted_texts))
            print(f"      Scraped {medical_filtered} medical sections from {source_name or url}")
            if non_medical_filtered > 0:
                print(f"        (Filtered out {non_medical_filtered} non-medical sections)")
        else:
            print(f"      No medical content extracted from {url}")
    except requests.exceptions.RequestException as e:
        print(f"      Error accessing {url}: {e}")
    except Exception as e:
        print(f"      Error scraping {url}: {e}")


def save_raw_data_to_csv():
    print("\n5. Saving raw downloaded data to CSV...")
    
    raw_records = []
    
    try:
        # Process CSV files
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
                        
                        # Find text columns
                        text_cols = []
                        for col in df.columns:
                            col_lower = col.lower()
                            if col_lower in ['text', 'statement', 'claim', 'content', 'message', 'article', 'title']:
                                text_cols.append(col)
                        
                        # If no text columns found, try object type columns
                        if not text_cols:
                            for col in df.columns:
                                if df[col].dtype == 'object':
                                    text_cols.append(col)
                        
                        # Extract text data
                        for _, row in df.iterrows():
                            for col in text_cols:
                                if pd.notna(row[col]):
                                    text = str(row[col]).strip()
                                    if len(text) >= 10 and is_medical_content(text):
                                        raw_records.append({
                                            'text': text[:10000],
                                            'source_file': os.path.basename(file),
                                            'source_type': 'csv',
                                            'label': None  
                                        })
                    except Exception as e:
                        print(f"      Error reading CSV {file}: {e}")
        
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Split content into sections
                        sections = content.split('--- CLAIM/FACT ---')
                        
                        for section in sections:
                            # Remove metadata lines
                            lines = [line for line in section.split('\n') 
                                    if not line.startswith('SOURCE:') 
                                    and not line.startswith('URL:')
                                    and not line.startswith('LABEL:')
                                    and line.strip()]
                            
                            text = '\n'.join(lines).strip()
                            
                            if len(text) >= 50 and is_medical_content(text, min_medical_keywords=2):
                                raw_records.append({
                                    'text': text[:10000],
                                    'source_file': os.path.basename(file),
                                    'source_type': 'scraped',
                                    'label': None 
                                })
                    except Exception as e:
                        print(f"      Error reading TXT {file}: {e}")
        
        # Remove duplicates
        if raw_records:
            raw_df = pd.DataFrame(raw_records)
            raw_df = raw_df.drop_duplicates(subset=['text'], keep='first')
            

            raw_df.to_csv(RAW_DATA_OUTPUT, index=False, quoting=1, escapechar='\\')
            print(f"      Saved {len(raw_df)} raw records to {RAW_DATA_OUTPUT}")
            print(f"      These records will be labeled and saved to medical_dataset.csv when you run process_and_label_data.py")
        else:
            print(f"      No raw data found to save")
    
    except Exception as e:
        print(f"      Error saving raw data: {e}")

def analyze_downloaded_data():
    try:
        total_files = 0
        total_csv = 0
        total_txt = 0
        total_size = 0
        
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                total_files += 1
                total_size += os.path.getsize(file_path)
                
                if file.endswith('.csv'):
                    total_csv += 1
                elif file.endswith('.txt'):
                    total_txt += 1
        
        print(f"\n   Download Statistics:")
        print(f"      Total files: {total_files}")
        print(f"      CSV files: {total_csv}")
        print(f"      Text files: {total_txt}")
        print(f"      Total size: {total_size / (1024*1024):.2f} MB")
        
        # Check medical content percentage in text files
        medical_files = 0
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if is_medical_content(content, min_medical_keywords=5):
                                medical_files += 1
                    except:
                        pass
        
        if total_txt > 0:
            medical_percentage = (medical_files / total_txt) * 100
            print(f"      Medical text files: {medical_files}/{total_txt} ({medical_percentage:.0f}%)")
        
    except Exception as e:
        print(f"   Could not analyze data: {e}")


if __name__ == '__main__':
    print("="*60)
    print("MEDICAL DATA DOWNLOADER")
    print("="*60)
    print("\nDownloading datasets from various sources...")
    print("After download, run 'process_and_label_data.py' to process and label the data.\n")
    
    print("1. Downloading Kaggle datasets...")
    download_kaggle_datasets([]) 


    print("\n2. Downloading medical datasets from GitHub/URLs...")
    direct_download_targets = [
    
        {'url': 'https://github.com/kinit-sk/medical-misinformation-dataset/archive/refs/heads/main.zip', 
         'name': 'monant_misinfo.zip',
         'description': 'Monant Medical Misinformation Dataset (317k articles)'},
        
        {'url': 'https://github.com/styxsys0927/Med-MMHL/archive/refs/heads/main.zip',
         'name': 'med_mmhl_multi_disease.zip',
         'description': 'Med-MMHL Dataset (Multi-Modal, Multi-Disease)'},
        
        {'url': 'https://github.com/soroushjavdan/Medical-News-Dataset/archive/refs/heads/main.zip',
         'name': 'medical_news.zip',
         'description': 'Medical News Dataset (credible statements)'},
        
        {'url': 'https://github.com/KaiDMML/FakeNewsNet/archive/refs/heads/master.zip',
         'name': 'fakenewsnet.zip',
         'description': 'FakeNewsNet (includes health misinformation)'},
        
        {'url': 'https://github.com/GateNLP/medical-fact-checking/archive/refs/heads/master.zip',
         'name': 'medical_fact_checking.zip',
         'description': 'Medical Fact Checking corpora'},
        
        {'url': 'https://github.com/COVID-science/misinfo-claims/archive/refs/heads/main.zip',
         'name': 'covid_misinfo_claims.zip',
         'description': 'COVID-19 misinformation claims'},
     
        {'url': 'https://github.com/ncbi-nlp/BioSentVec/archive/refs/heads/master.zip',
         'name': 'biosentvec.zip',
         'description': 'BioSentVec: Biomedical sentence embeddings'},
        
        {'url': 'https://github.com/ncbi-nlp/BioWordVec/archive/refs/heads/master.zip',
         'name': 'biowordvec.zip',
         'description': 'BioWordVec: Biomedical word embeddings'},
    ]

    total_targets = len(direct_download_targets)
    for idx, target in enumerate(direct_download_targets, 1):
        print(f"   [{idx}/{total_targets}] Downloading: {target.get('description', target['name'])}")
        if is_url_accessible(target['url']):
            download_url_file(target['url'], target['name'])
        else:
            print(f"      Skipping {target['name']} (URL not reachable)")

 
    print("\n3. Hugging Face dataset downloads skipped (require authentication).")
    print("   Use locally bundled samples or update URLs if you have access.")
    

    print("\n4. Web scraping skipped (all previous endpoints returned 404/405).")
    

    print("\n4. Analyzing downloaded data...")
    analyze_downloaded_data()
    

    save_raw_data_to_csv()
    
    print("\n" + "="*60)
    print("DOWNLOAD PROCESS COMPLETED")
    print("="*60)
    print(f"\nData saved to: {DATA_DIR}/")
    print(f"Raw data saved to: {RAW_DATA_OUTPUT}")
    print("\nAll downloaded data has been filtered for medical content")
    print("Raw data saved to CSV (unlabeled)")
    print("\nNext step: Run 'python process_and_label_data.py' to label the data and save to medical_dataset.csv")
    print("="*60)