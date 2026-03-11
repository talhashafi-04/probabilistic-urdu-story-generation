import cloudscraper
import time
import os

def download_sample_htmls():
    # The milestone IDs you want to check (skipping 2000 since we know it)
    sample_ids = [10, 50, 100, 200, 500, 800, 1200, 1500, 2500, 2800]
    
    # Make a clean folder for them
    save_folder = "sample_htmls"
    os.makedirs(save_folder, exist_ok=True)
    
    # Initialize Cloudscraper
    scraper = cloudscraper.create_scraper(browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    })
    
    print(f"Downloading {len(sample_ids)} sample HTML files...")
    
    for story_id in sample_ids:
        url = f"https://www.urdupoint.com/kids/detail/moral-stories/anything-{story_id}.html"
        print(f"Fetching ID {story_id}...")
        
        try:
            response = scraper.get(url, timeout=15)
            
            if response.status_code == 200:
                # Normalize name to just the integer
                filename = os.path.join(save_folder, f"{story_id}.html")
                
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"  -> Saved as {filename}")
                
            elif response.status_code == 404:
                print(f"  -> 404 Not Found (Story might be deleted)")
            else:
                print(f"  -> Failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"  -> Error: {e}")
            
        # Be polite to Cloudflare
        time.sleep(2) 

    print("\nAll samples downloaded successfully!")

if __name__ == "__main__":
    download_sample_htmls()
