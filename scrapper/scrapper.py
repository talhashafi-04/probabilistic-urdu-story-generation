import cloudscraper
from bs4 import BeautifulSoup
import re
import time

def clean_and_tag_urdu_text(text):
    """Cleans the text to only keep Urdu characters and injects NLP tags."""
    text = re.sub(r'[^\u0600-\u06FF\s۔،؟]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return ""
        
    text = text.replace('۔', ' <EOS> ')
    text = text.replace('؟', ' <EOS> ')
    
    return text + " <EOP>\n"

def scrape_urdu_corpus():
    output_file = "full_urdu_corpus.txt"
    success_count = 0 
    
    # Initialize the Cloudflare-bypassing scraper
    scraper = cloudscraper.create_scraper(browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    })
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")

    print("Starting extraction with Cloudflare bypass...")

    # Let's test the small batch again
    for story_id in range(1201, 2953): 
        url = f"https://www.urdupoint.com/kids/detail/moral-stories/anything-{story_id}.html"
        
        try:
            # We use scraper.get() instead of requests.get()
            response = scraper.get(url, timeout=15)
            
            if response.status_code == 404:
                print(f"[{story_id}] 404 Not Found")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            story_container = soup.find('div', class_='txt_detail') 
            
            if not story_container:
                page_title = soup.title.string if soup.title else 'No Title'
                print(f"[{story_id}] Skipped. Loaded page title: {page_title}")
                continue
            
            raw_text = story_container.get_text(separator=' ')
            processed_story = clean_and_tag_urdu_text(raw_text)
            
            if processed_story:
                processed_story += "<EOT>\n\n"
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(processed_story)
                
                success_count += 1
                print(f"[{story_id}] Success! Total saved: {success_count}")
                
        except Exception as e:
            print(f"[{story_id}] Error: {e}")
            
        # Give it a slightly longer sleep to avoid triggering aggressive rate limits

    print(f"\nScraping complete! Successfully built corpus with {success_count} stories.")

if __name__ == "__main__":
    scrape_urdu_corpus()