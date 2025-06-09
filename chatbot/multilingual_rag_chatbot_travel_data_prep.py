# Jade Oakes
# April 28, 2025
# Scrapes Wikivoyage articles in English and Spanish for a list of cities,
# and saves structured JSONL files for each city and language.
# Also combines all the scraped files into a single JSONL file for downstream use.

import requests
import os
import json
import time
from tqdm import tqdm

# User agent to avoid request blocks by Wikivoyage
HEADERS = {
    "User-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


def get_all_city_articles_api(lang_code="en", category="City articles"):
    """Fetches all city article titles from a Wikivoyage category using API"""

    all_cities = []
    session = requests.Session()

    base_url = f"https://{lang_code}.wikivoyage.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 500,
        "format": "json"
    }

    while True:
        print(f"Fetching batch... (Total cities so far: {len(all_cities)})")
        try:
            response = session.get(base_url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Failed to fetch category members: {e}")
            break

        batch = data.get('query', {}).get('categorymembers', [])
        all_cities.extend([item['title'] for item in batch if item['ns'] == 0])  # ns=0 ensures it's an article

        if 'continue' in data:
            params['cmcontinue'] = data['continue']['cmcontinue']
        else:
            break

        time.sleep(1)  # polite delay

    if not all_cities:
        print("No cities found.")
    return all_cities


def fetch_wikivoyage_sections_api(city, lang_code="en"):
    """Fetches the text of a Wikivoyage article using API"""

    base_url = f"https://{lang_code}.wikivoyage.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": city,
        "explaintext": True,  # plain text instead of HTML
        "format": "json"
    }

    try:
        response = requests.get(base_url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Failed to fetch article for {city} ({lang_code}): {e}")
        return None

    pages = data.get('query', {}).get('pages', {})
    if not pages:
        print(f"No pages found for {city} ({lang_code})")
        return None

    page = next(iter(pages.values()))
    extract = page.get('extract')
    if not extract:
        print(f"No extract found for {city} ({lang_code})")
        return None

    return {"Introduction": extract}  # Wrap in dict to match your old format


def save_city_to_jsonl(city, lang_code="en", save_path="data/wikivoyage/scraped_cities_data"):
    """Scrapes and saves a single city's content to a JSONL file"""

    # Create directory to save the JSONL files
    os.makedirs(save_path, exist_ok=True)

    # Generate a filename to be formatted cityname_lang.jsonl
    filename = f"{city.replace(' ', '_').replace('/', '_').lower()}_{lang_code}.jsonl"
    jsonl_path = os.path.join(save_path, filename)

    if os.path.exists(jsonl_path):
        print(f"Already exists: {filename} (skipping)")
        return

    # Fetch content sections for the city in the specified language
    data = fetch_wikivoyage_sections_api(city, lang_code=lang_code)

    if not data:
        return

    # Write each section of the article as a separate line in the JSONL file
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for section, content in data.items():
            record = {
                lang_code: f"{section}: {content}",
                "lang": lang_code,
                "source": "wikivoyage",
                "city": city
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {city} ({lang_code}) to {jsonl_path}")


def batch_scrape(cities, lang_code="en", delay=2):
    """Scrape each city and save to a JSONL file for the specified language"""
    for city in tqdm(cities, desc=f"Scraping ({lang_code})", unit="city"):
        save_city_to_jsonl(city, lang_code=lang_code)
        time.sleep(delay)


def combine_jsonl_files(input_dir="data/wikivoyage/scraped_cities_data", output_file="data/combined_travel_data.jsonl"):  # add _version# if needed
    """Combines all city JSONL files into a single JSONL file"""

    # Create directory / ensure it exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # List all JSONL files in the directory
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    # Open output file to write content
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Iterate over each JSONL file
        for file in tqdm(jsonl_files, desc="Combining JSONL files", unit="file"):
            filepath = os.path.join(input_dir, file)
            # Open each individual JSONL file and write its contents to the output file
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

    print(f"\nCombined {len(jsonl_files)} JSONL files into {output_file}")


if __name__ == "__main__":
    start_time = time.time()

    # English Wikivoyage scraping
    category_url_en = "https://en.wikivoyage.org/wiki/Category:City_articles"
    print("\nScraping English city list...")
    cities_en = get_all_city_articles_api(lang_code="en", category="City articles")
    print(f"Found {len(cities_en)} English cities")

    print("\nScaping English Wikivoyage pages...")
    en_start = time.time()
    batch_scrape(cities_en, lang_code="en")
    print(f"Finished English scraping in {time.time() - en_start:.2f} seconds.\n")

    # Spanish Wikivoyage scraping
    category_url_es = "https://es.wikivoyage.org/wiki/Categoría:Wikiviajes:Artículos_ciudad"
    print("\nScraping Spanish city list...")
    cities_es = get_all_city_articles_api(lang_code="es", category="Wikiviajes:Artículos ciudad")
    print(f"Found {len(cities_es)} Spanish cities")

    print("Starting Spanish Wikivoyage scraping...")
    es_start = time.time()
    batch_scrape(cities_es, lang_code="es")
    print(f"Finished Spanish scraping in {time.time() - es_start:.2f} seconds.\n")

    # Combine all JSONL files
    print("\nCombining JSONL files...")
    combine_start = time.time()
    combine_jsonl_files()
    print(f"Finished combining in {time.time() - combine_start:.2f} seconds.\n")

    print(f"Total runtime: {time.time() - start_time:.2f} seconds.")