#!/usr/bin/env python3
"""
Aimé Leon Dore Product Scraper
Scrapes all products from aimeleondore.com and stores them in Supabase with image embeddings.
"""

import asyncio
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import cloudscraper
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from supabase import create_client
from tqdm.asyncio import tqdm
from webdriver_manager.chrome import ChromeDriverManager

# Import for embeddings
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import io

# Constants
BASE_URL = "https://www.aimeleondore.com"
SHOP_ALL_URL = "https://www.aimeleondore.com/collections/shop-all"
SUPABASE_URL = "https://yqawmzggcgpeyaaynrjk.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"

# Embedding model
EMBEDDING_MODEL_NAME = "google/siglip-base-patch16-384"

class AimeLeonDoreScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.scraper = cloudscraper.create_scraper()
        self.supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def get_random_headers(self) -> Dict[str, str]:
        """Generate random headers to avoid detection."""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def setup_selenium_driver(self, headless: bool = True) -> webdriver.Chrome:
        """Set up Selenium WebDriver with Chrome."""
        chrome_options = Options()

        # Essential options for CI/headless environments
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--window-size=1920,1080")

        if headless:
            chrome_options.add_argument("--headless")

        # Anti-detection options
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Add user agent
        chrome_options.add_argument(f"--user-agent={self.ua.random}")

        # Additional options for stability
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")  # Speed up loading

        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Execute script to remove webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        return driver

    async def scrape_product_urls(self) -> List[str]:
        """Scrape all product URLs from the shop-all collection page."""
        # Check if running in CI environment
        is_ci = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'

        if is_ci:
            logger.info("Running in CI environment, using requests-based scraping...")
            return await self._scrape_product_urls_requests()
        else:
            logger.info("Running locally, using Selenium for full scraping...")
            return await self._scrape_product_urls_selenium()

    async def _scrape_product_urls_selenium(self) -> List[str]:
        """Scrape product URLs using Selenium (for local development)."""
        logger.info("Starting to scrape product URLs with Selenium...")
        product_urls = set()

        driver = self.setup_selenium_driver()

        try:
            driver.get(SHOP_ALL_URL)
            logger.info(f"Loaded page: {SHOP_ALL_URL}")

            # Wait for initial products to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-product-url]"))
            )

            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_count = 0
            max_scrolls = 50  # Safety limit

            while scroll_count < max_scrolls:
                # Scroll to bottom
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait for new content to load
                time.sleep(3)

                # Check if page height changed
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    # Try one more time in case loading was slow
                    time.sleep(2)
                    final_height = driver.execute_script("return document.body.scrollHeight")
                    if final_height == new_height:
                        logger.info("Reached end of page")
                        break

                last_height = new_height
                scroll_count += 1
                logger.info(f"Scroll {scroll_count}, height: {new_height}")

                # Extract product URLs from current page
                soup = BeautifulSoup(driver.page_source, 'lxml')
                product_links = soup.find_all('a', href=re.compile(r'/products/'))

                for link in product_links:
                    href = link.get('href')
                    if href and '/products/' in href:
                        full_url = urljoin(BASE_URL, href)
                        product_urls.add(full_url)

                logger.info(f"Found {len(product_urls)} unique product URLs so far")

            logger.info(f"Total unique product URLs found: {len(product_urls)}")

        except Exception as e:
            logger.error(f"Error scraping product URLs with Selenium: {e}")
            # Fallback to requests method
            logger.info("Falling back to requests-based scraping...")
            return await self._scrape_product_urls_requests()
        finally:
            driver.quit()

        return list(product_urls)

    async def _scrape_product_urls_requests(self) -> List[str]:
        """Scrape product URLs using requests (for CI environments)."""
        logger.info("Starting to scrape product URLs with requests...")
        product_urls = []

        try:
            headers = self.get_random_headers()
            response = requests.get(SHOP_ALL_URL, headers=headers, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml')
                product_links = soup.find_all('a', href=re.compile(r'/products/'))

                for link in product_links:
                    href = link.get('href')
                    if href and '/products/' in href:
                        full_url = urljoin(BASE_URL, href)
                        product_urls.append(full_url)

                # Remove duplicates
                product_urls = list(set(product_urls))
                logger.info(f"Found {len(product_urls)} unique product URLs")
            else:
                logger.error(f"Failed to fetch page: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error scraping product URLs with requests: {e}")

        return product_urls


    async def download_image(self, session: aiohttp.ClientSession, image_url: str) -> Optional[bytes]:
        """Download image from URL."""
        try:
            async with session.get(image_url, timeout=30) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.warning(f"Failed to download image {image_url}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {e}")
            return None

    def generate_embedding(self, image_bytes: bytes) -> Optional[List[float]]:
        """Generate 768-dimensional embedding from image bytes using SigLIP."""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Process image with dummy text (required for SigLIP)
            inputs = self.processor(
                images=image,
                text=[""],  # Empty text input
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # For SigLIP, the image embeddings are in outputs.image_embeds
                if hasattr(outputs, 'image_embeds'):
                    embedding = outputs.image_embeds
                elif hasattr(outputs, 'pooler_output'):
                    embedding = outputs.pooler_output
                else:
                    # Fallback to last hidden state mean
                    embedding = outputs.last_hidden_state.mean(dim=1)

                embedding = embedding.cpu().numpy().flatten()

            # Ensure it's 768 dimensions (SigLIP base should output 768)
            if len(embedding) != 768:
                logger.warning(f"Embedding dimension is {len(embedding)}, expected 768")
                return None

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    async def scrape_product_details(self, session: aiohttp.ClientSession, product_url: str) -> Optional[Dict]:
        """Scrape detailed product information from individual product page."""
        try:
            headers = self.get_random_headers()
            async with session.get(product_url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {product_url}: HTTP {response.status}")
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')

                # Extract product information
                product_data = {}

                # Title
                title_elem = soup.find('h1', {'class': re.compile(r'product.*title', re.I)}) or \
                           soup.find('h1') or \
                           soup.find('meta', {'property': 'og:title'})
                product_data['title'] = title_elem.get('content') if title_elem and title_elem.name == 'meta' else \
                                      title_elem.get_text(strip=True) if title_elem else None

                if not product_data['title']:
                    # Try alternative selectors
                    title_elem = soup.find('title')
                    if title_elem:
                        product_data['title'] = title_elem.get_text(strip=True).replace(' | Aimé Leon Dore', '')

                # Price
                price_elem = soup.find('span', {'class': re.compile(r'price', re.I)}) or \
                           soup.find('meta', {'property': 'product:price:amount'})
                if price_elem:
                    if price_elem.name == 'meta':
                        price_text = price_elem.get('content', '')
                    else:
                        price_text = price_elem.get_text(strip=True)

                    # Extract numeric price
                    price_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', price_text.replace(',', ''))
                    if price_match:
                        product_data['price'] = float(price_match.group(1))
                    else:
                        product_data['price'] = None
                else:
                    product_data['price'] = None

                # Image URL
                image_elem = soup.find('meta', {'property': 'og:image'}) or \
                           soup.find('img', {'class': re.compile(r'product.*image', re.I)})
                if image_elem:
                    if image_elem.name == 'meta':
                        product_data['image_url'] = image_elem.get('content')
                    else:
                        img_src = image_elem.get('src') or image_elem.get('data-src')
                        if img_src:
                            product_data['image_url'] = urljoin(BASE_URL, img_src) if img_src.startswith('/') else img_src

                # Description
                desc_elem = soup.find('meta', {'name': 'description'}) or \
                          soup.find('div', {'class': re.compile(r'product.*description|description', re.I)})
                product_data['description'] = desc_elem.get('content') if desc_elem and desc_elem.name == 'meta' else \
                                            desc_elem.get_text(strip=True) if desc_elem else None

                # Category (try to extract from breadcrumbs or URL)
                category_elem = soup.find('nav', {'class': re.compile(r'breadcrumb', re.I)})
                if category_elem:
                    categories = [a.get_text(strip=True) for a in category_elem.find_all('a')]
                    product_data['category'] = ' > '.join(categories[1:]) if len(categories) > 1 else None
                else:
                    # Extract from URL
                    url_parts = urlparse(product_url).path.split('/')
                    if len(url_parts) >= 3:
                        product_data['category'] = url_parts[2].replace('-', ' ').title()

                # Size information (if available)
                size_elem = soup.find('select', {'name': re.compile(r'size', re.I)}) or \
                          soup.find('div', {'class': re.compile(r'size.*option', re.I)})
                if size_elem:
                    if size_elem.name == 'select':
                        sizes = [opt.get_text(strip=True) for opt in size_elem.find_all('option') if opt.get('value')]
                        product_data['size'] = ', '.join(sizes)
                    else:
                        product_data['size'] = size_elem.get_text(strip=True)

                # Additional metadata
                metadata = {
                    'url': product_url,
                    'scraped_at': time.time()
                }

                # Try to get color variants
                color_elems = soup.find_all('input', {'name': 'Color'})
                if color_elems:
                    colors = [elem.get('value') for elem in color_elems if elem.get('value')]
                    if colors:
                        metadata['colors'] = colors

                # Check if sold out - very conservative approach
                # Default to AVAILABLE, only mark as sold out if we find explicit evidence
                is_sold_out = False
                
                # Look for buttons with exact "Sold Out" or "Out of Stock" text
                all_buttons = soup.find_all('button')
                for button in all_buttons:
                    button_text = button.get_text(strip=True).lower()
                    # Only match exact phrases to avoid false positives
                    if button_text in ['sold out', 'out of stock', 'unavailable']:
                        is_sold_out = True
                        break
                
                # If no explicit sold-out button found, check for disabled add-to-cart with sold-out text
                if not is_sold_out:
                    # Find add-to-cart buttons
                    cart_buttons = soup.find_all('button', string=re.compile(r'add|cart|buy', re.I))
                    for button in cart_buttons:
                        is_disabled = button.get('disabled') is not None
                        button_text = button.get_text(strip=True).lower()
                        # Only mark as sold out if button is disabled AND explicitly says sold out
                        if is_disabled and ('sold out' in button_text or 'out of stock' in button_text):
                            is_sold_out = True
                            break
                
                # Default to available - only mark sold out if we found clear evidence
                metadata['availability'] = 'sold_out' if is_sold_out else 'available'
                product_data['is_sold_out'] = is_sold_out

                product_data['metadata'] = json.dumps(metadata)

                # Generate ID from URL
                product_id = urlparse(product_url).path.split('/')[-1]
                if not product_id:
                    product_id = str(hash(product_url)) % 1000000
                product_data['id'] = f"aime-{product_id}"
                product_data['url'] = product_url  # Store URL directly for easier access

                return product_data

        except Exception as e:
            logger.error(f"Error scraping product {product_url}: {e}")
            return None

    async def process_products(self, product_urls: List[str], batch_size: int = 10) -> int:
        """Process all products: scrape details, generate embeddings, and store in database.
        
        Returns:
            int: Number of products successfully processed and stored
        """
        logger.info(f"Processing {len(product_urls)} products...")
        products_stored = 0

        async with aiohttp.ClientSession() as session:
            # Process in batches to avoid overwhelming the server
            for i in range(0, len(product_urls), batch_size):
                batch_urls = product_urls[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(product_urls) + batch_size - 1)//batch_size}")

                tasks = []
                for url in batch_urls:
                    tasks.append(self.scrape_product_details(session, url))

                # Scrape product details
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions, None results, and sold-out products
                valid_products = []
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Exception in batch: {result}")
                    elif result is not None:
                        # Skip sold-out products
                        if result.get('is_sold_out', False):
                            logger.info(f"Skipping sold-out product: {result.get('title', 'Unknown')}")
                            continue
                        valid_products.append(result)

                logger.info(f"Valid products in batch: {len(valid_products)}")

                # Process embeddings and store
                for product in tqdm(valid_products, desc="Processing products"):
                    try:
                        # Download image and generate embedding
                        if product.get('image_url'):
                            image_bytes = await self.download_image(session, product['image_url'])
                            if image_bytes:
                                embedding = self.generate_embedding(image_bytes)
                                if embedding:
                                    product['embedding'] = embedding
                                else:
                                    logger.warning(f"Failed to generate embedding for {product['id']}")
                                    continue
                            else:
                                logger.warning(f"Failed to download image for {product['id']}")
                                continue
                        else:
                            logger.warning(f"No image URL for {product['id']}")
                            continue

                        # Extract product URL - prefer direct field, fallback to metadata
                        product_url = product.get('url')
                        if not product_url:
                            try:
                                if product.get('metadata'):
                                    metadata_dict = json.loads(product['metadata'])
                                    product_url = metadata_dict.get('url')
                            except (json.JSONDecodeError, TypeError):
                                pass

                        # Prepare data for Supabase
                        db_product = {
                            'id': product['id'],
                            'source': 'scraper',
                            'product_url': product_url,
                            'image_url': product['image_url'],
                            'brand': 'Aime',
                            'title': product['title'],
                            'description': product.get('description'),
                            'category': product.get('category'),
                            'gender': 'man',
                            'price': product.get('price'),
                            'currency': 'USD',
                            'second_hand': False,
                            'metadata': product.get('metadata'),
                            'size': product.get('size'),
                            'country': 'US',
                            'embedding': product['embedding']
                        }

                        # Remove None values
                        db_product = {k: v for k, v in db_product.items() if v is not None}

                        # Insert into Supabase with better error handling
                        try:
                            result = self.supabase.table('products').upsert(db_product).execute()
                            
                            # Check if insertion was successful
                            if hasattr(result, 'data') and result.data:
                                logger.info(f"Successfully stored product: {product['title']} (ID: {product['id']})")
                                products_stored += 1
                            elif hasattr(result, 'data') and result.data == []:
                                # Empty response might still be success for upsert
                                logger.info(f"Upsert completed for product: {product['title']} (ID: {product['id']})")
                                products_stored += 1
                            else:
                                logger.warning(f"Unexpected response for product {product['title']}: {result}")
                                
                            # Log the actual response for debugging (only in debug mode)
                            if logger.level == "DEBUG":
                                logger.debug(f"Supabase response: {result}")
                            
                        except Exception as db_error:
                            logger.error(f"Database error storing product {product['title']} (ID: {product['id']}): {db_error}")
                            logger.error(f"Product data: {db_product}")
                            raise  # Re-raise to see the full error

                    except Exception as e:
                        logger.error(f"Error processing product {product.get('id', 'unknown')}: {e}")

                # Small delay between batches
                await asyncio.sleep(1)

        logger.info(f"Processing complete. Successfully stored {products_stored} products.")
        return products_stored

    async def verify_database_insertion(self, expected_count: int):
        """Verify that products were actually inserted into the database."""
        try:
            # Query products with source='scraper' and brand='Aime' from today
            result = self.supabase.table('products').select('id', count='exact').eq('source', 'scraper').eq('brand', 'Aime').execute()
            
            total_count = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
            
            logger.info(f"Database verification: Found {total_count} products in database (expected at least {expected_count})")
            
            if total_count == 0:
                logger.warning("⚠️  No products found in database! There may be an insertion issue.")
            elif total_count < expected_count:
                logger.warning(f"⚠️  Only {total_count} products found, expected {expected_count}. Some products may not have been inserted.")
            else:
                logger.info(f"✅ Successfully verified {total_count} products in database")
                
        except Exception as e:
            logger.error(f"Error verifying database: {e}")

    async def run(self):
        """Main scraping function."""
        logger.info("Starting Aimé Leon Dore scraper...")

        try:
            # Step 1: Scrape all product URLs
            product_urls = await self.scrape_product_urls()
            logger.info(f"Found {len(product_urls)} products to process")

            # Step 2: Process all products
            products_processed = await self.process_products(product_urls)

            # Step 3: Verify database insertion
            logger.info("Verifying database insertion...")
            await self.verify_database_insertion(products_processed)

            logger.info("Scraping completed successfully!")

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


async def main():
    """Main entry point."""
    # Set up logging
    logger.add("scraper.log", rotation="1 day", level="INFO")

    scraper = AimeLeonDoreScraper()
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())