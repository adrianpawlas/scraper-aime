#!/usr/bin/env python3
"""
Simple runner script for the Aimé Leon Dore scraper.
"""

import asyncio
import sys
from scraper import AimeLeonDoreScraper

async def main():
    """Main scraping function."""
    scraper = AimeLeonDoreScraper()

    try:
        # Step 1: Scrape all product URLs
        print("Step 1: Scraping product URLs...")
        product_urls = await scraper.scrape_product_urls()
        print(f"Found {len(product_urls)} products to process")

        # Step 2: Process all products
        print("Step 2: Processing products and generating embeddings...")
        await scraper.process_products(product_urls)

        print("Scraping completed successfully!")

    except Exception as e:
        print(f"Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        print("Starting Aimé Leon Dore Product Scraper...")
        print("This may take several hours for the complete catalog.")
        print("Check scraper.log for detailed logs.")
        asyncio.run(main())
        print("Scraping completed successfully!")
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Scraping failed: {e}")
        sys.exit(1)