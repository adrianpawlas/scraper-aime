#!/usr/bin/env python3
"""
Test version of the scraper that only processes a few products.
"""

import asyncio
import sys
from scraper import AimeLeonDoreScraper

async def test_scraper():
    """Test the scraper with just a few products."""
    scraper = AimeLeonDoreScraper()

    try:
        # Only get a few product URLs for testing - use non-headless to see what's happening
        product_urls = await scraper._scrape_product_urls_headless()
        test_urls = product_urls[:2]  # Only test with 2 products

        print(f"Testing with {len(test_urls)} products: {test_urls}")

        if test_urls:
            # Process just these few products
            await scraper.process_products(test_urls, batch_size=1)
            print("Test completed successfully!")
        else:
            print("No product URLs found - website may have changed")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scraper())