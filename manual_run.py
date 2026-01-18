#!/usr/bin/env python3
"""
Manual runner script for the Aimé Leon Dore scraper.
Loads environment variables and runs the scraper.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from scraper import AimeLeonDoreScraper

# Load environment variables
load_dotenv()

def check_env_vars():
    """Check if required environment variables are set."""
    required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment:")
        print("- SUPABASE_URL=your-supabase-url")
        print("- SUPABASE_ANON_KEY=your-anon-key")
        return False

    return True

async def main():
    """Main scraping function."""
    if not check_env_vars():
        sys.exit(1)

    scraper = AimeLeonDoreScraper()

    try:
        # Step 1: Scrape all product URLs
        print("🔍 Step 1: Scraping product URLs...")
        product_urls = await scraper.scrape_product_urls()
        print(f"📊 Found {len(product_urls)} products to process")

        # Step 2: Process all products
        print("⚙️  Step 2: Processing products and generating embeddings...")
        await scraper.process_products(product_urls)

        print("✅ Scraping completed successfully!")

    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        print("🚀 Starting Aimé Leon Dore Product Scraper...")
        print("📝 Check scraper.log for detailed logs.")
        print("⏰ This may take several hours for the complete catalog.")
        print("=" * 60)

        asyncio.run(main())

        print("=" * 60)
        print("✅ Scraping completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Scraping interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        sys.exit(1)