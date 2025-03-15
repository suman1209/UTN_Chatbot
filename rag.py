'''
# Install the package
pip install -U crawl4ai

# Run post-installation setup
crawl4ai-setup

# Verify your installation
crawl4ai-doctor
'''

import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.utn.de/en/study/german-rental-contract/",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())