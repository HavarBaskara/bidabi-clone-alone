import asyncio
import aiohttp
import csv
import os
from aiohttp import ClientSession, ClientTimeout

API_URL = "https://world.openfoodfacts.org/cgi/search.pl"
HEADERS = {"User-Agent": "MyAwesomeApp/1.0"}

OUTPUT_DIR = "data"

CATEGORY = "sugar" #"bread", "milk", "champagnes", "butter" 
TARGET_COUNT = 180
PAGE_SIZE = 100
MAX_PAGES = 50

MAX_CONCURRENT_REQUESTS = 10
MAX_CONCURRENT_IMAGES = 10


# -------------------------
# Helpers
# -------------------------
def get_best_image(product):
    return (
        product.get("image_url")
        or product.get("image_front_url")
        or product.get("image_small_url")
        or product.get("image_thumb_url")
    )


def is_valid_product(product):
    required = ["_id", "product_name", "categories_tags"]
    if not all(product.get(f) for f in required):
        return False
    return bool(get_best_image(product))


def extract_product_info(product):
    return [
        product.get("_id"),
        product.get("product_name"),
        ", ".join(product.get("categories_tags", [])),
        product.get("ingredients_text", ""),
        get_best_image(product)
    ]


# -------------------------
# Async API fetch
# -------------------------
async def fetch_page(session, category, page, page_size, sem):
    params = {
        "action": "process",
        "tagtype_0": "categories",
        "tag_contains_0": "contains",
        "tag_0": category,
        "page": page,
        "page_size": page_size,
        "json": 1
    }

    async with sem:
        try:
            async with session.get(API_URL, params=params) as resp:
                data = await resp.json()
                return data.get("products", [])
        except Exception as e:
            print(f"⚠ Erreur API page {page} :", e)
            return []


# -------------------------
# Async image download
# -------------------------
async def download_image(session, url, image_id, sem, folder="data/images/sugar"):
    if not url:
        return

    os.makedirs(folder, exist_ok=True)

    ext = url.split(".")[-1].split("?")[0]
    filename = os.path.join(folder, f"{image_id}.{ext}")

    if os.path.exists(filename):
        return

    async with sem:
        try:
            async with session.get(url) as resp:
                content = await resp.read()
                with open(filename, "wb") as f:
                    f.write(content)
        except Exception as e:
            print(f"⚠ Impossible de télécharger {url} :", e)


# -------------------------
# Main scraping logic
# -------------------------
async def scrape(category, target_count, page_size, max_pages):
    timeout = ClientTimeout(total=60)
    sem_api = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    sem_img = asyncio.Semaphore(MAX_CONCURRENT_IMAGES)

    async with ClientSession(headers=HEADERS, timeout=timeout) as session:
        valid_products = []
        image_tasks = []
        page = 1

        while len(valid_products) < target_count and page <= max_pages:
            print(f"→ Téléchargement page {page}…")

            products = await fetch_page(session, category, page, page_size, sem_api)
            if not products:
                print("Aucun produit trouvé sur cette page.")
                break

            for product in products:
                if is_valid_product(product):
                    info = extract_product_info(product)
                    valid_products.append(info)

                    image_url = info[-1]
                    image_id = info[0]

                    task = asyncio.create_task(
                        download_image(session, image_url, image_id, sem_img)
                    )
                    image_tasks.append(task)

                    if len(valid_products) >= target_count:
                        break

            page += 1

        await asyncio.gather(*image_tasks)
        return valid_products


# -------------------------
# CSV export
# -------------------------
def save_to_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["foodId", "label", "category", "foodContentsLabel", "image"])
        writer.writerows(rows)


# -------------------------
# Entry point
# -------------------------
def main():
    products = asyncio.run(scrape(CATEGORY, TARGET_COUNT, PAGE_SIZE, MAX_PAGES))
    output_file = f"{OUTPUT_DIR}/metadata_{CATEGORY}_{TARGET_COUNT}.csv"
    save_to_csv(output_file, products)
    print(f"✔ Fichier {output_file} créé. Produits valides collectés : {len(products)}")


if __name__ == "__main__":
    main()
