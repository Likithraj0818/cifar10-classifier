"""
Automated Image Scraping Pipeline
===================================
Scrapes Google Images for all 10 CIFAR-10 categories
and organizes them into a dataset/ folder.

Usage:
    python scraper.py
    python scraper.py --images 100      # scrape 100 images per class
    python scraper.py --class airplane  # scrape only one class
"""

import os
import argparse
import shutil
from icrawler.builtin import GoogleImageCrawler

# ── CIFAR-10 categories ───────────────────────────────────────────────────────
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Search queries — more specific = better quality images
SEARCH_QUERIES = {
    "airplane":    "airplane aircraft flying",
    "automobile":  "automobile car vehicle",
    "bird":        "bird animal wildlife",
    "cat":         "cat domestic animal",
    "deer":        "deer animal wildlife",
    "dog":         "dog domestic animal",
    "frog":        "frog amphibian animal",
    "horse":       "horse animal",
    "ship":        "ship vessel ocean",
    "truck":       "truck vehicle transport"
}

def scrape_class(class_name, output_dir, num_images):
    """Scrape images for a single class from Google Images."""
    save_path = os.path.join(output_dir, class_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n📥 Scraping '{class_name}' → {save_path}")
    print(f"   Query : \"{SEARCH_QUERIES[class_name]}\"")
    print(f"   Target: {num_images} images")

    crawler = GoogleImageCrawler(
        storage={"root_dir": save_path},
        log_level=40  # suppress verbose logs (40 = ERROR only)
    )

    crawler.crawl(
        keyword=SEARCH_QUERIES[class_name],
        max_num=num_images,
        file_idx_offset=0
    )

    # Count what was actually downloaded
    downloaded = len([
        f for f in os.listdir(save_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    print(f"   ✅ Downloaded: {downloaded} images")
    return downloaded

def print_summary(output_dir, results):
    """Print a summary table of what was scraped."""
    print("\n" + "="*45)
    print("  SCRAPING SUMMARY")
    print("="*45)
    total = 0
    for cls, count in results.items():
        bar = "█" * (count // 5)
        print(f"  {cls:<12} {count:>4} images  {bar}")
        total += count
    print("-"*45)
    print(f"  {'TOTAL':<12} {total:>4} images")
    print("="*45)
    print(f"\n📁 Dataset saved to: {os.path.abspath(output_dir)}")
    print("\nFolder structure:")
    print(f"  {output_dir}/")
    for cls in results:
        print(f"    ├── {cls}/")
    print()

def main():
    parser = argparse.ArgumentParser(description="Scrape Google Images for CIFAR-10 categories")
    parser.add_argument("--images", type=int, default=50,
                        help="Number of images to scrape per class (default: 50)")
    parser.add_argument("--output", type=str, default="dataset",
                        help="Output directory (default: dataset/)")
    parser.add_argument("--class", dest="single_class", type=str, default=None,
                        help="Scrape only one class (e.g. --class airplane)")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing dataset/ folder before scraping")
    args = parser.parse_args()

    # Validate single class
    if args.single_class and args.single_class not in CLASSES:
        print(f"❌ Unknown class '{args.single_class}'. Choose from: {', '.join(CLASSES)}")
        return

    # Clean if requested
    if args.clean and os.path.exists(args.output):
        print(f"🗑️  Removing existing '{args.output}/' folder...")
        shutil.rmtree(args.output)

    targets = [args.single_class] if args.single_class else CLASSES

    print("="*45)
    print("  CIFAR-10 IMAGE SCRAPING PIPELINE")
    print("="*45)
    print(f"  Classes  : {len(targets)}")
    print(f"  Per class: {args.images} images")
    print(f"  Source   : Google Images")
    print(f"  Output   : {args.output}/")
    print("="*45)

    results = {}
    for cls in targets:
        results[cls] = scrape_class(cls, args.output, args.images)

    print_summary(args.output, results)
    print("✅ Scraping pipeline complete!")
    print("   You can now use the dataset/ folder to train your CNN model.\n")

if __name__ == "__main__":
    main()
