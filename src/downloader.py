# Downloads the CUAD dataset and prepares a 50-contract working subset

import shutil
import urllib.request
from urllib.error import URLError, HTTPError
import zipfile
from pathlib import Path
import logging
import random

logger = logging.getLogger(__name__)

CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
DATA_DIR = Path("data")
CONTRACTS_DIR = DATA_DIR / "contracts"
ZIP_FILENAME = "CUAD_v1.zip"


def download_with_retry(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    """Downloads a file with retry logic and progress logging."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading dataset (attempt {attempt}/{retries})...")
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as f:
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                chunk = 1024 * 1024  # 1 MB chunks

                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total > 0:
                        pct = downloaded * 100 // total
                        if pct % 25 == 0:
                            logger.info(f"  {pct}% ({downloaded // (1024*1024)} MB / {total // (1024*1024)} MB)")

            logger.info("Download complete.")
            return

        except (HTTPError, URLError) as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if attempt == retries:
                raise


def generate_mock_contracts(dest_dir: Path, count: int = 50) -> None:
    """
    Generates synthetic legal contract PDFs using reportlab.
    This is the fallback used when the Zenodo download is unavailable.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        raise ImportError("reportlab is required for mock contract generation. Run: uv pip install reportlab")

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating {count} synthetic contracts as fallback...")

    companies = [
        "Acme Industrial Corp", "Globex Software Systems", "Initech Services LLC",
        "Hooli Technologies", "Veer Logistics Ltd", "Wonka Confectionery",
        "Wayne Enterprises", "Stark Industries", "Tyrell Automation Corp",
    ]

    termination_variants = [
        "Either party may terminate this Agreement by giving {notice} days written notice. "
        "Termination for breach requires written notice and a {cure}-day cure period.",
        "This Agreement runs for {term} year(s) and may be terminated immediately upon material breach "
        "not remedied within {cure} business days of written notice.",
        "Either party may terminate without cause upon {notice} days written notice, "
        "provided all outstanding invoices are settled.",
    ]
    confidentiality_variants = [
        "The Receiving Party shall hold all Confidential Information in strict confidence "
        "and shall not disclose it for {duration} years from disclosure.",
        "Both parties agree to mutual non-disclosure. Confidentiality obligations survive "
        "termination and continue for {duration} years.",
        "Proprietary data shared under this Agreement shall not be disclosed to any third party "
        "without prior written consent, and this restriction lasts {duration} years post-termination.",
    ]
    liability_variants = [
        "Except for gross negligence, neither party shall be liable for indirect or consequential damages. "
        "Aggregate liability is capped at ${cap:,}.",
        "Total liability of either party for all claims arising under this Agreement shall not exceed ${cap:,}.",
        "Neither party shall be liable for punitive or exemplary damages. "
        "Maximum liability shall not exceed fees paid in the preceding {months} months.",
    ]

    styles = getSampleStyleSheet()

    for i in range(1, count + 1):
        company_a, company_b = random.sample(companies, 2)
        notice = random.choice([30, 60, 90])
        cure = random.choice([10, 15, 30])
        term = random.choice([1, 2, 3])
        duration = random.choice([2, 3, 5])
        cap = random.choice([10_000, 50_000, 100_000, 250_000])
        months = random.choice([6, 12])

        termination = random.choice(termination_variants).format(notice=notice, cure=cure, term=term)
        confidentiality = random.choice(confidentiality_variants).format(duration=duration)
        liability = random.choice(liability_variants).format(cap=cap, months=months)

        filepath = dest_dir / f"Contract_{i:03d}.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)

        story = [
            Paragraph("<b>COMMERCIAL SERVICE AGREEMENT</b>", styles["Title"]),
            Spacer(1, 20),
            Paragraph(
                f"This Agreement is entered into between <b>{company_a}</b> and <b>{company_b}</b>.",
                styles["Normal"],
            ),
            Spacer(1, 10),
            Paragraph("<b>1. Purpose.</b> The parties agree to collaborate on joint business development projects.", styles["Normal"]),
            Spacer(1, 10),
            Paragraph(f"<b>2. Term &amp; Termination.</b> {termination}", styles["Normal"]),
            Spacer(1, 10),
            Paragraph(f"<b>3. Confidentiality.</b> {confidentiality}", styles["Normal"]),
            Spacer(1, 10),
            Paragraph(f"<b>4. Limitation of Liability.</b> {liability}", styles["Normal"]),
            Spacer(1, 15),
            Paragraph("IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.", styles["Normal"]),
        ]
        doc.build(story)

    logger.info(f"Generated {count} mock contracts in '{dest_dir}'.")


def setup_contracts_subset(
    url: str = CUAD_URL,
    data_dir: Path = DATA_DIR,
    contracts_dir: Path = CONTRACTS_DIR,
    subset_size: int = 50,
) -> None:
    """
    Ensures contracts_dir contains at least subset_size PDF files.
    First tries to download from Zenodo; if that fails, generates synthetic contracts.
    """
    if contracts_dir.exists():
        existing = list(contracts_dir.glob("*.pdf"))
        if len(existing) >= subset_size:
            logger.info(f"Found {len(existing)} PDFs in '{contracts_dir}'. Skipping download.")
            return

    contracts_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / ZIP_FILENAME

    try:
        if not zip_path.exists():
            download_with_retry(url, zip_path)
        else:
            logger.info(f"Using cached zip at '{zip_path}'.")

        logger.info(f"Extracting {subset_size} contracts from zip archive...")
        extracted = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            pdf_entries = sorted([
                f for f in zf.namelist()
                if "full_contract_pdf" in f and f.lower().endswith(".pdf")
            ])

            if not pdf_entries:
                raise FileNotFoundError("No PDF files found inside the zip archive.")

            for entry in pdf_entries[:subset_size]:
                name = Path(entry).name
                with zf.open(entry) as src, open(contracts_dir / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1

        logger.info(f"Extracted {extracted} contracts to '{contracts_dir}'.")

    except Exception as e:
        logger.warning(f"Could not get dataset from Zenodo: {e}")
        logger.warning("Falling back to synthetic contract generation...")
        generate_mock_contracts(contracts_dir, count=subset_size)
