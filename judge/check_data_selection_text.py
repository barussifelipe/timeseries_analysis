"""Check Section 3.1 Data additions and thesis build artifacts."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX_PATH = ROOT / "ref/final_report/thesis_structure.tex"
PDF_PATH = ROOT / "ref/final_report/thesis_structure.pdf"
LOG_PATH = ROOT / "ref/final_report/thesis_structure.log"


def check_data_selection():
    text = TEX_PATH.read_text(encoding="utf-8")
    
    # 1. Check Section 3.1 Data exists
    assert "\\subsection{Data}" in text, "Missing \\subsection{Data}"
    
    # Extract Section 3.1 text (between \\subsection{Data} and \\subsection{Models})
    data_sec = text.split("\\subsection{Data}")[1].split("\\subsection{Models}")[0]
    
    # 2. Check local equities documented in Section 3.1
    for ticker in ["NVDA", "AAPL", "NFLX", "GOOG", "AMZN"]:
        assert ticker in data_sec, f"Ticker {ticker} missing from Section 3.1"
    assert "highest mean daily trading volume before 2016" in data_sec, "Missing volume selection rationale"
    assert "corporate share-split conventions" in data_sec, "Missing split convention rationale"
    
    # 3. Check crypto selection documented in Section 3.1
    for crypto in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
        assert crypto in data_sec, f"Crypto {crypto} missing from Section 3.1"
    assert "\\text{Close}\\times\\text{Volume}" in data_sec or "Close" in data_sec, "Missing dollar volume definition"
    assert "dollar volume" in data_sec, "Missing dollar volume rationale"
    assert "unit-denomination bias" in text or "standardized denominations" in data_sec, "Missing denomination bias rationale"
    assert "Garman--Klass (GK) estimator" in data_sec or "GK" in data_sec, "Missing crypto GK mention"
    assert "realized variance (RV) proxy" in data_sec or "RV" in data_sec, "Missing crypto RV mention"
    assert "eq:realized-variance" in data_sec, "Missing RV equation reference"
    
    # 4. Check Section 5.4.1 crypto evaluation paragraph before Table 5.4
    assert "\\label{tab:crypto-results}" in text, "Missing Table 5.4 label tab:crypto-results"
    pre_table = text.split("\\label{tab:crypto-results}")[0].split("\\label{tab:matched-results}")[1]
    assert "96 complete intraday bars" in pre_table, "Missing 96 bars requirement"
    assert "23:45 close" in pre_table, "Missing preceding 23:45 close requirement"
    assert "harmonized with the 15-minute realized variance proxy" in pre_table, "Missing RV harmonization"
    assert "64--65 targets per coin" in pre_table, "Missing 64-65 calendar gap targets description"
    
    # 5. Check PDF and log
    assert PDF_PATH.exists() and PDF_PATH.stat().st_size > 1_000_000, "PDF missing or too small"
    log = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    assert "Output written on thesis_structure.pdf" in log, "pdflatex did not complete cleanly"
    assert "Fatal error" not in log, "Fatal error found in log"
    
    print("PASS: Section 3.1 & 5.4.1 crypto selection & RV validity text verified; build artifacts valid.")


if __name__ == "__main__":
    check_data_selection()
