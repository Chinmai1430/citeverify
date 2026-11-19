from pathlib import Path

import pymupdf


# Inputs: PyMuPDF page, starting y-coordinate, text content, font settings, and spacing below.
# Outputs: Next y-coordinate after the drawn text block (raises ValueError if page overflows).
def _draw_flow_box(
    page: pymupdf.Page,
    y: float,
    text: str,
    fontsize: float = 10.5,
    fontname: str = "helv",
    color: tuple[float, float, float] = (0.1, 0.1, 0.1),
    gap: float = 8.0,
    bottom_limit: float = 750.0,
) -> float:
    rect = pymupdf.Rect(64, y, 548, bottom_limit)
    unused = page.insert_textbox(
        rect,
        text,
        fontsize=fontsize,
        fontname=fontname,
        color=color,
        align=pymupdf.TEXT_ALIGN_LEFT,
    )
    if unused < 0:
        raise ValueError(f"Text overflowed PDF page at y={y}: {text[:50]}...")
    return (bottom_limit - unused) + gap


# Inputs: Output file path where the 2-page realistic academic demo PDF will be saved.
# Outputs: Path to the generated PDF file.
def build_demo_paper(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()

    # --- Page 1: Title, Abstract, Body with 5 Planted Claims, Discussion ---
    page1 = doc.new_page(width=612, height=792)
    y = 42.0

    y = _draw_flow_box(
        page1,
        y,
        "CiteVerify Demo Corpus  |  Rapid Clinical Literature Review (2026)",
        fontsize=8.5,
        fontname="hebo",
        color=(0.35, 0.35, 0.35),
        gap=8.0,
    )
    y = _draw_flow_box(
        page1,
        y,
        "Evaluating Deep Learning Diagnostics and Pharmacological Interventions in Modern Clinical Medicine",
        fontsize=15.0,
        fontname="hebo",
        gap=6.0,
    )
    y = _draw_flow_box(
        page1,
        y,
        "A. Sharma, M. Chen, and R. Patel  -  Department of Clinical Informatics",
        fontsize=9.5,
        fontname="heit",
        color=(0.25, 0.25, 0.25),
        gap=8.0,
    )
    page1.draw_line(
        pymupdf.Point(64, y),
        pymupdf.Point(548, y),
        color=(0.75, 0.75, 0.75),
        width=0.8,
    )
    y += 14.0

    y = _draw_flow_box(page1, y, "Abstract", fontsize=11.5, fontname="hebo", gap=4.0)
    y = _draw_flow_box(
        page1,
        y,
        (
            "Hospital teams routinely review diagnostic software alongside pharmacological protocols "
            "before adopting new clinical workflows. This brief synthesis examines selected literature "
            "spanning medical imaging algorithms, emergency triage tools, and inpatient respiratory "
            "therapies. Careful verification of primary citations remains central to safe patient care."
        ),
        gap=12.0,
    )

    y = _draw_flow_box(page1, y, "1. Introduction", fontsize=11.5, fontname="hebo", gap=4.0)
    y = _draw_flow_box(
        page1,
        y,
        (
            "Modern healthcare systems rely heavily on peer-reviewed literature when updating "
            "diagnostic pathways and inpatient treatment guidelines. Because clinical manuscripts "
            "frequently combine automated decision tools with drug trial summaries, citation "
            "accuracy directly shapes bedside policy."
        ),
        gap=12.0,
    )

    y = _draw_flow_box(
        page1,
        y,
        "2. Deep Learning in Medical Imaging",
        fontsize=11.5,
        fontname="hebo",
        gap=4.0,
    )
    y = _draw_flow_box(
        page1,
        y,
        (
            "In an evaluation of retinal fundus photographs from adults with diabetes, a deep "
            "learning algorithm achieved 97.5% sensitivity and 93.4% specificity for detecting "
            "referable diabetic retinopathy on the EyePACS-1 validation dataset [1]. "
            "Multi-center prospective randomized clinical trials across 30 countries confirmed "
            "that an AI breast cancer screening system safely replaced human radiologists in "
            "routine hospital practice [2]."
        ),
        gap=12.0,
    )

    y = _draw_flow_box(
        page1,
        y,
        "3. Inpatient Therapeutics and Triage Systems",
        fontsize=11.5,
        fontname="hebo",
        gap=4.0,
    )
    y = _draw_flow_box(
        page1,
        y,
        (
            "In the RECOVERY randomized trial of hospitalized COVID-19 patients, dexamethasone "
            "increased overall 28-day mortality to 62.4% compared with 25.7% in the usual care "
            "group [3]. An autonomous clinical transformer model reduced emergency department "
            "misdiagnosis rates by 74.2% across forty European hospitals in a 2024 multi-center "
            "audit [4]. A multinational hospital registry analysis reported that hydroxychloroquine "
            "regimens were associated with decreased in-hospital survival and higher ventricular "
            "arrhythmia rates in COVID-19 patients [5]."
        ),
        gap=12.0,
    )

    y = _draw_flow_box(page1, y, "4. Discussion", fontsize=11.5, fontname="hebo", gap=4.0)
    _draw_flow_box(
        page1,
        y,
        (
            "Clinical governance committees must inspect both original study designs and "
            "post-publication editorial notices prior to drafting institutional recommendations. "
            "Automated citation auditing offers a practical safeguard during rapid literature triage."
        ),
        gap=12.0,
    )

    # --- Page 2: References Section ---
    page2 = doc.new_page(width=612, height=792)
    y = 56.0
    y = _draw_flow_box(page2, y, "References", fontsize=13.0, fontname="hebo", gap=6.0)
    page2.draw_line(
        pymupdf.Point(64, y),
        pymupdf.Point(548, y),
        color=(0.75, 0.75, 0.75),
        width=0.8,
    )
    y += 14.0

    references = [
        (
            "[1] Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., et al. "
            "(2016). Development and Validation of a Deep Learning Algorithm for Detection of Diabetic "
            "Retinopathy in Retinal Fundus Photographs. JAMA, 316(22), 2402-2410. "
            "https://doi.org/10.1001/jama.2016.17216"
        ),
        (
            "[2] McKinney, S. M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H., "
            "et al. (2020). International evaluation of an AI system for breast cancer screening. "
            "Nature, 577(7788), 89-94. https://doi.org/10.1038/s41586-019-1799-6"
        ),
        (
            "[3] Horby, P., Lim, W. S., Emberson, J. R., Mafham, M., Bell, J. L., Linsell, L., et al. "
            "(2021). Dexamethasone in Hospitalized Patients with Covid-19. New England Journal of "
            "Medicine, 384(8), 693-704. https://doi.org/10.1056/NEJMoa2021436"
        ),
        (
            "[4] Kowalski, T., & Verma, S. (2024). Autonomous Diagnostic Transformers for Emergency "
            "Triage Across European Hospitals. International Journal of Clinical AI Systems, "
            "18(4), 112-129."
        ),
        (
            "[5] Mehra, M. R., Desai, S. S., Ruschitzka, F., & Patel, A. N. (2020). Hydroxychloroquine "
            "or chloroquine with or without a macrolide for treatment of COVID-19: a multinational "
            "registry analysis. The Lancet, 395(10240). https://doi.org/10.1016/S0140-6736(20)31180-6"
        ),
    ]

    for ref in references:
        y = _draw_flow_box(page2, y, ref, fontsize=10.0, gap=12.0)

    doc.save(output_path)
    doc.close()
    return output_path


if __name__ == "__main__":
    target = Path(__file__).parent / "demo_paper.pdf"
    saved = build_demo_paper(target)
    print(f"Generated {saved} successfully.")