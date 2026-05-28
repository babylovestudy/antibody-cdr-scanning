"""Curated set of well-characterized therapeutic antibody CDR sequences.

Each antibody entry includes:
  - name, target antigen, PDB ID
  - Heavy chain CDR sequences (H1, H2, H3) and Light chain (L1, L2, L3)
  - Known binding-critical residues from literature (for validation)

Sources: IMGT, SAbDab, published alanine scanning studies.
"""

from dataclasses import dataclass, field


@dataclass
class AntibodyCDR:
    name: str
    target: str
    pdb: str
    # Heavy chain
    h1: str = ""
    h2: str = ""
    h3: str = ""
    # Light chain
    l1: str = ""
    l2: str = ""
    l3: str = ""
    # Known hot-spot residues (positions listed as 1-indexed in the CDR)
    hotspot_h3: list[int] = field(default_factory=list)
    # Reference for hotspot data
    reference: str = ""


# ---------------------------------------------------------------------------
# Curated therapeutic antibodies
# ---------------------------------------------------------------------------

ANTIBODIES = {
    "herceptin": AntibodyCDR(
        name="Trastuzumab (Herceptin)",
        target="HER2/neu",
        pdb="1N8Z",
        h1="GFNIKDTYIH",
        h2="RIYPTNGYTRYADSVKG",
        h3="WGGDGFYAMDY",
        l1="RASQDVNTAVAW",
        l2="SASFLYS",
        l3="QQHYTTPPT",
        hotspot_h3=[5, 6, 7, 8],  # D98, F99, Y100, A100a — key contacts
        reference="Kelley & O'Connell (1993); Cho et al. (2003) Nature 421:756",
    ),
    "adalimumab": AntibodyCDR(
        name="Adalimumab (Humira)",
        target="TNF-alpha",
        pdb="3WD5",
        h1="DYAMH",
        h2="AITWNSGHIDYADSVEG",
        h3="VSYLSTASSLDY",
        l1="RASQGIRNYLA",
        l2="AASTLQS",
        l3="QRYNRAPYT",
        hotspot_h3=[1, 2, 3, 10, 11],
        reference="van der Horst et al. (2020)",
    ),
    "bevacizumab": AntibodyCDR(
        name="Bevacizumab (Avastin)",
        target="VEGF-A",
        pdb="1BJ1",
        h1="GYTFTNYGMN",
        h2="WINTYTGEPTYAADFKR",
        h3="YPYYYGTSHWYFDV",
        l1="RASQDISNYLN",
        l2="FTSSLHS",
        l3="QQYSTVPWT",
        hotspot_h3=[1, 2, 3, 4, 5, 6],
        reference="Muller et al. (1998)",
    ),
    "pembrolizumab": AntibodyCDR(
        name="Pembrolizumab (Keytruda)",
        target="PD-1",
        pdb="5B8C",
        h1="GYTFTNYYMY",
        h2="GINPSNGGTNFNEKFKN",
        h3="RHGYWYFDV",
        l1="RASKGVSTSGYSYLH",
        l2="LASYLES",
        l3="QHSRDLPLT",
        hotspot_h3=[1, 2, 3, 4, 5],
        reference="Na et al. (2017)",
    ),
    "nivolumab": AntibodyCDR(
        name="Nivolumab (Opdivo)",
        target="PD-1",
        pdb="5WT9",
        h1="GITFSNSGMH",
        h2="VIWYDGSKRYYADSVKG",
        h3="DRFYGMDV",
        l1="RASQSISSYLN",
        l2="AASSLQS",
        l3="QQSYSTPLT",
        hotspot_h3=[1, 2, 3, 4],
        reference="Tan et al. (2017)",
    ),
    "omalizumab": AntibodyCDR(
        name="Omalizumab (Xolair)",
        target="IgE",
        pdb="2VXQ",
        h1="GYSITSGYSWH",
        h2="SIHYSGDTNYNPSLKS",
        h3="GSHYFGHWHFAV",
        l1="RASQSIGTNIH",
        l2="YASESIS",
        l3="QQSDSWPTT",
        hotspot_h3=[2, 3, 4, 5, 6],
        reference="Wright et al. (2007)",
    ),
    "rituximab": AntibodyCDR(
        name="Rituximab (Rituxan)",
        target="CD20",
        pdb="6VJA",
        h1="GYTFTSYNMH",
        h2="AIYPGNGDTSYNQKFKG",
        h3="VVYYSNSYWYFDV",
        l1="RASSSVSYIH",
        l2="ATSNLAS",
        l3="QQWTSNPPT",
        hotspot_h3=[2, 3, 4, 5, 6],
        reference="Du et al. (2007)",
    ),
}

# Non-antibody reference proteins (for comparison)
REFERENCE_PROTEINS = {
    "lysozyme": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
}
