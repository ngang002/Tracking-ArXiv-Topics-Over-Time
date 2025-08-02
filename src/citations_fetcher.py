import requests
import re

ADS_TOKEN = "1u28melyemIiyqhuXsH71TdmIuR5700vmJrw5DlF"

def strip_arxiv_version(_arXiv_ID):
    return re.sub(r'v\d+$', '', _arXiv_ID)

def fetch_ADS_metadata(arXiv_ID):

    arXiv_ID = strip_arxiv_version(arXiv_ID) 

    headers = {"Authorization": f"Bearer {ADS_TOKEN}"}
    url = f"https://api.adsabs.harvard.edu/v1/search/query?q=arxiv:{arXiv_ID}&fl=citation_count,keyword,bibcode,doi"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch ADS metadata for {arXiv_ID}")
        return None

    docs = response.json().get("response", {}).get("docs", [])
    if not docs:
        return None

    doc = docs[0]
    return {
        "citations": doc.get("citation_count"),
        "keywords": doc.get("keyword", []),
        "raw_json": doc
    }

def fetch_SemanticScholar_metadata(arXiv_ID):

    """
        Wrapping with a check to ensure the orientation of the papers are in a proper format
    """
    arXiv_ID = strip_arxiv_version(arXiv_ID)
    if arXiv_ID and arXiv_ID.startswith("astro-ph/"):
        paper_id = f"{arXiv_ID}"
    elif re.match(r"\d{4}\.\d{5}", arXiv_ID):
        paper_id = f"{arXiv_ID}"
    elif re.match(r"\d{4}\.\d{5}v\d{1}", arXiv_ID):
        paper_id = arXiv_ID.split('v')[0]
    else:
        print("Invalid arXiv ID format")
    print(paper_id)

    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}?fields=citationCount,fieldsOfStudy,externalIds"
    response = requests.get(url)
    print(url, response)
    if response.status_code != 200:
        return None

    data = response.json()
    return {
        "citations": data.get("citationCount"),
        "keywords": data.get("fieldsOfStudy", []),
        "raw_json": data
    }


def fetch_OpenAlex_metadata(arXiv_ID):
    arXiv_ID = strip_arxiv_version(arXiv_ID)
    url = f"https://api.openalex.org/works/https://arxiv.org/abs/{arXiv_ID}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get metadata: {response.status_code}")
        return None