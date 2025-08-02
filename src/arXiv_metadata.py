import feedparser
import urllib

import sqlite3
from datetime import datetime
import time

import logging

from citations_fetcher import *
from category_maps import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARXIVE_DATABASE_NAME = "arxive_astro_3000.db"

# Set up database
def initialize_db(db_name=ARXIVE_DATABASE_NAME):
    
    _conn = sqlite3.connect(db_name)
    cur = _conn.cursor()

    # Drop the table if it exists (for development / schema changes)
    cur.execute('DROP TABLE IF EXISTS arxiv_papers')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS arxiv_papers (
            arXiv_ID TEXT,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            published TEXT,
            updated TEXT,
            categories TEXT,
            citations INTEGER,
            keywords TEXT,
            external_json TEXT
        )
    ''')
    _conn.commit()
    return _conn

def write_to_db(listOfPapers, conn, db_name=ARXIVE_DATABASE_NAME):

    cur = conn.cursor()
    for paper in listOfPapers:
        cur.execute('''
            INSERT OR IGNORE INTO arxiv_papers
            (arXiv_ID, title, abstract, authors, published, updated, categories)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            paper.arXiv_ID,
            paper.title,
            paper.abstract,
            paper.authors,
            paper.published,
            paper.updated,
            paper.categories
        ))

    conn.commit()
    conn.close()
    print(f"Inserted {len(listOfPapers)} papers into {db_name}")
    return None

class ArXivMetadataFetcher:

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit_delay = 3  # seconds between requests to respect arXiv's rate limits
        
        # arXiv astrophysics categories
        self.astro_categories = {
            'astro-ph': 'Astrophysics (general)',
            'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
            'astro-ph.EP': 'Earth and Planetary Astrophysics',
            'astro-ph.GA': 'Astrophysics of Galaxies',
            'astro-ph.HE': 'High Energy Astrophysical Phenomena',
            'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
            'astro-ph.SR': 'Solar and Stellar Astrophysics'
        }

        self.feed = []
        self.papers = []



    def _build_query(self, _categories, _titles, _authors, _abstracts, _start_date, _end_date, _start, _batch_size):

        BASE_URL = "http://export.arxiv.org/api/search_query?"
        
        """ 
            QUERY: http://export.arxiv.org/api/query?search_query=cat:CATEGORIES+AND+
                                                                    ti:TITLE+AND+
                                                                    au:AUTHOR+AND+
                                                                    abs:ABSTRACT+AND+
                                                                    submittedDate:[START TO END]
                                                                    &start=X&max_results=Y
        """
        _query_components = []
        # Category logic
        if _categories:
            for category in _categories:
                if category in CATEGORY_MAP:
                    # Expand top-level category into subcategories
                    cats = CATEGORY_MAP[category]
                    cat_query = "+OR+".join([f"cat:{c}" for c in cats])
                    _query_components.append(cat_query)
                else:
                    _query_components.append(f"cat:{category}")
        else:
            print("No category specified — querying across all fields.")

        if _titles:
            for _title in _titles:
                _query_components.append(f'+OR+ti:"{_title}"')
        if _authors:
            for _author in _authors:
                _query_components.append(f'+OR+au:"{_author}"')
        if _abstracts: 
            for _abstract in abstracts:
                _query_components.append(f'+OR+abs:"{_abstract}"')
        
        # print(f"query_components = {_query_components}")
        _query_string = "+OR+".join(_query_components)

        if _start_date and _end_date:
            # Format: submittedDate:[YEAR-MONTH-DAY TO YEAR-MONTH-DAY]
            _query_string += f"+AND+submittedDate:[{_start_date}+TO+{_end_date}]"
        elif _start_date:
            _query_string += f"AND+submittedDate:[{_start_date}+TO+*]"
        elif _end_date:
            _query_string += f"AND+submittedDate:[*+TO+{_end_date}]"
        else:
            print("No date range specified. Fetching most recent results.")

        params = {
            "search_query": _query_string,
            "start": _start,
            "max_results": _batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }


        # FEED_URL = BASE_URL+_query_string # +f"&start={_start}&max_results={_max_results}" # f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
        FEED_URL = "https://export.arxiv.org/api/query?search_query=(cat:astro-ph+OR+cat:astro-ph.CO+OR+cat:astro-ph.GA+OR+cat:astro-ph.HE+OR+cat:astro-ph.SR+OR+cat:astro-ph.IM)"\
                                                                    +f"+AND+submittedDate:[{_start_date}+TO+{_end_date}]"\
                                                                    +f"&start={_start}&max_results={_batch_size}" 
        # FEED_URL = "https://export.arxiv.org/api/query?search_query=cat:astro-ph.CO+OR+cat:astro-ph.GA+AND+submittedDate:[202301010600+TO+202401010600]"+f"&start={_start}&max_results={_max_results}" 

        logger.info(f"Query: {_query_string}")
        logger.info(f"Feed URL: {FEED_URL}")

        return FEED_URL

    def fetch_arxiv_papers(self, selection_dict):

        categories = selection_dict['categories']
        titles    = selection_dict['titles']
        authors   = selection_dict['authors']
        abstracts = selection_dict['abstracts']
        start_date    = selection_dict['start_date']
        end_date      = selection_dict['end_date']
        start    = selection_dict['start']
        max_results   = selection_dict['max_results']
        batch_size = selection_dict['batch_size']
        delay = selection_dict['delay']
        
        logging.info(f"Start Date: {start_date} \t End Date: {end_date}")

        parsed_feed = []
        for start in range(start, max_results, batch_size):
            # logger.info(f"\nFetching batch starting at {start} and ending at {start+batch_size-1}")
            feed_URL = self._build_query(categories, titles, authors, abstracts, start_date, end_date, start, batch_size)
       
            parsed_batch = feedparser.parse(feed_URL)['entries']
            logger.info(f"Number of Entries Parsed: {len(parsed_batch)}")
            for _ in parsed_batch: parsed_feed.append(_)

            # Add delay to respect API rate limits
            logger.info(f"Sleeping for {delay} seconds to avoid rate limits...")
            time.sleep(delay)

        return parsed_feed


class Papers: 

    def __init__(self, arXiv_ID, title, abstract, authors, citations, published, updated, categories, keywords): 
        # Extract metadata from each entry
        self.arXiv_ID = arXiv_ID
        self.title = title
        self.abstract = abstract
        self.authors = authors 
        self.citations = citations
        self.published = published
        self.updated = updated
        self.categories = categories
        self.keywords = keywords


def extract_metadata(arXivMetadata):

    """
        Input: Metadata fpr
    """
    arXiv_ID = arXivMetadata.id.split('/')[-1]
    arXiv_IDFull = arXivMetadata.id
    
    title = arXivMetadata.title.strip().replace('\n', ' ')
    abstract = arXivMetadata.summary.strip().replace('\n', ' ')
    authors = ', '.join(author.name for author in arXivMetadata.authors)
    published = arXivMetadata.published
    updated = arXivMetadata.updated
    categories = ', '.join(tag['term'] for tag in arXivMetadata.tags)

    # print(arXiv_ID, arXiv_IDFull)
    # ADS_metadata = fetch_ADS_metadata(arXiv_ID)
    # print(ADS_metadata)
    # semanticScholar_metadata = fetch_SemanticScholar_metadata(arXiv_ID)
    # print(semanticScholar_metadata)
    # openAlex_metadata = fetch_OpenAlex_metadata(arXiv_ID)
    # print(openAlex_metadata)
    
    # paper = Papers(arXiv_ID, title, abstract, authors, ADS_metadata['citations'], published, updated, categories, ADS_metadata['keywords'])
    paper = Papers(arXiv_ID, title, abstract, authors, 0, published, updated, categories, "")
    
    return paper


def generatePaperList(_arXivList):

    _listOfPapers = []
    for _, arXivMetadata in enumerate(_arXivList):
        
        _listOfPapers.append(extract_metadata(arXivMetadata))

        # Set the rate limit to Semantic Scholar's rate limit (100 requests every 5 minutes)
        # if _ == 99: time.wait(300)

    return _listOfPapers 


if __name__ == "__main__":

    DB_NAME = ARXIVE_DATABASE_NAME
    conn = initialize_db(db_name=DB_NAME) 
    listOfPapers = []

    for YEAR in range(1995, 2025+1):
        for MONTH in range(1, 13, 1):

            # Initialize fetcher
            fetcher = ArXivMetadataFetcher()

            SELECTION_DICT = {"categories": ["astro-ph"],
                                "titles": None, 
                                "authors": None, 
                                "abstracts": None, 
                                "start_date": f"{YEAR}{str(MONTH).zfill(2)}010600", "end_date": f"{YEAR}{str(MONTH).zfill(2)}310600", 
                                # "start_date": f"{str(YEAR).zfill(4)}{str(MONTH).zfill(2)}010600", "end_date": f"{str(YEAR).zfill(4)}{str(MONTH).zfill(2)}280600", 
                                # "start_date": None, "end_date": None, 
                                "start": 0, "max_results": 3000, "batch_size": 1000, "delay": 10}

            arXivList = fetcher.fetch_arxiv_papers(SELECTION_DICT)

            listOfPapers_MonthYear = generatePaperList(arXivList) # [extract_metadata(arXivMetadata) for arXivMetadata in arXivList]
            logger.info(f"Number of papers added to database in {str(MONTH).zfill(2)}/{YEAR}: {len(listOfPapers)}")

            for _ in listOfPapers_MonthYear: listOfPapers.append(_)
            logger.info(f"Number of papers in database: {len(listOfPapers)}")

            print("\n \n \n")
        
        # for p in listOfPapers:
        #     logger.info(f"{p.arXiv_ID}, {p.title}, {p.categories}, {p.published}, {p.updated}")

    write_to_db(listOfPapers, conn, DB_NAME)
    # print(feed)











