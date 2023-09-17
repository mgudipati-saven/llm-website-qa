import requests
from bs4 import BeautifulSoup, SoupStrainer


def scrape(site):
    urls = []

    def scrape_helper(current_site):
        nonlocal urls

        # send a GET request to the website
        resp = requests.get(current_site)

        # parse the HTML content of the page with BeautifulSoup
        soup = BeautifulSoup(resp.text, 'html.parser')

        for a in soup.find_all('a'):
            # check if the link has an href attribute
            if 'href' in a.attrs:
                href = a.attrs['href']
                if href and href.startswith('/') or href.startswith('#'):
                    url = current_site + href

                if url not in urls:
                    urls.append(url)

                    # recursively call the function to scrape the website
                    scrape_helper(site)

    scrape_helper(site)
    return urls


scrape('https://www.remotion.dev/')
