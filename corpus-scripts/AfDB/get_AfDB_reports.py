import requests
import json
import os
import re
import hashlib
from utils.utils import get_request,run_threaded_for
from bs4 import BeautifulSoup
from tqdm import tqdm

file_path = os.path.dirname(os.path.abspath(__file__))
document_types = ['completion report of the project','pcr','project completion report','completion report']
header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36'}

documents_page_url = 'https://www.afdb.org/en/documents/project-operations/projectprogramme-completion-reports'

p_bar = None


def match_target(document_type: str, document_target: list[str]):
    '''Check if document type is in target'''
    treated_document_type = document_type.lower().strip()

    for target in document_target:
        # print('Matching',target,'|', treated_document_type)

        # multi word
        ## check if each word is in list and in order
        if len(target.split(' ')) > 1:
            target_words = target.split(' ')
            target_word_indexes = []
            valid = True
            for target_word in target_words:
                try:
                    found_index = treated_document_type.find(target_word)
                    if target_word in treated_document_type:
                        if target_word_indexes and found_index < target_word_indexes[-1]:
                            valid = False
                            break
                        target_word_indexes.append(found_index)
                    else:
                        valid = False
                        break
                except Exception as e:
                    valid = False
                    break
            if valid:
                return target
            
        # single word
        elif target in treated_document_type:
            return target
        
    return None


def get_AfDB_document(link: str):
    '''Download document from url page'''
    print('Getting document: ',link)

    r = get_request(link,headers=header)
    if r:
        soup = BeautifulSoup(r.text, 'lxml')

        # get pdf vizualizer frame
        frame = soup.find('iframe',{'class':'pdf'})
        if frame:
            # get '#document' element
            document_url = frame['data-src']

            # create document folder
            page_title = soup.find('title').text
            folder_hash = hashlib.md5(page_title.encode('utf-8')).hexdigest()
            if not os.path.exists(f'{file_path}/data/files/{folder_hash}'):
                os.mkdir(f'{file_path}/data/files/{folder_hash}')

            # download document
            r = get_request(document_url,headers=header)
            if r:
                with open(f'{file_path}/data/files/{folder_hash}/pcr.pdf', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=128):
                        f.write(chunk)

            # save metadata
            metadata = {
                'title': page_title,
                'link': link
            }
            with open(f'{file_path}/data/files/{folder_hash}/metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f)




def get_page_documents(soup: BeautifulSoup):
    '''Get all project completion documents from a page in AfDB'''
    table = soup.find('tbody')
    links = table.find_all('a',href=True)
    for link in links:
        link_text = link.text
        # check if document is of desired type
        if match_target(link_text,document_types):
            link = f'https://www.afdb.org{link["href"]}'
            # get document
            get_AfDB_document(link)
        p_bar.update(1)



def get_project_documents():
    '''Get all project completion documents from AfDB'''
    global p_bar

    # create data folder
    if not os.path.exists(f'{file_path}/data/files'):
        os.makedirs(f'{file_path}/data/files')


    page = 1
    r = get_request(documents_page_url,headers=header)
    if r:
        soup = BeautifulSoup(r.text, 'lxml')
        text = soup.get_text()
        # get number of documents
        documents_meta = re.search(r'Displaying (\d+).+?(\d+).+? of (\d+)',text)
        n_documents = documents_meta.group(3)
        first_doc_number = documents_meta.group(1)
        last_doc_number = int(documents_meta.group(2))
        n_documents = int(n_documents)
        print('Number of documents:',n_documents)

        p_bar = tqdm(total=n_documents,desc='Downloading documents')
        # get documents
        get_page_documents(soup)
        # request all pages
        while last_doc_number < n_documents:
            page += 1
            page_url = f'{documents_page_url}?page={page}'
            print('Requesting',page_url)
            r = get_request(page_url,headers=header)
            if r:
                soup = BeautifulSoup(r.text, 'lxml')
                text = soup.get_text()
                # get number of documents
                documents_meta = re.search(r'Displaying (\d+).+?(\d+).+? of (\d+)',text)
                n_documents = documents_meta.group(3)
                first_doc_number = documents_meta.group(1)
                last_doc_number = int(documents_meta.group(2))
                n_documents = int(n_documents)

                if last_doc_number < n_documents:
                    # get documents
                    get_page_documents(soup)
                




if __name__ == '__main__':
    get_project_documents()


