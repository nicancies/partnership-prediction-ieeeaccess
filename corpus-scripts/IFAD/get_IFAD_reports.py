'''Script to obtain IFAD reports'''

import json
import re
import sys
import jellyfish
import pandas
import argparse
import os
import fitz
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils.utils import *


file_path = os.path.dirname(os.path.abspath(__file__))
debug_mode = False
projects_bar = tqdm(total=0)

IFAD_projects_list_url = 'https://www.ifad.org/en/web/operations/projects-and-programmes'
IFAD_projects_url = 'https://www.ifad.org/en/web/operations/-/project/'
IFAD_ioe_url = 'https://ioe.ifad.org/en/' # independent office of evaluation

header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36'}


default_project_types = ['project completion report', 'final project design report', 'final design report',
                         'project design report', 'mid term review', 'supervision report','president\'s report'
                         ,'(mid-term) review report','supervision mission','interim phase review','med-term review']

evaluation_project_types = ['ppe', 'ppa','pcr','project performance evaluation','project performance assessment','project completion report']


country_iso_codes_path = f'{file_path}/ifad_countries_iso3_mapping.json'
country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))


ioe_country_name_mapping = json.load(open(f'{file_path}/ioe_country_name_mapping.json','r',encoding='utf-8'))

def process_args():
    parser = argparse.ArgumentParser(
        description="Script to obtain IFAD reports"
    )
    parser.add_argument("-pd","--project_documents"             , action='store_true', default=False                                            , help="Get default projects' documents")
    parser.add_argument("-ped","--project_evaluation_documents" , action='store_true', default=False                                            , help="Get projects' evaluation documents")
    parser.add_argument("-t","--threads"                        , type=int, nargs="?", default=1                                                , help="Number of threads to use. Default=1")
    parser.add_argument("-d","--debug"                          , action='store_true', default=False                                            , help="Debug mode. Default=False")
    return parser.parse_args()

def is_completion_report_digest(pdf_path):
    '''Diferentiate between completion report and completion report digest'''
    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = fitz.open(pdf_file)
        for page_num in range(len(pdf_reader)):
            text = ''
            page = pdf_reader[page_num]
            blocks = page.get_text('blocks',
                                flags=fitz.TEXTFLAGS_BLOCKS|fitz.TEXT_DEHYPHENATE)
            for block in blocks:
                text += block[4]
            text = text.lower()
            if re.search(r'(completion\s+report\s+digest)|(project\s+completion\s+digests)', text):
                return True
    except Exception as e:
        pass
    return False

def get_ifad_closed_projects():
    '''Get closed projects from ifad.org
    
    Returns list of projects'''
    project_list = []
    response = get_request(IFAD_projects_list_url,headers=header)
    if response:
        soup = BeautifulSoup(response.text, 'lxml')
        projects_table = soup.find('div',{'class':'project-browser-tab tab4'})
        projects = projects_table.find_all('div',{'class':'table-row'})

        for project in projects:
            columns = project.find_all('div')
            name = columns[0].text.strip()
            id = columns[1].text.strip()
            country = columns[2].text.strip()
            date = columns[3].text.strip()
            p = {}
            p['name'] = name
            p['id'] = id
            p['country'] = country
            p['date'] = date
            project_list.append(p)

    return project_list


def get_downloaded_projects():
    '''Create list of projects in data folder'''
    project_list = []
    data_folder_path = f'{file_path}/data'
    if os.path.isdir(data_folder_path):
        for country_folder in os.listdir(data_folder_path): 
            if os.path.isdir(f'{data_folder_path}/{country_folder}'):
                for project_folder in os.listdir(f'{data_folder_path}/{country_folder}'):
                    if os.path.isdir(f'{data_folder_path}/{country_folder}/{project_folder}'):
                        project_metadata_path = f'{data_folder_path}/{country_folder}/{project_folder}/metadata.json'
                        if os.path.isfile(project_metadata_path):
                            project_metadata = json.load(open(project_metadata_path))
                            project_list.append(project_metadata)
    else:
        print('Data folder not found')

    return project_list

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


def download_document(file: str, project_id: str, link: str, document_name: str):
    '''Download document from url page
    
    - follow link and get download url'''

    # follow link
    download_link = None
    r = get_request(link,headers=header)
    if r:
        soup = BeautifulSoup(r.text, 'lxml')
        # get links
        buttons = soup.find_all('button')
        # find download link
        for button in buttons:
            if 'download' in button.text.lower():
                arguments = button['onclick'].split("'")
                for argument in arguments:
                    if 'https' in argument or '/documents' in argument:
                        download_link = argument
                        if not 'https' in download_link:
                            download_link = f'https://www.ifad.org{download_link}'
                        break
                if download_link:
                    break

        if download_link:
            print(f'Downloading {document_name} from {download_link}')
            r = get_request(download_link,headers=header)
            if r:
                filename = file
                # check if needs id for file
                if os.path.exists(file):
                    file_dir = os.path.dirname(file)
                    filename = f'{file_dir}/{os.path.basename(file).split(".")[0]}'
                    id = 2
                    while os.path.exists(f'{filename}_{id}.pdf'):
                        id += 1
                    filename = f'{filename}_{id}.pdf'
                    
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=128):
                        f.write(chunk)



def convert_total_project_cost(cost:str)->float:
    '''Convert total project cost to float'''
    budget_value = 0
    value = re.search(r'US\$\s*((\d+[.,]?)+)\s*(million)?',cost)
    if value:
        budget = value.group(1).replace(',','')
        split_budget = budget.split('.')
        if len(split_budget) > 1:
            budget_not_fract = budget.split('.')[0]
            budget_fract = budget.split('.')[1]
            budget_value = int(budget_not_fract) + int(budget_fract) / 100
        else:
            budget_value += int(budget)
        
        million_budget = True if value.group(3) else False
        if million_budget:
            budget_value *= 1000000
            budget_value = round(budget_value,2)

    return budget_value

    
def get_default_documents(project_id:str, project_folder: str):
    '''Get default documents for single project'''
    project_url = f'{IFAD_projects_url}{project_id}'
    print(f'Getting default documents from {project_url}')
    r = get_request(project_url,headers=header)
    if r is not None:
        soup = BeautifulSoup(r.text, 'lxml')

        # extra metadata
        metadata_table = soup.find('div',{'class':'m-3 project-row'})
        metadata = metadata_table.find_all('dt')

        print(f'Found {len(metadata)} metadata')
        extra_metadata = {}
        for meta in metadata:
            element = meta
            classes = element.find_next_sibling()['class']
            while classes and 'project-row-text' in classes:
                key = meta.text.lower().strip()
                value = meta.find_next_sibling().text.strip()
                extra_metadata[key] = value
                element = element.find_next_sibling()
                classes = element['class'] if element else []

        # add metadata
        metadata_file = f'{project_folder}/metadata.json'
        metadata = json.load(open(metadata_file,'r'))
        metadata.update(extra_metadata)

        country = metadata['country']
        project_country = country_iso_codes[country]
        metadata['country_iso3'] = project_country
        metadata['end_date'] = metadata['duration'].split('-')[-1].strip()
        metadata['total_budget'] = convert_total_project_cost(metadata['total project cost'])

        json.dump(metadata, open(metadata_file,'w',encoding='utf-8'), indent=4)

        # relevant document tables
        relevant_document_tables_names = ['pcr digest','president\'s reports','interim (mid-term) review report',
                                          'supervision and implementation support documents','project completion report',
                                          'project design reports']
        document_type_map = {
            'pcr digest':'pcrd',
            'president\'s reports':'pr',
            'interim (mid-term) review report':'mtr',
            'supervision and implementation support documents':'sr',
            'project completion report':'pcr',
            'project design reports':'pdr'
        }
        tables = soup.find_all('h2')
        for table_name in relevant_document_tables_names:
            table = None
            for t in tables:
                if table_name in t.find(string=True,recursive=False).lower().strip():
                    table = t
                    # get correct parent
                    while 'portlet-content' not in table['class']:
                        table = table.parent
                        
                    break
            if not table:
                continue
            
            # h2 inside main div
            table = table.parent

            # get documents
            # get all links
            links = table.find_all('a',href=True)
            # get all documents
            documents = links


            # download documents
            for document in documents:
                link = document['href']
                document_name = document.text
                
                document_type = None
                document_type = document_type_map.get(table_name)

                if document_type == 'pdr':
                    if 'final' in document_name.lower():
                        document_type = 'fpdr'

                download_document(f'{project_folder}/{document_type}.pdf', project_id, link,document_name)


def is_evaluation_document(doc_url:str)->str:
    '''Check if document is an evaluation document. Return document type or None'''
    doc_type = None

    print(f'|-------| Checking if {doc_url} is an evaluation document')

    doc_r = get_request(doc_url,headers=header)
    if doc_r:
        # analyse pdf with fitz
        try:
            doc = fitz.open(stream=doc_r.content, filetype="pdf")
            
            for page in doc:
                if 'project performance evaluation' in page.get_text().lower():
                    doc_type = 'ppe'
                    break
                elif 'project performance assessment' in page.get_text().lower():
                    doc_type = 'ppa'
                    break
        except Exception as e:
            print('|-------| Error while analysing pdf',e)

    return doc_type


def get_ioe_documents(project_name:str,project_country:str,project_date:str,project_duration:list, project_folder: str):
    '''Get independent evaluation documents for single project'''
    ioe_url = f'{IFAD_ioe_url}search?q={project_name}'
    print(f'Getting ioe documents from {ioe_url}  ||||  {project_name} | {project_country} | {project_date} | {project_duration}')
    r = get_request(ioe_url,headers=header)
    if r is not None:
        soup = BeautifulSoup(r.text, 'lxml')

        # get all results
        results_div = soup.find('div',class_='ifad-main-search-results')
        if results_div:
            results = results_div.find_all('a',href=True)

            # search for results with similar name
            expected_result = f'{project_name}'
            matched_result = []
            for result in results:
                title = result.text.strip()
                if jellyfish.jaro_winkler_similarity(title, expected_result) > 0.8:
                    matched_result.append(result)

            best_candidate = None
            best_candidate_similarity = 0
            # download documents
            for result in matched_result:
                link = result['href']
                print(f'Found {link}')
                r = get_request(link,headers=header)
                if r:
                    soup = BeautifulSoup(r.text, 'lxml')
                    # check if country name in project
                    if (not project_country.lower() in soup.text.lower() and not ioe_country_name_mapping[project_country].lower() in soup.text.lower()):
                        continue

                    # check if project date is valid
                    date_element = soup.find('span',class_='annual-card-detail-date blue')
                    if date_element:
                        date = date_element.text
                        date_year = date.strip().split(' ')[-1]
                        if int(date_year) < int(project_duration[1]):
                            continue

                    title = result.text.strip()
                    similarity = jellyfish.jaro_winkler_similarity(title, expected_result)
                    if similarity > best_candidate_similarity:
                        best_candidate = link
                        best_candidate_similarity = similarity

            if best_candidate:
                print(f'Best candidate: {best_candidate}')
                link = best_candidate
                r = get_request(link,headers=header)
                if r:
                    soup = BeautifulSoup(r.text, 'lxml')
                    # get all links in result
                    links = soup.find_all('a',href=True)
                    unique_links = []
                    for link in links:
                        # skip duplicates
                        if link['href'] in unique_links:
                            print(f'Skipping duplicate link: {link["href"]}')
                            continue
                        
                        # get document name
                        document_name = link.text.strip()
                        element = link
                        while not document_name:
                            document_name = element.parent.text.strip()
                            element = element.parent
                        document_url = link['href']
                        document_name = clean_document_name(document_name)

                        
                        if 'https' not in document_url:
                            document_url = f'https://ioe.ifad.org{document_url}'

                        # if target type, download
                        if '.pdf' in document_url and ((document_type := match_target(document_url, evaluation_project_types)) or (document_type := is_evaluation_document(document_url))):
                            if document_type == 'project performance evaluation':
                                document_type = 'ppe'
                            elif document_type == 'project performance assessment':
                                document_type = 'ppa'

                            print(f'Downloading {document_name} from {document_url}')
                            r = get_request(document_url,headers=header)
                            if r:
                                file_name = f'{project_folder}/{document_type}.pdf'
                                # id document if same type exists
                                if os.path.exists(file_name):
                                    id = 2
                                    while os.path.exists(f'{project_folder}/{document_type}_{id}.pdf'):
                                        id += 1
                                    file_name = f'{project_folder}/{document_type}_{id}.pdf'
                                # save document
                                with open(file_name, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=128):
                                        f.write(chunk)

                                unique_links.append(link['href'])

                    # read online button (special case)
                    buttons = soup.find_all('button')
                    read_online_button = None
                    for button in buttons:
                        if 'read' in button.text.lower() and 'online' in button.text.lower() and 'english' in button.text.lower():
                            read_online_button = button
                            break

                    if read_online_button:
                        # extract document link
                        pattern = r'window\.open\(("|\')(.+?)("|\')'
                        match = re.search(pattern, read_online_button['onclick'])
                        if match:
                            document_url = match.group(2)
                            document_name = clean_document_name(document_name)
                            if 'https' not in document_url:
                                document_url = f'https://ioe.ifad.org{document_url}'

                            # if target type, download
                            if '.pdf' in document_url and ((document_type := match_target(document_url, evaluation_project_types)) or (document_type := is_evaluation_document(document_url))):
                                if document_type == 'project performance evaluation':
                                    document_type = 'ppe'
                                elif document_type == 'project performance assessment':
                                    document_type = 'ppa'

                                print(f'Reading online document from {document_url}')
                                r = get_request(document_url,headers=header)
                                if r:
                                    file_name = f'{project_folder}/{document_type}.pdf'
                                    if os.path.exists(file_name):
                                        id = 2
                                        while os.path.exists(f'{project_folder}/{document_type}_{id}.pdf'):
                                            id += 1
                                        file_name = f'{project_folder}/{document_type}_{id}.pdf'
                                    with open(file_name, 'wb') as f:
                                        for chunk in r.iter_content(chunk_size=128):
                                            f.write(chunk)




def get_project_default_documents(project:dict, project_folder: str):
    '''Get documents for single project from project page'''
    project_id = project['id']
    project_name = project['name']
    project_country = project['country']
    project_url = f'{IFAD_projects_url}{project_id}'
    #ioe_url = f'{IFAD_ioe_url}search?q={project_name}'
    if debug_mode:
        print(f'Project {project_id} | {project_url} | {project_country}')

    # project default documents
    get_default_documents(project_id, project_folder)









def get_projects_documents(projects: list, data_folder: str):
    '''Get documents for each project'''
    if debug_mode:
        print('Getting documents for each project...')
    print(f'Getting documents for {len(projects)} projects...')
    for project in projects:
        project_id = project['id']
        project_name = project['name']
        project_country = project['country']

        # create country folder if it doesn't exist
        country_folder = f'{data_folder}/{project_country}'
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)

        # create project folder if it doesn't exist
        project_folder = f'{country_folder}/{project_id}'
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)

        # create project metadata file if it doesn't exist
        metadata_file = f'{project_folder}/metadata.json'
        if not os.path.exists(metadata_file):
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(project, f, indent=4)

        # get documents
        get_project_default_documents(project, project_folder)

        # update bar
        projects_bar.update(1)



def get_projects_evaluation_documents(projects: list, data_folder: str):
    '''Get independent evaluation documents for each project'''

    for project in projects:
        project_id = project['id']
        project_name = project['name']
        project_country = project['country']
        project_date = ifad_date_to_date(project['date'])
        project_duration = [p.strip() for p in project['duration'].split('-')]

        get_ioe_documents(project_name, project_country,project_date,
                          project_duration, f'{data_folder}/{project_country}/{project_id}')

        projects_bar.update(1)



if __name__ == "__main__":
    args = process_args()

    debug_mode = args.debug

    # create data folder if it doesn't exist
    data_folder = f'{file_path}/data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if args.project_documents:

        projects = get_ifad_closed_projects()
        # update bar
        projects_bar.reset(total=len(projects))
        run_threaded_for(get_projects_documents,projects,[data_folder],log=debug_mode,threads=args.threads)

    if args.project_evaluation_documents:

        projects = get_downloaded_projects()
        # update bar
        projects_bar.reset(total=len(projects))
        run_threaded_for(get_projects_evaluation_documents,projects,[data_folder],log=debug_mode,threads=args.threads)