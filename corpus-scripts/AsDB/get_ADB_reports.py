'''Script to obtain ADB reports'''

import datetime
import json
import re
import sys
import pandas
import argparse
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils.utils import get_request,run_threaded_for

run_path = os.getcwd()
document_types = {
    'project/program completion report' : 'pcr', 
    'reports and recommendations of the president' : 'rrp',
    'technical assistance completion report' : 'tcr',  
    'design and monitoring framework' : 'dmf', 
    'pcr validation report' : 'pvr',
    'validations of project completion reports' : 'pvr',
    'project performance evaluation report' : 'pper',
    'ta performance evaluation reports' : 'tper',
    }

evaluation_project_types = ['ppe', 'ppa']
debug_mode = False
projects_bar = tqdm(total=0)


header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36'}


def process_args():
    parser = argparse.ArgumentParser(
        description="Script to obtain ADB reports"
    )
    parser.add_argument("projects_file"         , type=str, nargs="?", default="projects.xlsx"                                  , help="File with projects")
    parser.add_argument("-t","--threads"        , type=int, nargs="?", default=1                                                , help="Number of threads to use. Default=1")
    parser.add_argument("-d","--debug"          , action='store_true', default=False                                            , help="Debug mode. Default=False")
    return parser.parse_args()



def get_projects(projects_file_path):
    '''Get projects from file
    
    Cleans entries:
    - remove duplicate projects
    - remove projects old than 2004'''

    projects = pandas.read_excel(projects_file_path,header=3)

    # remove duplicate projects (column 'Project No.')
    projects = projects.drop_duplicates(subset=['Project Number'])
    # remove non closed projects
    ## column status
    projects = projects[projects['Status'] == 'Closed']
    # remove projects older than 2004 (column 'Approval Date')
    projects['Approval Date'] = projects['Approval Date'].apply(lambda x: datetime.datetime.fromisoformat(str(x)).year if x and datetime.datetime.fromisoformat(str(x)).year > 2004 else 0)
    # clean column 'Project Number'
    ## remove entries with two projects (delimiter '/')
    projects['Project Number'] = projects['Project Number'].apply(lambda x: str(x).split('/')[0])
    projects['Project Number'] = projects['Project Number'].apply(lambda x: str(x).strip())

    # shuffle (for distributing work in threads)
    #projects = projects.sample(frac=1)

    # fix index
    projects = projects.reset_index(drop=True)

    return projects



def download_document(file: str, project_id: str, link: str, document_name: str):
    '''Download document from url page
    
    Tries 2 methods:
    - direct download
    - follow link and get download url'''
    # follow link
    download_link = None
    r = get_request(link,headers=header)
    if r:
        soup = BeautifulSoup(r.text, 'lxml')
        # get links
        links = soup.find_all('a',href=True)
        # find download link
        for link in links:
            if 'download' in link.text.lower():
                download_link = f'https://www.adb.org{link["href"]}'
                break

        if download_link:
            r = get_request(download_link,headers=header)
            if r:
                with open(file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=128):
                        f.write(chunk)


def match_target(document_type: str, document_types: dict):
    '''Check if document type is in list of target types'''
    document_type_treated = document_type.lower().strip()

    for type in document_types.keys():
        #print('Matching',type, document_type_treated)
        if type in document_type_treated:
            return document_types[type]
        
    return None

    

def get_project_documents(project_id: str, project_folder: str):
    '''Get documents for single project
    
    Also get extra metadata for project: end date'''
    projects_url = f'https://www.adb.org/projects/{project_id}/main'
    if debug_mode:
        print(f'Project {project_id} | {projects_url}')


    # open metadata file
    metadata_file = f'{project_folder}/metadata.json'
    metadata = json.load(open(metadata_file, 'r', encoding='utf-8'))

    r = get_request(projects_url,headers=header)
    if r is not None:
        soup = BeautifulSoup(r.text, 'lxml')

        # get all tables with
        tables = soup.find_all('table')
        ## filter tables with column 'document type'
        filtered_tables = []
        for table in tables:
            table_headers = table.find_all('th')
            for table_header in table_headers:
                if table_header.text.strip().lower() == 'document type':
                    filtered_tables.append(table)
                    break

        ## get rows
        rows = []
        for table in filtered_tables:
            rows.extend(table.find_all('tr'))
        ## for each row, check document type, download if target
        for row in rows:
            # get document type (2nd column)
            table_cells = row.find_all('td')
            if table_cells and len(table_cells) > 1:
                document_type = table_cells[1].text
                if t := match_target(document_type,document_types):
                    if debug_mode:
                        print(f'Project {project_id} | Getting document {t}')

                    # get document link
                    link = row.find('a', href=True)
                    # get document name
                    document_name:str = link['href'].split('/')[-1]
                    document_name = document_name[document_name.find(project_id):]
                    link = f'https://www.adb.org{link["href"]}'
                    # download document
                    download_document(f'{project_folder}/{t}.pdf',project_id,link,document_name)


        # check for extra metadata (end date)
        rows = soup.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            for i,col in enumerate(cols):
                if 'Last PDS Update' in col.text.strip():
                    end_date = cols[i+1].text
                    # get year
                    year = re.search(r'\d{4}',end_date)
                    end_date = year.group() if year else None
                    metadata['end_date'] = end_date
                    break
            if 'end_date' in metadata:
                break

        # save metadata
        json.dump(metadata, open(metadata_file, 'w', encoding='utf-8'), indent=4)









def get_projects_documents(indexes: list[int],projects: pandas.DataFrame, data_folder: str):
    '''Get documents for each project'''
    # limit projects to download
    if indexes:
        projects = projects.iloc[indexes[0]:indexes[-1]]
    if debug_mode:
        print('Getting documents for each project...')
    print(f'Getting documents for {len(projects)} projects...')
    for _, row in projects.iterrows():
        project_id = row['Project Number']
        country = row['Country']
        date = row['Approval Date']

        # skip if date is 0
        if date == 0:
            continue

        # create country folder if it doesn't exist
        country_folder = f'{data_folder}/{country}'
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)

        # create project folder if it doesn't exist
        project_folder = f'{country_folder}/{project_id}'
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)
        else:
            continue

        # create metadata file if it doesn't exist
        metadata_file_path = f'{project_folder}/metadata.json'
        if not os.path.exists(metadata_file_path):
            data = {
                'name': str(row['Project Name']).strip(),
                'country': str(row['Country']).strip(),
                'id': project_id,
                'date': row['Approval Date'],
            }
            with open(metadata_file_path, 'w') as f:
                json.dump(data, f, indent=4)

        

        # get documents
        get_project_documents(project_id, project_folder)

        # update bar
        projects_bar.update(1)



if __name__ == "__main__":
    args = process_args()

    debug_mode = args.debug

    # create data folder if it doesn't exist
    data_folder = f'{run_path}/data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    projects_file_path = f'{run_path}/{args.projects_file}'
    if os.path.exists(projects_file_path):

        projects = get_projects(projects_file_path)
        # update bar
        projects_bar.reset(total=len(projects))

        run_threaded_for(get_projects_documents,projects.index.tolist(),[projects,data_folder],log=debug_mode,threads=args.threads)

    else:
        print(f"File {projects_file_path} not found")