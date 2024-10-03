'''Script to create control corpus for UNDP'''


import datetime
import os
import json
import re
import shutil
import sys
import tqdm
import fitz
from utils.utils import get_request
from langdetect import detect


file_path = os.path.dirname(os.path.abspath(__file__))
corpus_folder = f'{file_path}/../corpus'
undp_partnered_projects_folder = f'{corpus_folder}/projects/partner/UNDP'
data_folder = f'{file_path}/data'
undp_api = 'https://api.open.undp.org/api'
undp_api_projects_list = f'{undp_api}/project_list'
undp_api_project = f'{undp_api}/projects'

ifi_donors_ids_file_path = f'{file_path}/UNDP_IFI_funding_id.json'
if os.path.exists(ifi_donors_ids_file_path):
    ifi_donors_ids = json.load(open(ifi_donors_ids_file_path, 'r', encoding='utf-8'))
else:
    sys.exit(f"File {ifi_donors_ids_file_path} not found")


evaluation_documents_types = ['pe','mtr']

def match_donors_ids(donors:list,ifi_donors_ids:dict):
    '''Match donors ids with ifi_donors_ids'''
    matched_donors = set()
    for donor in donors:
        for ifi in ifi_donors_ids.keys():
            if donor in ifi_donors_ids[ifi]:
                matched_donors.add(ifi)
                break
    if 'UNDP' in matched_donors:
        matched_donors.remove('UNDP')
    matched_donors = list(matched_donors)
    return matched_donors


def is_partnered(project_info):
    '''Check if project is partnered.'''
    project_partners = []
    for output in project_info['outputs']:
        donors = output['donor_id']
        project_partners += match_donors_ids(donors, ifi_donors_ids)
    project_partners = list(set(project_partners))

    return len(project_partners) > 0


def is_evaluation_document(document_name:str):
    '''Check if document is evaluation document.'''
    if 'evaluation' in document_name.lower() or ' te ' in document_name.lower() or 'terminal evaluation' in document_name.lower() and ( 'mid' not in document_name.lower() and 'term' not in document_name.lower()):
        return True
    elif ('mtr' in document_name.lower() or ('mid' in document_name.lower() and 'term' in document_name.lower())): 
        return True
    return False


def is_english_document(pdf):
    '''Check if document is english.'''
    if type(pdf) == str:
        doc = fitz.open(pdf)
    else:
        doc = fitz.open('pdf', pdf)
    all_text = ""
    english_sentences = 0
    total_sentences = 0

    contains_text = False
    for page in doc:
        if page.get_text().strip():
            contains_text = True
            text = page.get_text()
            sentences = re.split(r'(?<=[.!?]) +', text)
            for sentence in sentences:
                if sentence.strip():
                    total_sentences += 1
                    try :
                        if detect(sentence) == 'en':
                            english_sentences += 1
                    except:
                        pass
                    all_text += sentence + "\n"

    english_ratio = english_sentences / total_sentences if total_sentences > 0 else 0
    return contains_text and english_ratio > 0.5



def get_non_partnered_projects(countries:set=None):
    '''Get non partnered projects list from API.'''
    parameters = {
        'year' : '2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024',
        'budget_source' : None
    }

    country_iso_codes_path = f'{file_path}/undp_countries_iso3_mapping.json'
    country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))
    projects = {}
    r = get_request(undp_api_projects_list,params=parameters)
    if r:
        n_projects = r.json()['data']['count']
        print('Number of projects:', n_projects)
        p_bar = tqdm.tqdm(total=n_projects)
        while r:
            
            r = r.json()
            # add projects to list
            for project in r['data']['data']:
                p_bar.update(1)
                id = project['project_id']
                country = project['country'].strip() if project['country'] else 'Other'
                if countries and country not in countries:
                    continue

                project_url = f'{undp_api_project}/{id}.json'
                project_info = get_request(project_url)
                if project_info:
                    project_info = project_info.json()
                    if not project_info['end'] or datetime.date.fromisoformat(project_info['end']) > datetime.date.today() or is_partnered(project_info):
                        continue
                    project_data = project_info
                    project_data['id'] = id
                    project_data['country'] = country
                    project_data['country_iso3'] = country_iso_codes[country] if country in country_iso_codes else 'REG'
                    project_data['sdg'] = project['sdg']
                    project_data['sector'] = project['sector']
                    
                    projects[id] = project_data
                
            if r['data']['links']['next']:
                r = get_request(r['data']['links']['next'])
            else:
                r = None

    return projects





def get_project_documents(project_id:str,documents: list, project_folder: str):
    '''Get documents for single project'''

    document_names = documents[0]
    documents_url = documents[1]
    documents_types = documents[2]

    # get evaluation documents
    ## no "mid" or "term" in name
    # get final mid term review
    evalution_doc_id = 0
    for i,document_name in enumerate(document_names):
        doc_url = None
        # evaluation doc
        if documents_types[i] in ['pdf','doc'] and ('evaluation' in document_name.lower() or ' te ' in document_name.lower() or 'terminal evaluation' in document_name.lower() and ( 'mid' not in document_name.lower() and 'term' not in document_name.lower())):
            print(f'Project {project_id} | Found evaluation document: {document_name} | type: {documents_types[i]}')
            doc_url = documents_url[i]
            file_name = f'{project_folder}/PE_{evalution_doc_id}.{documents_types[i]}'
        # final mid term review
        elif documents_types[i] in ['pdf','doc'] and ('mid' in document_name.lower() and 'term' in document_name.lower() or 'mtr' in document_name.lower()) and 'final' in document_name.lower():
            print(f'Project {project_id} | Found mid term review document: {document_name} | type: {documents_types[i]}')
            doc_url = documents_url[i]
            file_name = f'{project_folder}/MTR.{documents_types[i]}'

        if doc_url:
            r = get_request(doc_url)
            if r:
                evalution_doc_id += 1
                with open(file_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=128):
                        f.write(chunk)

    # if no evaluation document found, get a mtr
    if not evalution_doc_id:
        for i,document_name in enumerate(document_names):
            if documents_types[i] in ['pdf','doc'] and ('mtr' in document_name.lower() or ('mid' in document_name.lower() and 'term' in document_name.lower())):
                print(f'Project {project_id} | Found MTR document: {document_name} | type: {documents_types[i]}')
                doc_url = documents_url[i]
                file_name = f'{project_folder}/MTR.{documents_types[i]}'
                r = get_request(doc_url)
                if r:
                    with open(file_name, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=128):
                            f.write(chunk)



def analyze_partnered_projects()->dict:
    '''Analyse projects that are partnered (from corpus folder). Create dict of with analysis.
    
    Dic structure:
    {
        'project_id_1': {
            'documents': [document1, document2, ...],
            'end_year' : <year>,
            'sdg' : <sdg>,
            'country' : <country>,
        },
        'project_id_2': {
            'documents': [document1, document2, ...],
            'end_year' : <year>,
            'sdg' : <sdg>,
            'country' : <country>,
        },
        ...
    }
    '''


    analyzes = {}

    if os.path.isdir(undp_partnered_projects_folder):

        for project in os.listdir(undp_partnered_projects_folder):
            if os.path.isdir(f'{undp_partnered_projects_folder}/{project}'):
                # read metadata
                metadata = json.load(open(f'{undp_partnered_projects_folder}/{project}/metadata.json'))
                country = metadata['country']
                end_year = metadata['end_date']
                sdg = metadata['sdg']


                if project not in analyzes:
                    analyzes[project] = {
                        'documents': [], ## TODO: change to list if needed
                        'end_date' : end_year,
                        'sdg' : sdg,
                        'country' : country
                    }

                # read documents
                for document in os.listdir(f'{undp_partnered_projects_folder}/{project}'):
                    if document.endswith('.pdf'):
                        analyzes[project]['documents'].append(document)


    return analyzes



def get_control_project(counterpart_project:dict, possible_projects:dict)->str:
    '''Get control project for UNDP. It needs to be a project that is not partnered, has at least one evaluation document, and should have closed in the same year as the counterpart project.'''
    control_project = None

    print(f'Possible projects: {len(possible_projects)}')


    counterpart_project_sdg = [sdg['id'] for sdg in counterpart_project['sdg']]

    priority_end_year = True if counterpart_project['end_date'] else False
    priority_sector = True if counterpart_project_sdg else False
    priority_country = True
    i = 0
    project_keys_list = list(possible_projects.keys())
    while i < len(possible_projects) and not control_project:
        valid = True
        project_id = project_keys_list[i]
        project = possible_projects[project_id]
        project_docs = project['document_name']

        document_names = project_docs[0]
        document_urls = project_docs[1]
        documents_types = project_docs[2]

        # check priorities

        ## check if project has same country
        if priority_country:
            if project['country'] != counterpart_project['country']:
                valid = False


        ## check if at least on sdg in common
        if priority_sector and valid:
            project_sdg = [sdg['id'] for sdg in project['sdg']]

            if not any(sdg in counterpart_project_sdg for sdg in project_sdg):
                valid = False
            

        ## check if project has same end year
        if priority_end_year and valid:
            project_duration = project['end'].split('-')[0]
            if not project_duration or project_duration != counterpart_project['end_date']:
                valid = False


        ## check if project has evaluation document
        has_evaluation_doc = False
        if valid and 'has_evaluation_doc' not in project:
            for j,doc in enumerate(document_names):
                # check if doc is type pdf
                if documents_types[j] not in ['pdf','doc']:
                    continue

                if not is_evaluation_document(doc):
                    continue

                # check if doc is in english
                r = get_request(document_urls[j])
                if r:
                    if is_english_document(r._content):
                        has_evaluation_doc = True
                        break

            if not has_evaluation_doc:
                valid = False
                # remove project if no evaluation doc
                del possible_projects[project_id]
                project_keys_list = list(possible_projects.keys())
                i -= 1
            else:
                project['has_evaluation_doc'] = has_evaluation_doc


        

        if valid:
            control_project = project_id

        i += 1
        # if none found, remove priority and check again
        if i >= len(possible_projects) and not control_project:
            i = 0
            if priority_sector:
                priority_sector = False
            elif priority_end_year:
                priority_end_year = False
            elif priority_country:
                priority_country = False
                # reset priorities
                priority_end_year = True
                priority_sector = True
            else:
                break


    return control_project

def create_control_corpus(analyzes: dict,available_projects:dict=None):
    '''Create control corpus for UNDP'''

    control_corpus_path = f'{file_path}/control_corpus'

    # create control corpus folder
    if not os.path.exists(control_corpus_path):
        os.makedirs(control_corpus_path)

    p_bar = tqdm.tqdm(total=len(analyzes))
    matched_projects = 0
  
    for project_id in analyzes:
        p_bar.update(1)

        project = analyzes[project_id]
        # find control project
        control_project_id = get_control_project(project,available_projects)

        if control_project_id:
            control_project = available_projects[control_project_id]
            control_project_country = control_project['country']
            project_country_iso3 = control_project['country_iso3']
            project_sectors = [s['name'] for s in control_project['sector']]
            project_sector = project_sectors[0] if len(project_sectors) == 1 else project_sectors
            control_project_end_date = re.search(r'\d{4}',control_project['end']).group(0) if 'end' in control_project else None

            # create country folder
            if not os.path.exists(f'{control_corpus_path}/{control_project_country}'):
                os.makedirs(f'{control_corpus_path}/{control_project_country}')

            # add control project to corpus
            control_project_path = f'{control_corpus_path}/{control_project_country}/{control_project_id}'
            if not os.path.exists(control_project_path):
                os.makedirs(control_project_path)

            # create project metadata
            control_project_metadata = {
                'name': control_project['project_title'],
                'country': control_project_country,
                'id': control_project_id,
                'approval_date': control_project['start'],
                'close_date': control_project['end'],
                'sdg' : control_project['sdg'],
                'sector' : project_sector,
                'end_date' : control_project_end_date,
                'country_iso3' : project_country_iso3,
                'total_budget': control_project['budget'],
                'expenditure': control_project['expenditure'],
            }
            with open(f'{control_project_path}/metadata.json', 'w') as outfile:
                json.dump(control_project_metadata, outfile)

            # download documents to corpus
            get_project_documents(control_project_id,available_projects[control_project_id]['document_name'],control_project_path)
            
            # remove non english documents
            for doc in os.listdir(control_project_path):
                if doc.split('.')[-1] in ['pdf'] and not is_english_document(f'{control_project_path}/{doc}'):
                    os.remove(f'{control_project_path}/{doc}')

            del available_projects[control_project_id]
                        
            matched_projects += 1

    print(f'Number of matched projects: {matched_projects}')

            

            




if __name__ == "__main__":
    
    analyzes = analyze_partnered_projects()
    print(analyzes)
    print('Number of partnered projects:', len(analyzes))
    project_countries = set()
    for project in analyzes.values():
        project_countries.add(project['country'])

    if not os.path.exists(f'{file_path}/available_projects.json'):
        available_projects = get_non_partnered_projects(project_countries)
        with open(f'{file_path}/available_projects.json', 'w') as outfile:
            json.dump(available_projects, outfile)
    else:
        with open(f'{file_path}/available_projects.json') as json_file:
            available_projects = json.load(json_file)

    print('Number of available projects:', len(available_projects))


    create_control_corpus(analyzes,available_projects=available_projects)