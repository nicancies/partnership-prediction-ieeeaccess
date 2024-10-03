'''Script to obtain UNDP reports'''

import datetime
import json
import re
import sys
import argparse
import os
from tqdm import tqdm
from utils.utils import get_request,run_threaded_for

run_path = os.getcwd()

debug_mode = False
projects_bar = tqdm(total=0)

undp_api = 'https://api.open.undp.org/api'
undp_api_projects_list = f'{undp_api}/project_list'
undp_api_project = f'{undp_api}/projects'



def process_args():
    parser = argparse.ArgumentParser(
        description="Script to obtain UNDP reports"
    )
    parser.add_argument("-idif","--ifi_donors_ids_file" , type=str, nargs="?", default="UNDP_IFI_funding_id.json"                       , help="File with IFI donors ids")
    parser.add_argument("-t","--threads"                , type=int, nargs="?", default=1                                                , help="Number of threads to use. Default=1")
    parser.add_argument("-d","--debug"                  , action='store_true', default=False                                            , help="Debug mode. Default=False")
    return parser.parse_args()




def get_projects(ifi_donors_ids : dict):
    '''Get partenered projects list from API.
    
    Partners are stored in ifi_donors_ids'''
    parameters = {
        'year' : '2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024',
        'budget_source' : None
    }

    country_iso_codes_path = f'{run_path}/undp_countries_iso3_mapping.json'
    country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))

    projects = []
    for ifi,ids in ifi_donors_ids.items():
        if debug_mode:
            print(f'Listing projects for {ifi}')
        # ignore UNDP (all projects should be in UNDP)
        if ifi.lower() != 'undp' and ids:
            # search projects for each id
            for ifi_id in ids:
                parameters['budget_source'] = ifi_id
                r = get_request(undp_api_projects_list,params=parameters)
                while r:
                    r = r.json()
                    # add projects to list
                    for project in r['data']['data']:
                        id = project['project_id']
                        country = project['country'].strip() if project['country'] else 'Other'
                        country_iso3 = country_iso_codes[country] if country in country_iso_codes else 'REG'
                        projects.append({
                            'id':id,
                            'country':country,
                            'country_iso3':country_iso3,
                            'sdg':project['sdg'],
                            'sector':project['sector'],
                            })
                        
                    if r['data']['links']['next']:
                        r = get_request(r['data']['links']['next'])
                    else:
                        r = None

    return projects


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


    

def get_project_documents(project_id:str,documents: list, project_folder: str):
    '''Get documents for single project'''
    if debug_mode:
        print(f'Project {project_id} | Getting documents...')

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
            print(f'Project {project_id} | Found evaluation document: {document_name}')
            doc_url = documents_url[i]
            file_name = f'{project_folder}/PE_{evalution_doc_id}.{documents_types[i]}'
        # final mid term review
        elif documents_types[i] in ['pdf','doc'] and (('mid' in document_name.lower() and 'term' in document_name.lower() or 'mtr' in document_name.lower()) and 'final' in document_name.lower()):
            print(f'Project {project_id} | Found mid term review document: {document_name}')
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
                print(f'Project {project_id} | Found MTR document: {document_name}')
                doc_url = documents_url[i]
                file_name = f'{project_folder}/MTR.{documents_types[i]}'
                r = get_request(doc_url)
                if r:
                    with open(file_name, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=128):
                            f.write(chunk)


    










def get_projects_documents(projects: list, data_folder: str,ifi_donors_ids : dict):
    '''Get documents for each project'''

    if debug_mode:
        print('Getting documents for each project...')
        print(f'Getting documents for {len(projects)} projects...')
    for project in projects:
        # update bar
        projects_bar.update(1)

        project_id = project['id']
        project_country = project['country']
        project_country_iso3 = project['country_iso3']
        project_sdg = project['sdg']
        project_sectors = [s['name'] for s in project['sector']]
        project_sector = project_sectors[0] if len(project_sectors) == 1 else project_sectors

        # get project
        project_url = f'{undp_api_project}/{project_id}.json'
        r = get_request(project_url)
        if r is None:
            if debug_mode:
                print(f'Project {project_id} not found')
            continue

        r = r.json()

        # check if project is closed
        if not r['end'] or datetime.date.fromisoformat(r['end']) > datetime.date.today():
            if debug_mode:
                print(f'Project {project_id} is not closed')
            continue


        # create country folder if it doesn't exist
        country_folder = f'{data_folder}/{project_country}'
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)

        # create project folder if it doesn't exist
        project_folder = f'{country_folder}/{project_id}'
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)
        else:
            pass

        # get partners
        project_partners = []
        for output in r['outputs']:
            donors = output['donor_id']
            project_partners += match_donors_ids(donors, ifi_donors_ids)
        project_partners = list(set(project_partners))


        # create metadata file if it doesn't exist
        metadata_file_path = f'{project_folder}/metadata.json'
        # if not os.path.exists(metadata_file_path):
        data = {
            'name': r['project_title'].strip(),
            'country': project_country,
            'id': project_id,
            'approval_date': r['start'],
            'close_date': r['end'],
            'partnerships': project_partners,
            'sdg': project_sdg,
            'total_budget': r['budget'],
            'expenditure': r['expenditure'],
            'end_date' : re.search(r'\d{4}',r['end']).group(0) if 'end' in r else None,
            'sector' : project_sector,
            'country_iso3' : project_country_iso3
        }
        with open(metadata_file_path, 'w') as f:
            json.dump(data, f, indent=4)

        # get documents
        #get_project_documents(project_id,r['document_name'], project_folder)

        



if __name__ == "__main__":
    args = process_args()

    debug_mode = args.debug

    ifi_donors_ids_file_path = f'{run_path}/{args.ifi_donors_ids_file}'
    if os.path.exists(ifi_donors_ids_file_path):
        ifi_donors_ids = json.load(open(ifi_donors_ids_file_path, 'r', encoding='utf-8'))
    else:
        sys.exit(f"File {ifi_donors_ids_file_path} not found")     

    # create data folder if it doesn't exist
    data_folder = f'{run_path}/data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)


    projects = get_projects(ifi_donors_ids)
    # update bar
    projects_bar.reset(total=len(projects))

    run_threaded_for(get_projects_documents,projects,[data_folder,ifi_donors_ids],log=debug_mode,threads=args.threads)
