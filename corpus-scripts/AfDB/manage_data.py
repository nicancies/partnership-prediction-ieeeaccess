import json
import re
import shutil
import traceback
from bs4 import BeautifulSoup
import os

import jellyfish
from tqdm import tqdm 
from utils.utils import get_request

file_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f'{file_path}/data'
folders_dir = f'{data_dir}/files'
projects_dir = f'{data_dir}/projects'
control_corpus_dir = f'{file_path}/control_corpus'
iso_codes_path = f'{file_path}/../iso_code_map.json'
iso_codes = json.load(open(iso_codes_path,'r',encoding='utf-8'))

######### check all files with given name in directory (recursive)

def check_data_folder():
    directory_path = folders_dir
    types = {}
    n_projects_with_files = 0
    n_projects = 0
    partenered = 0
    idd_projects = 0
    unique_ids = set()
    projects_url = 0
    dated = 0
    rated = 0
    valid_partnered = 0

    not_idd_example = []
    not_rated_example = []
    not_date_example = []
    for project_folder in os.listdir(directory_path):
        if os.path.isdir(f'{directory_path}/{project_folder}'):
            n_projects += 1
            files_list = os.listdir(f'{directory_path}/{project_folder}')
            for file in files_list:
                if file.endswith('.pdf'):
                    n_projects_with_files += 1
                    break
            for file in files_list:
                filename = file.split('.')[0].split('_')[0] if not file.startswith('AfDB') else file.split('.')[0].split('_')[2]
                if filename in types:
                    types[filename] += 1
                else:
                    types[filename] = 1
            if 'metadata.json' in files_list:
                metadata_file = json.load(open(f'{directory_path}/{project_folder}/metadata.json', 'r', encoding='utf-8'))
                if 'partnerships' in metadata_file and metadata_file['partnerships']:
                    partenered += 1
                    if 'end_date' in metadata_file and metadata_file['end_date'] and int(metadata_file['end_date']) > 2004 and 'id' in metadata_file:
                        valid_partnered += 1
                if 'end_date' in metadata_file and metadata_file['end_date'] != 0:
                    dated += 1
                else:
                    if 'id' in metadata_file:
                        not_date_example.append(f'{directory_path}/{project_folder}')
                if 'id' in metadata_file:
                    idd_projects += 1
                    unique_ids.add(metadata_file['id'])
                else:
                    not_idd_example.append(f'{directory_path}/{project_folder}')
                if 'project_url' in metadata_file:
                    projects_url += 1
                if 'project_rating' in metadata_file and metadata_file['project_rating']:
                    rated += 1
                else:
                    if 'id' in metadata_file:
                        not_rated_example.append(f'{directory_path}/{project_folder}')


    print(f'Projects with files: {n_projects_with_files}/{n_projects}')

    for type in types:
        print(f'{type}: {types[type]}')

    
    print(f'IDD projects: {idd_projects}')
    print(f'Not idd example: {not_idd_example[11]}')
    print(f'Unique ids: {len(unique_ids)}')
    print(f'Projects with url: {projects_url}')
    print(f'Partenered projects: {partenered}')
    print(f'Valid partnered projects: {valid_partnered}')
    print(f'Dated projects: {dated}')
    print(f'Not date example: {not_date_example[0]}')
    print(f'Rated projects: {rated}')
    print(f'Not rated example: {not_rated_example[0]}')



#### New name for partnered projects files



IFI_list_file_path = f'{file_path}/../IFI_list.json'
IFI_dict = json.load(open(IFI_list_file_path,'r',encoding='utf-8'))


def map_partners(partners:set):
    '''Map partners using IFI dict'''
    mapped_partners = set()
    ifi_keys_lower = [k.lower() for k in IFI_dict.keys()]
    for partner in partners:
        # check if partner is in IFI dict
        if partner in ifi_keys_lower:
            for ifi in IFI_dict.keys():
                if partner in ifi.lower():
                    mapped_partners.add(ifi)
        else:
            # check if partner is alternative name for an IFI
            for ifi in IFI_dict.keys():
                if partner in IFI_dict[ifi].lower():
                    mapped_partners.add(ifi)
    return mapped_partners
    


def rename_files(dir:str,partnered:bool):

    # iterate countries
    countries = os.listdir(dir)

    for country in countries:
        if os.path.isdir(f'{dir}/{country}'):
            # iterate projects
            projects = os.listdir(f'{dir}/{country}')
            for project in projects:
                if os.path.isdir(f'{dir}/{country}/{project}'):
                    files = os.listdir(f'{dir}/{country}/{project}')
                    if 'metadata.json' in files:
                        try:
                            metadata = json.load(open(f'{dir}/{country}/{project}/metadata.json'))
                            if 'partnerships' in metadata and metadata['partnerships'] or not partnered:
                                partners = map_partners(metadata['partnerships'])
                                files = [f for f in files if f.endswith('.pdf')]
                                # rename files
                                ## format: AfDB_<project_id>_<doc_type>_<is partnered>_<list of partners>_<country>_<date>.pdf
                                for file in files:
                                    # check if file is renamed
                                    if file.startswith('AfDB_'):
                                        continue

                                    file_type = file.split('.')[0].upper()
                                    date = metadata['end_date']
                                    country_code = metadata['country_iso3']
                                    is_partnered = 'TRUE' if partnered else 'FALSE'
                                    partners_list = '' if not partners else '_'.join(partners) + '_'

                                    file_name = f'AfDB_{project}_{file_type}_{is_partnered}_{partners_list}{country_code}_{date if date else ""}.pdf'
                                    os.rename(f'{dir}/{country}/{project}/{file}', f'{dir}/{country}/{project}/{file_name}')
                                    print(f'Renamed {file} to {file_name}')

                        except Exception as e:
                            print('Project',project,'metadata.json',e)


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




def inverse_rename_files():
    # iterate countries
    dir = folders_dir

    folders = os.listdir(dir)
    for folder in folders:
        if os.path.isdir(f'{dir}/{folder}'):
            files = os.listdir(f'{dir}/{folder}')
            if 'metadata.json' in files:
                try:
                    files = [f for f in files if f.endswith('.pdf')]
                    # rename files
                    ## format: <doc_type>_<id>?.pdf
                    for file in files:
                        try:
                            if not file.startswith('AfDB_'):
                                continue
                            file_type = 'pcr'
                            file_id = file.split('.')[0].split('_')[-1] if len(file.split('_')) > 1 else None
                            file_name = f'{file_type}'
                            if file_id and len(file_id) != 4:
                                file_name += f'_{file_id}'
                            file_name += '.pdf'
                            print(f'Renamed {file} to {file_name}')
                            os.rename(f'{dir}/{folder}/{file}', f'{dir}/{folder}/{file_name}')
                        except Exception as e:
                            print(e)
                except Exception as e:
                    print(e)
                            


def map_iso_code(country):
    '''Map country code. Using similarity'''
    most_similar_country = None
    max_similarity = 0
    for iso_code in iso_codes:
        similarity = jellyfish.jaro_winkler_similarity(country,iso_codes[iso_code])
        if similarity > max_similarity:
            most_similar_country = iso_code
            max_similarity = similarity
    return most_similar_country


def map_afdb_countries():
    '''Map country code. Using similarity'''
    countries = os.listdir(f'{data_dir}/projects')
    mapping = {}
    for country in countries:
        country_iso_code = map_iso_code(country)
        mapping[country] = country_iso_code

    return mapping

def clean_metadata():
    '''Clean fields from metadata.
    
    Clean country field and end date field'''
    dir = data_dir
    # iterate countries
    countries = os.listdir(dir)
    p_bar = tqdm(countries,total=len(countries))

    # load asdb iso code mapping
    country_iso_codes_path = f'{file_path}/afdb_countries_iso3_mapping.json'
    country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))

    for country in countries:
        if os.path.isdir(f'{dir}/{country}'):
            # iterate projects
            projects = os.listdir(f'{dir}/{country}')
            for project in projects:
                if os.path.isdir(f'{dir}/{country}/{project}'):
                    files = os.listdir(f'{dir}/{country}/{project}')
                    if 'metadata.json' in files:
                        try:
                            metadata = json.load(open(f'{dir}/{country}/{project}/metadata.json'))
                            project_country = country_iso_codes[country]
                            metadata['country_iso3'] = project_country
                            metadata['end_date'] = metadata['date'] if 'date' in metadata and metadata['date'] else None

                            json.dump(metadata,open(f'{dir}/{country}/{project}/metadata.json','w',encoding='utf-8'),indent=4)
                        except Exception as e:
                            print(e)

        p_bar.update(1)


def create_data_project_dir():
    '''Create projects directory in data folder'''

    dir = data_dir

    # create projects folder
    if not os.path.exists(f'{dir}/projects'):
        os.mkdir(f'{dir}/projects')
    else:
        shutil.rmtree(f'{dir}/projects')
        os.mkdir(f'{dir}/projects')

    # iterate folders
    folders = os.listdir(folders_dir)
    for folder in folders:
        if not os.path.exists(f'{folders_dir}/{folder}/metadata.json'): continue
        
        metadata = json.load(open(f'{folders_dir}/{folder}/metadata.json'))
        if 'id' in metadata and 'country' in metadata:
            country = metadata['country']
            id = metadata['id']
            # create country folder
            if not os.path.exists(f'{dir}/projects/{country}'):
                os.mkdir(f'{dir}/projects/{country}')
            # create project folder
            if not os.path.exists(f'{dir}/projects/{country}/{id}'):
                os.mkdir(f'{dir}/projects/{country}/{id}')
            
            # copy files to project folder
            files = os.listdir(f'{folders_dir}/{folder}')
            project_folder_files = os.listdir(f'{dir}/projects/{country}/{id}')
            for file_name in files:
                base_file_name = file_name.split('.')[0]
                if file_name in project_folder_files:
                    if file_name.endswith('.pdf'):
                        # id file if exists
                        file_id = len([f for f in project_folder_files if base_file_name in f])
                        shutil.copy(f'{folders_dir}/{folder}/{file_name}',f'{dir}/projects/{country}/{id}/{base_file_name}_{file_id}.pdf')
                    else:
                        # metadata file already exists
                        ## merge the two files
                        existing_metadata = json.load(open(f'{dir}/projects/{country}/{id}/metadata.json'))
                        metadata = json.load(open(f'{folders_dir}/{folder}/metadata.json'))
                        ## fill missing fields
                        for key, value in metadata.items():
                            if key not in existing_metadata:
                                existing_metadata[key] = value
                            elif not existing_metadata[key]:
                                existing_metadata[key] = value
                        json.dump(existing_metadata,open(f'{dir}/projects/{country}/{id}/metadata.json','w',encoding='utf-8'),indent=4)
                else:
                    shutil.copy(f'{folders_dir}/{folder}/{file_name}',f'{dir}/projects/{country}/{id}/{file_name}')
                


if __name__ == '__main__':
    # check_data_folder()
    # inverse_rename_files()
    # rename_files(projects_dir,True)
    # rename_files(projects_dir,True)
    # get_extra_metadata_partnerships(partnerships_only=False, logs=False) 

    create_data_project_dir() 

    # print(dict(sorted(map_afdb_countries().items())))
    # json.dump(dict(sorted(map_afdb_countries().items())),open(f'{file_path}/afdb_countries_iso3_mapping.json','w',encoding='utf-8'),indent=4)

    # clean_metadata()    

    # country_iso3_mapping = json.load(open(f'{file_path}/afdb_countries_iso3_mapping.json','r',encoding='utf-8'))

    # for folder in os.listdir(folders_dir):
    #     path = f'{folders_dir}/{folder}'
    #     if 'metadata.json' in os.listdir(path):
    #         metadata = json.load(open(f'{path}/metadata.json'))
    #         if 'country_iso3' not in metadata and 'country' in metadata:
    #             metadata['country_iso3'] = country_iso3_mapping[metadata['country']]
    #             json.dump(metadata,open(f'{path}/metadata.json','w',encoding='utf-8'),indent=4)
