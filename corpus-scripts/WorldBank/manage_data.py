import json
import re
import os
import traceback

import jellyfish
from tqdm import tqdm 
# from identify_WB_partnerships import *


file_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f'{file_path}/data'
control_corpus_dir = f'{file_path}/control_corpus'
iso_codes_path = f'{file_path}/../iso_code_map.json'
iso_codes = json.load(open(iso_codes_path,'r',encoding='utf-8'))

######### check all files with given name in directory (recursive)
def check_data_folder():
    directory_path = data_dir
    types = {}
    n_projects_with_files = 0
    n_projects = 0
    partnered = 0
    partnered_with_doc = 0
    valid_partnered = 0
    valid_partnered_with_doc = 0
    for country_folder in os.listdir(directory_path):
        if os.path.isdir(f'{directory_path}/{country_folder}'):
            for project_folder in os.listdir(f'{directory_path}/{country_folder}'):
                if os.path.isdir(f'{directory_path}/{country_folder}/{project_folder}'):
                    n_projects += 1
                    files_list = os.listdir(f'{directory_path}/{country_folder}/{project_folder}')
                    for file in files_list:
                        if file.endswith('.pdf'):
                            n_projects_with_files += 1
                            break
                    for file in files_list:
                        filename = file.split('.')[0].split('_')[-1] if not file.startswith('WB') else file.split('.')[0].split('_')[2]
                        if filename in types:
                            types[filename] += 1
                        else:
                            types[filename] = 1
                    if 'metadata.json' in files_list:
                        metadata_file = json.load(open(f'{directory_path}/{country_folder}/{project_folder}/metadata.json', 'r', encoding='utf-8'))
                        if 'partnerships' in metadata_file and metadata_file['partnerships'] != []:
                            partnered += 1

                            if len(files_list) > 1:
                                partnered_with_doc += 1

                            if metadata_file['end_date'] and int(metadata_file['end_date']) > 2004:
                                valid_partnered += 1
                                if len(files_list) > 1:
                                    valid_partnered_with_doc += 1

    print(f'Projects with files: {n_projects_with_files}/{n_projects}')

    for type in types:
        print(f'{type}: {types[type]}')


    print(f'Partenered projects: {partnered}')
    print(f'Partenered projects with documents: {partnered_with_doc}')

    print(f'Valid partenered projects: {valid_partnered}')
    print(f'Valid partenered projects with documents: {valid_partnered_with_doc}')


#### New name for partnered projects files




IFI_list_file_path = f'{file_path}/../IFI_list.json'
IFI_dict = json.load(open(IFI_list_file_path,'r',encoding='utf-8'))
countries_list_path = f'{file_path}/countryCodes'
countries_dict = {}
with open(countries_list_path,'r',encoding='utf-8') as f: 
    for line in f:
        groups = re.search(r'(.+)\s+?(\w\w)',line)
        if not groups:
            continue
        country_name = groups.group(1)
        country_code = groups.group(2)
        countries_dict[country_code.strip()] = country_name.strip()

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


def map_doc_type(doc_type):
    '''Map document type'''
    doc_type_dict = {
        'Implementation Completion and Results Report': 'ICR',
        'Implementation Completion Report Review' : 'ICRR',
        'Project Appraisal Document' : 'PAD',
        'Project Performance Assessment Report' : 'PPAR',
    }
    if doc_type not in doc_type_dict:
        return None
    return doc_type_dict[doc_type]


def inverse_map_doc_type(doc_type):
    '''Inverse map document type'''
    doc_type_dict = {
        'ICR' : 'Implementation Completion and Results Report',
        'ICRR' : 'Implementation Completion Report Review',
        'PAD' : 'Project Appraisal Document',
        'PPAR' : 'Project Performance Assessment Report',
    }
    if doc_type not in doc_type_dict:
        return None
    return doc_type_dict[doc_type]

def map_country(country_code):
    '''Map country code'''
    return countries_dict[country_code]



def rename_files(dir:str,partnered:bool):
    
    # partnership projects
    ## iterate countries
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
                                ## format: WB_<project_id>_<doc_type>_<is partenered>_<list of partners>_<country code>_<date>_<id>.pdf
                                for file in files:
                                    # if already renamed, skip
                                    if file.startswith('WB_'):
                                        continue

                                    file_type = file.split('.')[0].split('_')[-1]
                                    file_id = file.split('.')[0].split('_')[0]
                                    file_type = map_doc_type(file_type)
                                    if not file_type:
                                        continue
                                    date = metadata['end_date']
                                    country_code = metadata['country_iso3']
                                    is_partnered = 'TRUE' if partnered else 'FALSE'
                                    partners_list = '' if not partners else '_'.join(partners) + '_'

                                    file_name = f'WB_{project}_{file_type}_{is_partnered}_{partners_list}{country_code}_{date}_{file_id}'
                                    # if len(file.split('.')[0].split('_')) > 1:
                                    #     file_name += f'_{file.split(".")[0].split("_")[-1]}'
                                    file_name += '.pdf'
                                    os.rename(f'{dir}/{country}/{project}/{file}', f'{dir}/{country}/{project}/{file_name}')
                                    print(f'Renamed {file} to {file_name}')
                        except Exception as e:
                            print(e)



def inverse_rename_files():
    # iterate countries
    countries = os.listdir(data_dir)

    for country in countries:
        if os.path.isdir(f'{data_dir}/{country}'):
            # iterate projects
            projects = os.listdir(f'{data_dir}/{country}')
            for project in projects:
                if os.path.isdir(f'{data_dir}/{country}/{project}'):
                    files = os.listdir(f'{data_dir}/{country}/{project}')
                    if 'metadata.json' in files:
                        try:
                            metadata = json.load(open(f'{data_dir}/{country}/{project}/metadata.json'))
                            files = [f for f in files if f.endswith('.pdf')]
                            # rename files
                            ## format: <file_id>_<doc_type>
                            for file in files:
                                try:
                                    if not file.startswith('WB_'):
                                        continue
                                    file_type = inverse_map_doc_type(file.split('.')[0].split('_')[2])
                                    file_id = file.split('.')[0].split('_')[-1]
                                    file_name = f'{file_id}_{file_type}.pdf'
                                    os.rename(f'{data_dir}/{country}/{project}/{file}', f'{data_dir}/{country}/{project}/{file_name}')
                                    print(f'Renamed {file} to {file_name}')
                                except Exception as e:
                                    print(traceback.format_exc())
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


def map_wb_countries():
    '''Map country code. Using similarity'''
    countries = os.listdir(data_dir)
    mapping = {}
    for country in countries:
        country_iso_code = None
        if os.path.isdir(f'{data_dir}/{country}'):
            # iterate projects
            projects = os.listdir(f'{data_dir}/{country}')
            for project in projects:
                if os.path.isdir(f'{data_dir}/{country}/{project}'):
                    files = os.listdir(f'{data_dir}/{country}/{project}')
                    if 'metadata.json' in files:
                        try:
                            metadata = json.load(open(f'{data_dir}/{country}/{project}/metadata.json'))
                            country_iso_code = map_iso_code(metadata['countryshortname'])
                        except Exception as e:
                            print(e)
                if country_iso_code:
                    mapping[metadata['countryshortname']] = country_iso_code
                    break

    return mapping

def clean_metadata():
    '''Clean fields from metadata.
    
    Clean country field and end date field'''
    # iterate countries
    countries = os.listdir(data_dir)
    p_bar = tqdm(countries,total=len(countries))

    # load country iso code mapping
    country_iso_codes_path = f'{file_path}/wb_countries_iso3_mapping.json'
    country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))

    for country in countries:
        if os.path.isdir(f'{data_dir}/{country}'):
            # iterate projects
            projects = os.listdir(f'{data_dir}/{country}')
            for project in projects:
                if os.path.isdir(f'{data_dir}/{country}/{project}'):
                    files = os.listdir(f'{data_dir}/{country}/{project}')
                    if 'metadata.json' in files:
                        try:
                            metadata = json.load(open(f'{data_dir}/{country}/{project}/metadata.json'))
                            project_country = country_iso_codes[metadata['countryshortname']]
                            metadata['country_iso3'] = project_country
                            metadata['end_date'] = re.search(r'\d{4}',metadata['closingdate']).group(0) if 'closingdate' in metadata else None
                            metadata['total_budget'] = int(metadata['totalamt'].replace(',','')) if 'totalamt' in metadata else 0
                            sector = metadata['sector'] if 'sector' in metadata else None
                            # turn list of sector dict into single item or list of items (not dicts)
                            if type(sector) == list and len(sector) > 0 and type(sector[0]) == dict:
                                new_sector = []
                                for s in sector:
                                    new_sector.append(s['Name'])
                                if len(new_sector) == 1:
                                    new_sector = new_sector[0]
                                sector = new_sector
                            metadata['sector'] = sector

                            json.dump(metadata,open(f'{data_dir}/{country}/{project}/metadata.json','w',encoding='utf-8'),indent=4)
                        except Exception as e:
                            print(e)

        p_bar.update(1)






if __name__ == '__main__':
    check_data_folder()

    # rename_files(data_dir,True)
    # rename_files(control_corpus_dir,False)


    # inverse_rename_files()

    #print(dict(sorted(map_wb_countries().items())))
    # json.dump(dict(sorted(map_wb_countries().items())),open(f'{file_path}/wb_countries_iso3_mapping.json','w',encoding='utf-8'),indent=4)

    #clean_metadata()

    # folder_path = f'{file_path}/data/NP/P160748'
    # paterns = identify_partnerships(folder_path)

    # print(paterns)
