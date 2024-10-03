import json
import re
import traceback
from bs4 import BeautifulSoup
import os

import jellyfish
from tqdm import tqdm 
from utils.utils import get_request

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
    partenered = 0
    dated = 0
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
                        filename = file.split('.')[0].split('_')[0] if not file.startswith('AsDB') else file.split('.')[0].split('_')[2]
                        if filename in types:
                            types[filename] += 1
                        else:
                            types[filename] = 1
                    if 'metadata.json' in files_list:
                        metadata_file = json.load(open(f'{directory_path}/{country_folder}/{project_folder}/metadata.json', 'r', encoding='utf-8'))
                        if 'partnerships' in metadata_file and metadata_file['partnerships']:
                            partenered += 1
                        if 'date' in metadata_file and metadata_file['date'] != 0:
                            dated += 1


    print(f'Projects with files: {n_projects_with_files}/{n_projects}')

    for type in types:
        print(f'{type}: {types[type]}')

    print(f'Partenered projects: {partenered}')
    print(f'Dated projects: {dated}')



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
                                ## format: AsDB_<project_id>_<doc_type>_<is partnered>_<list of partners>_<country>_<date>.pdf
                                for file in files:
                                    # check if file is renamed
                                    if file.startswith('AsDB_'):
                                        continue

                                    file_type = file.split('.')[0].upper()
                                    date = metadata['end_date']
                                    country_code = metadata['country_iso3']
                                    is_partnered = 'TRUE' if partnered else 'FALSE'
                                    partners_list = '' if not partners else '_'.join(partners) + '_'

                                    file_name = f'AsDB_{project}_{file_type}_{is_partnered}_{partners_list}{country_code}_{date if date else ""}.pdf'
                                    os.rename(f'{dir}/{country}/{project}/{file}', f'{dir}/{country}/{project}/{file_name}')
                                    print(f'Renamed {file} to {file_name}')

                        except Exception as e:
                            print(e)


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


def get_extra_metadata_partnerships(partnerships_only:bool = True, logs:bool=False):
    '''Get extra metadata for partnered projects.
    
    Extra metadata includes:
    - end date
    - sectors
    - total budget'''


    header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36'}

    dir = data_dir
    countries = os.listdir(dir)

    total_projects = 0
    for country in countries:
        for project in os.listdir(f'{dir}/{country}'):
            total_projects += 1

    p_bar = tqdm(total=total_projects)

    for country in countries:
        if os.path.isdir(f'{dir}/{country}'):
            # iterate projects
            projects = os.listdir(f'{dir}/{country}')
            for project in projects:
                p_bar.update(1)

                if os.path.isdir(f'{dir}/{country}/{project}'):
                    files = os.listdir(f'{dir}/{country}/{project}')
                    if 'metadata.json' in files:
                        try:
                            metadata = json.load(open(f'{dir}/{country}/{project}/metadata.json'))
                            if not partnerships_only or 'partnerships' in metadata and metadata['partnerships']:
                                # if project has metadata, skip
                                if 'end_date' in metadata and metadata['end_date'] and 'sector' in metadata and metadata['sector'] \
                                    and 'total_budget' in metadata and metadata['total_budget']:
                                    continue

                                project_url = f'https://www.adb.org/projects/{project}/main'
                                r = get_request(project_url,headers=header)
                                if r:
                                    soup = BeautifulSoup(r.text, 'lxml')
                                    rows = soup.find_all('tr')
                                    for row in rows:
                                        cols = row.find_all('td')
                                        for i,col in enumerate(cols):
                                            # get end date
                                            if 'Last PDS Update' in col.text.strip():
                                                end_date = cols[i+1].text
                                                if logs:
                                                    print(f'Col : "Last PDS Update" found. Value: {end_date}')
                                                # get year
                                                year = re.search(r'\d{4}',end_date)
                                                end_date = year.group() if year else None
                                                metadata['end_date'] = end_date
                                            # get sector
                                            elif 'Sector / Subsector' in col.text.strip():
                                                if logs:
                                                    print(f'Col : "Sector / Subsector" found. Value: {cols[i+1].text}')
                                                sectors = re.search(r'([^\/]+)(\/\s*(.+))?',cols[i+1].text,re.MULTILINE)
                                                if sectors:
                                                    metadata['sector'] = sectors.group(1).strip()
                                                    metadata['subsector'] = sectors.group(3).strip() if sectors.group(3) else None
                                            # get total budget
                                            elif 'Source of Funding / Amount' in col.text.strip():
                                                if logs:
                                                    print(f'Col : "Source of Funding / Amount" found.')
                                                funding_row = row
                                                funders_table = funding_row.find_next('table')
                                                if funders_table:
                                                    rows = funders_table.find_all('tr')
                                                    total_budget = 0
                                                    for row in rows:
                                                        columns = row.find_all('td')
                                                        if columns and len(columns) > 1:
                                                            budget_value = convert_total_project_cost(columns[1].text)
                                                            total_budget += budget_value

                                                    metadata['total_budget'] = total_budget

                                        if 'end_date' in metadata and metadata['end_date'] and 'sector' in metadata and metadata['sector'] \
                                            and 'total_budget' in metadata and metadata['total_budget']:
                                            break
                                # save metadata
                                with open(f'{dir}/{country}/{project}/metadata.json','w',encoding='utf-8') as f:
                                    json.dump(metadata,f,indent=4)
                        except Exception as e:
                            print(traceback.format_exc())




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
                            if 'partnerships' in metadata and metadata['partnerships']:
                                partners = map_partners(metadata['partnerships'])
                                files = [f for f in files if f.endswith('.pdf')]
                                # rename files
                                ## format: <doc_type>_<id>?.pdf
                                for file in files:
                                    try:
                                        file_type = file.split('.')[0].split('_')[2].lower()
                                        file_id = file.split('.')[0].split('_')[-1] if len(file.split('.')[0].split('_')) > 5 + len(partners) else None
                                        file_name = f'{file_type}'
                                        if file_id:
                                            file_name += f'_{file_id}'
                                        file_name += '.pdf'
                                        print(f'Renamed {file} to {file_name}')
                                        os.rename(f'{data_dir}/{country}/{project}/{file}', f'{data_dir}/{country}/{project}/{file_name}')
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


def map_asdb_countries():
    '''Map country code. Using similarity'''
    countries = os.listdir(data_dir)
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
    country_iso_codes_path = f'{file_path}/asdb_countries_iso3_mapping.json'
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


if __name__ == '__main__':
    check_data_folder()
    #inverse_rename_files()
    #rename_files(data_dir,True)
    # rename_files(control_corpus_dir,False)
    # get_extra_metadata_partnerships(partnerships_only=False, logs=False)  

    # print(dict(sorted(map_asdb_countries().items())))
    # json.dump(dict(sorted(map_asdb_countries().items())),open(f'{file_path}/asdb_countries_iso3_mapping.json','w',encoding='utf-8'),indent=4)


    # clean_metadata()    