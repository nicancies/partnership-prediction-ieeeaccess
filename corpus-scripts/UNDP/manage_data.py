import json
import re
import traceback
import os

import jellyfish
from tqdm import tqdm 

file_path = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{file_path}/data' 
control_corpus_dir = f'{file_path}/control_corpus'
iso_codes_path = f'{file_path}/../iso_code_map.json'
iso_codes = json.load(open(iso_codes_path,'r',encoding='utf-8'))
######### check all files with given name in directory (recursive)a

def check_data_folder():
    output_list = []
    directory_path = control_corpus_dir
    types = {}
    n_projects_with_files = 0
    n_projects = 0
    for country_folder in os.listdir(directory_path):
        if os.path.isdir(f'{directory_path}/{country_folder}'):
            for project_folder in os.listdir(f'{directory_path}/{country_folder}'):
                if os.path.isdir(f'{directory_path}/{country_folder}/{project_folder}'):
                    n_projects += 1
                    files_list = os.listdir(f'{directory_path}/{country_folder}/{project_folder}')
                    metadata = json.load(open(f'{directory_path}/{country_folder}/{project_folder}/metadata.json', 'r', encoding='utf-8'))
                    # output_list.append(f'{project_folder};;{country_folder};;{metadata["name"]};;{metadata["partnerships"]};;{metadata["approval_date"]};;{metadata["close_date"]}')
                    for file in files_list:
                        if file.endswith('.pdf'):
                            n_projects_with_files += 1
                            # output_list.pop()
                            break
                    for file in files_list:
                        filename = file.split('.')[0].split('_')[0] if not file.startswith('UNDP') else file.split('.')[0].split('_')[2]
                        if filename in types:
                            types[filename] += 1
                        else:
                            types[filename] = 1

    print(f'Projects with files: {n_projects_with_files}/{n_projects}')

    for type in types:
        print(f'{type}: {types[type]}')




IFI_list_file_path = f'{file_path}/../IFI_list.json'
IFI_dict = json.load(open(IFI_list_file_path,'r',encoding='utf-8'))



def map_partners(partners:set):
    '''Map partners using IFI dict'''
    mapped_partners = set()
    ifi_keys_lower = [k.lower() for k in IFI_dict.keys()]
    for partner in partners:
        # check if partner is in IFI dict
        if partner.lower() in ifi_keys_lower:
            for ifi in IFI_dict.keys():
                if partner.lower() in ifi.lower():
                    mapped_partners.add(ifi)
        else:
            # check if partner is alternative name for an IFI
            for ifi in IFI_dict.keys():
                for name in IFI_dict[ifi]:
                    if partner.lower() in name.lower():
                        mapped_partners.add(ifi)
    return mapped_partners

def map_document_type(document_name):
    if 'project_evaluation' in document_name:
        return 'PE'
    elif 'mtr' in document_name:
        return 'MTR'
    return document_name


def rename_files(dir:str,partnered:bool):
    # iterate countries
    countries = os.listdir(dir)
    i = 0
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
                                partners = map_partners(metadata['partnerships']) if 'partnerships' in metadata else None
                                files = [f for f in files if f.endswith('.pdf')]
                                # rename files
                                ## format: AsDB_<project_id>_<doc_type>_<is partnered>_<list of partners>_<country>_<date>.pdf
                                for file in files:
                                    # check if file is renamed
                                    if file.startswith('UNDP_'):
                                        continue


                                    file_type = file.split('.')[0].split('_')[0].upper()
                                    date = metadata['end_date']
                                    country_code = metadata['country_iso3']
                                    is_partnered = 'TRUE' if partnered else 'FALSE'
                                    partners_list = '' if not partners else '_'.join(partners) + '_'
                                    
                                    file_name = f'UNDP_{project}_{file_type}_{is_partnered}_{partners_list}{country_code}_{date if date else ""}'
                                    # id if exists
                                    if len(file.split('.')[0].split('_'))>1:
                                        id = file.split('.')[0].split('_')[-1]
                                        file_name += f'_{id}'
                                    file_name += '.pdf'
                                    os.rename(f'{dir}/{country}/{project}/{file}', f'{dir}/{country}/{project}/{file_name}')
                                    print(f'Renamed {file} to {file_name}')
                                    i+=1

                        except Exception as e:
                            print(traceback.format_exc())

    print(f'Renamed {i} files')


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


def map_undp_countries():
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
    # load undp iso code mapping
    country_iso_codes_path = f'{file_path}/undp_countries_iso3_mapping.json'
    country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))

    # data folder
    # iterate countries
    dir = data_dir
    countries = os.listdir(dir)
    p_bar = tqdm(countries,total=len(countries))



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
                            metadata['end_date'] = re.search(r'\d{4}',metadata['close_date']).group(0) if 'close_date' in metadata else None

                            json.dump(metadata,open(f'{dir}/{country}/{project}/metadata.json','w',encoding='utf-8'),indent=4)
                        except Exception as e:
                            print(e)

        p_bar.update(1)

    # control corpus
    if os.path.exists(control_corpus_dir):
        # iterate countries
        countries = os.listdir(control_corpus_dir)
        p_bar = tqdm(countries,total=len(countries))



        for country in countries:
            if os.path.isdir(f'{control_corpus_dir}/{country}'):
                # iterate projects
                projects = os.listdir(f'{control_corpus_dir}/{country}')
                for project in projects:
                    if os.path.isdir(f'{control_corpus_dir}/{country}/{project}'):
                        files = os.listdir(f'{control_corpus_dir}/{country}/{project}')
                        if 'metadata.json' in files:
                            try:
                                metadata = json.load(open(f'{control_corpus_dir}/{country}/{project}/metadata.json'))
                                project_country = country_iso_codes[country]
                                metadata['country_iso3'] = project_country
                                metadata['end_date'] = re.search(r'\d{4}',metadata['close_date']).group(0) if 'close_date' in metadata else None

                                json.dump(metadata,open(f'{control_corpus_dir}/{country}/{project}/metadata.json','w',encoding='utf-8'),indent=4)
                            except Exception as e:
                                print(e)

            p_bar.update(1)



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
                                        file_id = file_id if file_id and len(file_id) != 4 else None
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



if __name__ == '__main__':
    check_data_folder()

    # rename_files(data_dir,True)
    # rename_files(control_corpus_dir,False)
    #inverse_rename_files()

    # turn_doc_to_pdf(f'{file_path}/data')


    # print(dict(sorted(map_undp_countries().items())))
    # json.dump(dict(sorted(map_undp_countries().items())),open(f'{file_path}/undp_countries_iso3_mapping.json','w',encoding='utf-8'),indent=4)

    # clean_metadata()


    # for country in os.listdir(data_dir):
    #     for project in os.listdir(f'{data_dir}/{country}'):
    #         for file in os.listdir(f'{data_dir}/{country}/{project}'):
    #             id = None
    #             if len(file.split('_')) > 1:
    #                 id = file.split('_')[-1].split('.')[0]

    #             if id and len(id) == 4:
    #                 os.rename(f'{data_dir}/{country}/{project}/{file}',f'{data_dir}/{country}/{project}/{file.split("_")[0]}.pdf')