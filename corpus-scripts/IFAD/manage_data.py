import json
import os
import re
import jellyfish
import fitz
from langdetect import detect
from tqdm import tqdm 

file_path = os.path.dirname(os.path.abspath(__file__))
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
    partnered_with_doc = 0
    projects_with_evaluation = 0
    dated = 0
    for country_folder in os.listdir(directory_path):
        if os.path.isdir(f'{directory_path}/{country_folder}'):
            for project_folder in os.listdir(f'{directory_path}/{country_folder}'):
                if os.path.isdir(f'{directory_path}/{country_folder}/{project_folder}'):
                    n_projects += 1
                    files_list = os.listdir(f'{directory_path}/{country_folder}/{project_folder}')
                    has_evaluation = False
                    for file in files_list:
                        if file.endswith('.pdf'):
                            n_projects_with_files += 1
                            break
                    for file in files_list:
                        filename = file.split('.')[0].split('_')[0] if not file.startswith('IFAD') else file.split('.')[0].split('_')[2]
                        if filename in types:
                            types[filename] += 1
                        else:
                            types[filename] = 1

                        if 'ppa' in filename.lower() or 'ppe' in filename.lower():
                            has_evaluation = True

                    if 'metadata.json' in files_list:
                        metadata_file = json.load(open(f'{directory_path}/{country_folder}/{project_folder}/metadata.json', 'r', encoding='utf-8'))
                        if 'partnerships' in metadata_file and metadata_file['partnerships']:
                            partenered += 1
                            if len(files_list) > 1:
                                partnered_with_doc += 1
                        if 'date' in metadata_file and metadata_file['date'] != 0:
                            dated += 1

                    if has_evaluation:
                        projects_with_evaluation += 1
                    else:
                        print(project_folder,metadata_file['name'],metadata_file['country'])


    print(f'Projects with files: {n_projects_with_files}/{n_projects}')

    for type in types:
        print(f'{type}: {types[type]}')

    print(f'Partenered projects: {partenered}')
    print(f'Partenered projects with doc: {partnered_with_doc}')
    print(f'Dated projects: {dated}')
    print(f'Projects with evaluation: {projects_with_evaluation}')


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


def map_doc_type(doc_type):
    '''Map document type'''
    doc_type_dict = {
        'project completion report': 'PCR',
        'pcr' : 'PCR',
        'president\'s report': 'PR',
        'pr' : 'PR',
        'ppe' : 'PPE',
        'project performance evaluation' : 'PPE',
        'ppa' : 'PPA',
        'project performance assessment' : 'PPA',
        'project design report' : 'PDR',
        'pdr' : 'PDR',
        'project completion report digest' : 'PCRD',
        'pcrd' : 'PCRD',
        'supervision report' : 'SR',
        'sr' : 'SR',
        '(mid-term) review report' : 'MTR',
        'mtr' : 'MTR',
        'final design report' : 'FDR',
        'fdr' : 'FDR',
        'mid term review' : 'MTR',
        'final project design report' : 'FPDR',
        'fpdr' : 'FPDR'
    }
    return doc_type_dict[doc_type]

def inverse_map_doc_type(doc_type):
    '''Inverse map document type'''
    doc_type_dict = {
        'PCR' : 'project completion report',
        'PR' : 'president\'s report',
        'PPE' : 'ppe',
        'PPA' : 'ppa',
        'PDR' : 'project design report',
        'PCRD' : 'project completion report digest',
        'SR' : 'supervision report',
        'MTR' : '(mid-term) review report',
        'FDR' : 'final design report',
        'FPDR' : 'final project design report'

    }
    return doc_type_dict[doc_type]



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
                                ## format: IFAD_<project_id>_<doc_type>_<is partnered>_<list of partners>_<country>_<date>_<id>?.pdf
                                for file in files:
                                    # check if file is renamed
                                    if file.startswith('IFAD_'):
                                        continue

                                    file_type = file.split('.')[0].split('_')[0]
                                    file_type = map_doc_type(file_type)
                                    date = metadata['end_date']
                                    country_code = metadata['country_iso3']
                                    is_partnered = 'TRUE' if partnered else 'FALSE'
                                    partners_list = '' if not partners else '_'.join(partners) + '_'

                                    file_name = f'IFAD_{project}_{file_type}_{is_partnered}_{partners_list}{country_code}_{date if date else ""}'
                                    # id if exists
                                    if len(file.split('.')[0].split('_')) > 1:
                                        file_name += f'_{file.split(".")[0].split("_")[-1]}'
                                    file_name += '.pdf'
                                    os.rename(f'{dir}/{country}/{project}/{file}', f'{dir}/{country}/{project}/{file_name}')
                                    print(f'Renamed {file} to {file_name}')

                        except Exception as e:
                            print(e)



def inverse_rename_files():
    dir =data_dir

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
                            if 'partnerships' in metadata and metadata['partnerships']:
                                partners = map_partners(metadata['partnerships'])
                                files = [f for f in files if f.endswith('.pdf')]
                                # rename files
                                ## format: <doc_type>_<id>?.pdf
                                for file in files:
                                    try:
                                        file_type = inverse_map_doc_type(file.split('.')[0].split('_')[2])
                                        file_id = file.split('.')[0].split('_')[-1] if len(file.split('.')[0].split('_')) > 6 + len(partners) else None
                                        file_name = f'{file_type}'
                                        if file_id and len(file_id) != 4:
                                            file_name += f'_{file_id}'
                                        file_name += '.pdf'
                                        os.rename(f'{dir}/{country}/{project}/{file}', f'{dir}/{country}/{project}/{file_name}')
                                        print(f'Renamed {file} to {file_name}')
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

def map_ifad_countries():
    '''Map country code. Using similarity'''
    countries = os.listdir(data_dir)
    mapping = {}
    for country in countries:
        country_iso_code = map_iso_code(country)
        mapping[country] = country_iso_code

    return mapping


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



def clean_metadata():
    '''Clean fields from metadata.
    
    Clean country field and end date field'''
    dir = data_dir
    # iterate countries
    countries = os.listdir(dir)
    total_projects = 0
    for country in countries:
        if os.path.isdir(f'{dir}/{country}'):
            total_projects += len(os.listdir(f'{dir}/{country}'))
    p_bar = tqdm(countries,total=total_projects)

    # load ifad iso code mapping
    country_iso_codes_path = f'{file_path}/ifad_countries_iso3_mapping.json'
    country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))
    match_cost = 0
    for country in countries:
        country_iso_code = map_iso_code(country)
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
                            metadata['end_date'] = metadata['duration'].split('-')[-1].strip()
                            metadata['total_budget'] = convert_total_project_cost(metadata['total project cost'])
                            if metadata['total_budget'] != 0:
                                match_cost += 1
                            json.dump(metadata,open(f'{dir}/{country}/{project}/metadata.json','w',encoding='utf-8'),indent=4)
                        except Exception as e:
                            print(e)

                p_bar.update(1)
    print(f'Number of projects with cost: {match_cost}')


def remove_ioe_files():
    '''Remove ioe files'''
    coutries = os.listdir(data_dir)
    for country in coutries:
        projects = os.listdir(f'{data_dir}/{country}')
        for project in projects:
            files = os.listdir(f'{data_dir}/{country}/{project}')
            for file in files:
                if 'ppe' in file.lower() or 'ppa' in file.lower() or 'project performance evaluation' in file.lower() or 'project performance assessment' in file.lower():
                    os.remove(f'{data_dir}/{country}/{project}/{file}')


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


def clean_files():
    '''Remove non english or non openable files'''
    total_files = 0
    for country in os.listdir(data_dir):
        for project in os.listdir(f'{data_dir}/{country}'):
            for file in os.listdir(f'{data_dir}/{country}/{project}'):
                if file.endswith('.pdf'):
                    total_files += 1

    p_bar = tqdm(total=total_files)
    coutries = os.listdir(data_dir)
    for country in coutries:
        projects = os.listdir(f'{data_dir}/{country}')
        for project in projects:
            files = os.listdir(f'{data_dir}/{country}/{project}')
            for file in files:
                if file.endswith('.pdf'):
                    p_bar.update(1)
                    file_path = f'{data_dir}/{country}/{project}/{file}'
                    try:
                        is_english = is_english_document(file_path)
                        if not is_english:
                            os.remove(file_path)
                            print(f'Removed {file_path}')
                    except Exception as e:
                        os.remove(file_path)
                        print(f'Removed {file_path}')


if __name__ == '__main__':
    check_data_folder()
    #rename_files(data_dir,True)
    # rename_files(control_corpus_dir,False)
    #inverse_rename_files()

    # remove_ioe_files()
    # clean_files()

    # print(dict(sorted(map_ifad_countries().items())))
    # json.dump(dict(sorted(map_ifad_countries().items())),open(f'{file_path}/ifad_countries_iso3_mapping.json','w',encoding='utf-8'),indent=4)

    # clean_metadata()

    # for country in os.listdir(data_dir):
    #     if os.path.isdir(f'{data_dir}/{country}'):
    #         for project in os.listdir(f'{data_dir}/{country}'):
    #             if os.path.isdir(f'{data_dir}/{country}/{project}'):
    #                 print(f'{data_dir}/{country}/{project}')
    #                 docs = os.listdir(f'{data_dir}/{country}/{project}')
    #                 for doc in docs:
    #                     try:
    #                         id = doc.split('.')[0].split('_')[-1]
    #                         doc_type = doc.split('.')[0].split('_')[0]
    #                         if len(id) == 4:
    #                             os.rename(f'{data_dir}/{country}/{project}/{doc}',f'{data_dir}/{country}/{project}/{doc_type}.pdf')
    #                     except Exception as e:
    #                         print(e)



    # Projects with files: 798/992
    # metadata: 992
    # president's report: 745
    # project completion report digest: 198
    # supervision report: 223
    # project design report: 28
    # (mid-term) review report: 48
    # ppa: 16
    # project completion report: 106
    # ppe: 20
    # final design report: 8
    # final project design report: 5
    # project performance evaluation: 9
    # mid term review: 2
    # Partenered projects: 266
    # Partenered projects with doc: 189
    # Dated projects: 992





# Projects with files: 803/992
# pr: 758
# mtr: 203
# sr: 1538
# pcr: 125
# pdr: 83
# fpdr: 39
# pcrd: 223
# ppe: 79
# ppa: 43
# metadata: 703
# Partenered projects: 180
# Partenered projects with doc: 121
# Dated projects: 703

# Projects with files: 501/992
# metadata: 992
# pr: 330
# pcrd: 221
# ppe: 62
# mtr: 113
# sr: 908
# fpdr: 19
# pcr: 84
# pdr: 53
# ppa: 41
# Partenered projects: 270
# Partenered projects with doc: 112
# Dated projects: 992


# Projects with files: 501/992
# metadata: 992
# pr: 330
# pcrd: 221
# ppe: 62
# mtr: 113
# sr: 908
# fpdr: 19
# pcr: 84
# pdr: 53
# ppa: 41
# Partenered projects: 270
# Partenered projects with doc: 112
# Dated projects: 992

# Projects with files: 501/992
# metadata: 992
# pr: 330
# pcrd: 221
# ppe: 64
# mtr: 113
# sr: 908
# fpdr: 19
# pcr: 84
# pdr: 53
# ppa: 48
# Partenered projects: 270
# Partenered projects with doc: 112
# Dated projects: 992