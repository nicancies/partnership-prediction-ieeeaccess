

import argparse
import json
import os
import random
import re
import shutil
import traceback
import fitz
from langdetect import detect
import pandas as pd
import tqdm


file_path = os.path.dirname(os.path.abspath(__file__))


def process_args():
    parser = argparse.ArgumentParser(
        description="Script to obtain WB reports"
    )
    parser.add_argument('-pc','--partner_corpus'                , action='store_true', default=False                                            , help="Create partner corpus. Default=False")
    parser.add_argument('-cc','--control_corpus'                , action='store_true', default=False                                            , help="Create control corpus. Default=False")
    parser.add_argument('--clean_corpus'                        , action='store_true', default=False                                            , help="Clean corpus. Default=False")
    parser.add_argument('-ac','--analyse_corpus'                , action='store_true', default=False                                            , help="Analyse corpus. Default=False")
    parser.add_argument('-ace','--analyse_corpus_excel'         , action='store_true', default=False                                            , help="Analyse corpus and save it in excel. Default=False")
    parser.add_argument('-co','--clean_organizations'           , type=str, nargs="*"                                                           , help="Remove organizations from corpus.")
    parser.add_argument("-d","--debug"                          , action='store_true', default=False                                            , help="Debug mode. Default=False")
    return parser.parse_args()



def prepare_organizations_data_folders(logs:bool=False):
    '''Rename files in data folders'''
    from AsDB.manage_data import rename_files as rename_files_asdb
    from IFAD.manage_data import rename_files as rename_files_ifad
    from WorldBank.manage_data import rename_files as rename_files_worldbank
    from UNDP.manage_data import rename_files as rename_files_undp
    from AfDB.manage_data import rename_files as rename_files_afdb
    if logs:
        print('Preparing data folders...')

    if logs:
        print('Renaming AsDB files...')
    asdb_data_dir = f'{file_path}/AsDB/data'
    rename_files_asdb(asdb_data_dir,partnered=True)

    if logs:
        print('Renaming IFAD files...')
    ifad_data_dir = f'{file_path}/IFAD/data'
    rename_files_ifad(ifad_data_dir,partnered=True)

    if logs:
        print('Renaming WorldBank files...')
    worldbank_data_dir = f'{file_path}/WorldBank/data'
    rename_files_worldbank(worldbank_data_dir,partnered=True)

    if logs:
        print('Renaming UNDP files...')
    undp_data_dir = f'{file_path}/UNDP/data'
    rename_files_undp(undp_data_dir,partnered=True)

    if logs:
        print('Renaming AfDB files...')
    afdb_data_dir = f'{file_path}/AfDB/data/projects'
    rename_files_afdb(afdb_data_dir,partnered=True)



def prepare_organizations_control_corpus(logs:bool=False):
    '''Prepare organizations control corpus'''
    from AsDB.control_corpus import analyze_partnered_projects as analyze_partnered_projects_asdb
    from AsDB.control_corpus import create_control_corpus as create_control_corpus_asdb
    from AsDB.manage_data import rename_files as rename_files_asdb
    from IFAD.control_corpus import analyze_partnered_projects as analyze_partnered_projects_ifad
    from IFAD.control_corpus import create_control_corpus as create_control_corpus_ifad
    from IFAD.manage_data import rename_files as rename_files_ifad
    from WorldBank.control_corpus import analyze_partnered_projects as analyze_partnered_projects_worldbank
    from WorldBank.control_corpus import create_control_corpus as create_control_corpus_worldbank
    from WorldBank.manage_data import rename_files as rename_files_worldbank
    from UNDP.control_corpus import analyze_partnered_projects as analyze_partnered_projects_undp
    from UNDP.control_corpus import create_control_corpus as create_control_corpus_undp
    from UNDP.manage_data import rename_files as rename_files_undp
    from AfDB.control_corpus import analyze_partnered_projects as analyze_partnered_projects_afdb
    from AfDB.control_corpus import create_control_corpus as create_control_corpus_afdb
    from AfDB.manage_data import rename_files as rename_files_afdb


    if logs:
        print('Preparing AsDB control corpus...')

    asdb_control_corpus_dir = f'{file_path}/AsDB/control_corpus'
    if not os.path.exists(asdb_control_corpus_dir):
        analysis = analyze_partnered_projects_asdb()
        create_control_corpus_asdb(analysis)
    rename_files_asdb(asdb_control_corpus_dir,partnered=False)

    if logs:
        print('Preparing IFAD control corpus...')

    ifad_control_corpus_dir = f'{file_path}/IFAD/control_corpus'
    if not os.path.exists(ifad_control_corpus_dir):
        analysis = analyze_partnered_projects_ifad()
        create_control_corpus_ifad(analysis)
    rename_files_ifad(ifad_control_corpus_dir,partnered=False)

    if logs:
        print('Preparing WorldBank control corpus...')

    worldbank_control_corpus_dir = f'{file_path}/WorldBank/control_corpus'
    if not os.path.exists(worldbank_control_corpus_dir):
        analysis = analyze_partnered_projects_worldbank()
        create_control_corpus_worldbank(analysis)
    rename_files_worldbank(worldbank_control_corpus_dir,partnered=False)

    if logs:
        print('Preparing UNDP control corpus...')

    undp_control_corpus_dir = f'{file_path}/UNDP/control_corpus'
    if not os.path.exists(undp_control_corpus_dir):
        analysis = analyze_partnered_projects_undp()
        create_control_corpus_undp(analysis)
    rename_files_undp(undp_control_corpus_dir,partnered=False)

    if logs:
        print('Preparing AfDB control corpus...')

    afdb_control_corpus_dir = f'{file_path}/AfDB/control_corpus'
    if not os.path.exists(afdb_control_corpus_dir):
        analysis = analyze_partnered_projects_afdb()
        create_control_corpus_afdb(analysis)
    rename_files_afdb(afdb_control_corpus_dir,partnered=False)



def is_evaluation_document(document_name:str,organization:str):

    doc_type = document_name.split('_')[2].lower()
    is_evaluation = False

    if organization == 'AsDB':
        if doc_type in ['pper','pvr','pcr','ppar']:
            is_evaluation = True
    elif organization == 'IFAD':
        if doc_type in ['ppe','ppa','pcr','pcrd','mtr','sr']:
            is_evaluation = True
    elif organization == 'WorldBank':
        if doc_type in ['icr','icrr','ppar']:
            is_evaluation = True
    elif organization == 'UNDP':
        if doc_type in ['mtr','pe']:
            is_evaluation = True
    elif organization == 'AfDB':
        if doc_type in ['pcr']:
            is_evaluation = True

    return is_evaluation




def create_corpus():
    '''Iterates over each organizations data folder and, for each project, if partnerships in metadata, moves files to corpus folder.

    On corpus folder, a new folder for the project is created where metadata is included. PDF files are inserted in documents folder.'''

    organizations_list = ['AsDB','IFAD','WB','UNDP','AfDB']

    corpus_folder = f'{file_path}/corpus'
    evaluation_documents_folder = f'{corpus_folder}/documents/evaluation'
    design_documents_folder = f'{corpus_folder}/documents/design'
    projects_folder = f'{corpus_folder}/projects/partner'

    # iterate over organizations
    for organization in organizations_list:
        print(f'Creating corpus for {organization}')
        # check if organization folder exists
        organization_name = organization if organization != 'WB' else 'WorldBank'
        organization_folder = f'{file_path}/{organization_name}/data'

        # create folder for organization in corpus/projects folder
        if not os.path.exists(f'{projects_folder}/{organization}'):
            os.mkdir(f'{projects_folder}/{organization}')

        if os.path.isdir(organization_folder):
            if organization == 'AfDB':
                ## data structure is slightly different
                organization_folder = f'{organization_folder}/projects'

            # iterate over countries
            for country_folder in os.listdir(organization_folder):
                if os.path.isdir(f'{organization_folder}/{country_folder}'):
                    country_folder_path = f'{organization_folder}/{country_folder}'
                    if os.path.isdir(country_folder_path):
                        # iterate over projects
                        for project_folder in os.listdir(country_folder_path):
                            project_folder_path = f'{country_folder_path}/{project_folder}'
                            if os.path.isdir(project_folder_path):
                                project_files = os.listdir(project_folder_path)
                                # has more than 1 file (they should have a metadata file)
                                if len(project_files) > 1:
                                    try:
                                        # check if project has metadata
                                        if 'metadata.json' in project_files:
                                            metadata_file_path = f'{project_folder_path}/metadata.json'
                                            metadata_file = json.load(open(metadata_file_path, 'r', encoding='utf-8'))
                                            if 'partnerships' in metadata_file and metadata_file['partnerships']:

                                                # check if project has id 
                                                if not metadata_file['id']:
                                                    continue

                                                # check if project is before 2004
                                                if not metadata_file['end_date'] or int(metadata_file['end_date']) <= 2004:
                                                    continue

                                                # create project folder in corpus folder
                                                project_folder_corpus = f'{projects_folder}/{organization}/{project_folder}'
                                                if not os.path.exists(project_folder_corpus):
                                                    os.mkdir(project_folder_corpus)
                                                else:
                                                    continue
                                                # copy pdf files and metadata to corpus folder
                                                for file in project_files:
                                                    if file.endswith('.pdf'):
                                                        # check if file can be opened (not corrupt)
                                                        try:
                                                            with open(f'{project_folder_path}/{file}', 'rb') as f:
                                                                f.read1(1)
                                                        except:
                                                            continue
                                                        destination = evaluation_documents_folder if is_evaluation_document(file,organization) else design_documents_folder
                                                        # copy file to corpus folder using shutil
                                                        shutil.copy(f'{project_folder_path}/{file}', destination)
                                                # copy metadata to corpus folder
                                                shutil.copy(metadata_file_path, project_folder_corpus)

                                    except Exception as e:
                                        print(f'Error creating corpus: {traceback.format_exc()}')



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

def clean_corpus(logs: bool = False):
    '''Removes from corpus invalid files, ex.: non-english files, non-pdf files, etc.
    
    After removing invalid files, remove projects with no documents'''

    if logs:
        print('Cleaning corpus...')

    corpus_folder = f'{file_path}/corpus'
    evaluation_documents_folder = f'{corpus_folder}/documents/evaluation'
    design_documents_folder = f'{corpus_folder}/documents/design'
    projects_folder = f'{corpus_folder}/projects/partner'

    evaluation_documents = os.listdir(evaluation_documents_folder)
    design_documents = os.listdir(design_documents_folder)

    p_bar = tqdm.tqdm(total=len(evaluation_documents + design_documents))

    # evaluation documents
    for document in evaluation_documents:
        p_bar.update(1)
        # if not document.startswith('AfDB'):
        #     continue

        # check if document is a pdf
        if not document.endswith('.pdf'):
            os.remove(f'{evaluation_documents_folder}/{document}')
            if logs:
                print(f'Removed {document}')
            continue
            
        # check document language
        try:
            is_english = is_english_document(f'{evaluation_documents_folder}/{document}')
            if not is_english:
                os.remove(f'{evaluation_documents_folder}/{document}')
                if logs:
                    print(f'Removed {document}')
        except Exception as e:
            os.remove(f'{evaluation_documents_folder}/{document}')
            if logs:
                print(f'Removed {document}')


    # design documents
    for document in design_documents:
        p_bar.update(1)
        # if not document.startswith('AfDB'):
        #     continue
        # check if document is a pdf
        if not document.endswith('.pdf'):
            os.remove(f'{design_documents_folder}/{document}')
            if logs:
                print(f'Removed {document}')
            continue
            
        # check document language
        try:
            is_english = is_english_document(f'{design_documents_folder}/{document}')
            if not is_english:
                os.remove(f'{design_documents_folder}/{document}')
                if logs:
                    print(f'Removed {document}')
        except Exception as e:
            os.remove(f'{design_documents_folder}/{document}')
            if logs:
                print(f'Removed {document}')

    if logs:
        print('Finished cleaning corpus')
        print('Removing projects with no documents...')

    # remove projects with no documents
    
    ## gather projects
    projects = {}
    for organization in os.listdir(projects_folder):
        organization_name = organization if organization != 'WorldBank' else 'WB'
        projects[organization_name] = {}
        organization_projects = os.listdir(f'{projects_folder}/{organization}')
        for project in organization_projects:
            projects[organization_name][project] = {'path': f'{projects_folder}/{organization}/{project}'}

    ## analyse for each project if documents exists
    documents = os.listdir(design_documents_folder) + os.listdir(evaluation_documents_folder)
    for organization in projects:
        if organization == 'WorldBank':
            organization = 'WB'
        organization_documents = [doc for doc in documents if doc.startswith(organization)]

        for project in projects[organization]:
            found_document = False
            for document in organization_documents:
                if project in document:
                    found_document = True
                    break

            if not found_document:
                shutil.rmtree(projects[organization][project]['path'])
                if logs:
                    print(f'Removed {project}')




def create_control_corpus():
    
    # create folder
    control_corpus_path = f'{file_path}/corpus'
    projects_folder = f'{control_corpus_path}/projects/control'
    documents_folder = f'{control_corpus_path}/documents'
    evaluation_documents_folder = f'{documents_folder}/evaluation'
    design_documents_folder = f'{documents_folder}/design'


    organizations_list = ['AsDB','AfDB','IFAD','WB','UNDP']

    # iterate over organizations
    for organization in organizations_list:
        print(f'Creating control corpus for {organization}')
        # check if organization folder exists
        organization_control_corpus_folder = f'{file_path}/{organization}/control_corpus'

        if os.path.isdir(organization_control_corpus_folder):
            
            # create folder for organization in control corpus
            if not os.path.exists(f'{projects_folder}/{organization}'):
                os.makedirs(f'{projects_folder}/{organization}')

            for country in os.listdir(organization_control_corpus_folder):
                country_folder = f'{organization_control_corpus_folder}/{country}'
                # copy organization control corpus to control corpus
                projects = os.listdir(country_folder)
                for project in projects:
                    # create project folder in projects directory
                    ## save metadata file in folder
                    if not os.path.exists(f'{projects_folder}/{organization}/{project}'):
                        os.makedirs(f'{projects_folder}/{organization}/{project}')

                    shutil.copyfile(f'{country_folder}/{project}/metadata.json', f'{projects_folder}/{organization}/{project}/metadata.json')

                    # copy documents to documents folder
                    ## don't copy metadata
                    documents = os.listdir(f'{country_folder}/{project}')
                    for doc in documents:
                        if 'metadata' in doc:
                            continue
                        destination = evaluation_documents_folder if is_evaluation_document(doc,organization) else design_documents_folder
                        shutil.copyfile(f'{country_folder}/{project}/{doc}', f'{destination}/{doc}')
                



def analyse_corpus():
    '''Analyse corpus: number of projects, number of documents of each organization, etc.'''
    report = {
        'Number of projects': 0,
        'Number of documents': 0,
        'AsDB': {},
        'WB': {},
        'IFAD': {},
        'UNDP': {},
        'AfDB': {}
    }

    corpus_folder = f'{file_path}/corpus'
    documents_folder = f'{corpus_folder}/documents'
    evaluation_documents_folder = f'{corpus_folder}/documents/evaluation'
    design_documents_folder = f'{corpus_folder}/documents/design'
    projects_folder = f'{corpus_folder}/projects/partner'

    # check projects
    organization_projects_folder = os.listdir(projects_folder)
    for organization in organization_projects_folder:
        report['Number of projects'] += len(os.listdir(f'{projects_folder}/{organization}'))
        report[organization]['Number of projects'] = len(os.listdir(f'{projects_folder}/{organization}'))

    # check documents
    documents = os.listdir(evaluation_documents_folder) + os.listdir(design_documents_folder)
    report['Number of documents'] = len(documents)
    for doc in documents:
        if 'TRUE' not in doc:
            continue
        
        fields = doc.split('_')
        organization = fields[0]
        type = fields[2]

        if type not in report[organization]:
            report[organization][type] = 0

        report[organization][type] += 1


    return report

def analyse_corpus_excel():
    '''Analyse corpus and save it in excel.'''

    control_projects_path = f'{file_path}/corpus/projects/control'
    partner_projects_path = f'{file_path}/corpus/projects/partner'
    documents_path = f'{file_path}/corpus/documents'
    evaluation_documents_path = f'{file_path}/corpus/documents/evaluation'
    design_documents_path = f'{file_path}/corpus/documents/design'

    documents = {}
    type_docs = set()

    documents_list = os.listdir(evaluation_documents_path) + os.listdir(design_documents_path)

    for doc in documents_list:
        project_id = doc.split('_')[1]
        doc_type = doc.split('_')[2]

        if doc_type not in type_docs:
            type_docs.add(doc_type)

        if not project_id in documents:
            documents[project_id] = set()
        
        documents[project_id].add(doc_type)

    projects = {}

    # control projects
    for organization in os.listdir(control_projects_path):
        for project in os.listdir(f'{control_projects_path}/{organization}'):
            metadata = json.load(open(f'{control_projects_path}/{organization}/{project}/metadata.json'))
            projects[project] = {
                'organization': organization,
                'type': 'control',
                'partners' : [],
                'country_iso3': metadata['country_iso3'],
                'rating': metadata['project_rating'] if 'project_rating' in metadata else None,
                'end_date': metadata['end_date'],
                'sector' : metadata['sector'] if type(metadata['sector']) == list else [metadata['sector']],
                'doc_types': list(documents[project])
            }


    # partner projects
    for organization in os.listdir(partner_projects_path):
        for project in os.listdir(f'{partner_projects_path}/{organization}'):
            metadata = json.load(open(f'{partner_projects_path}/{organization}/{project}/metadata.json'))
            projects[project] = {
                'organization': organization,
                'type': 'partner',
                'partners' : metadata['partnerships'],
                'country_iso3': metadata['country_iso3'],
                'rating': metadata['project_rating'] if 'project_rating' in metadata else None,
                'end_date': metadata['end_date'],
                'sector' : metadata['sector'] if type(metadata['sector']) == list else [metadata['sector']],
                'doc_types': list(documents[project])
            }


    max_partners = max([len(projects[project]['partners']) for project in projects])
    max_sectors = max([len(projects[project]['sector']) for project in projects])

    columns = ['proCode','orgAcro1']
    for i in range(max_partners):
        columns.append(f'orgAcro{i+2}')

    columns += ['coFin','rating','country']

    for i in range(max_sectors):
        columns.append(f'sector{i+1}')

    columns.append('year')

    for doc_type in type_docs:
        columns.append(doc_type)


    df = pd.DataFrame(data=[],columns=columns)

    for project in projects:
        row = {
            'proCode': project,
            'orgAcro1': projects[project]['organization']
        }

        for i in range(max_partners):
            if i < len(projects[project]['partners']):
                row[f'orgAcro{i+2}'] = projects[project]['partners'][i]
            else:
                row[f'orgAcro{i+2}'] = None

        if projects[project]['type'] == 'partner':
            row['coFin'] = 'true'
        else:
            row['coFin'] = 'false'

        row['rating'] = projects[project]['rating']
        row['country'] = projects[project]['country_iso3']

        for i in range(max_sectors):
            if i < len(projects[project]['sector']):
                row[f'sector{i+1}'] = projects[project]['sector'][i]
            else:
                row[f'sector{i+1}'] = None

        row['year'] = projects[project]['end_date']

        for doc_type in type_docs:
            if doc_type in projects[project]['doc_types']:
                row[doc_type] = 'true'
            else:
                row[doc_type] = 'false'

        df.loc[len(df)] = row

    df.to_excel('corpus_situation_extended.xlsx')



def clean_organizations(organizations:list[str], debug:bool=False):
    '''Remove organizations from corpus'''

    for organization in organizations:
        if debug:
            print('Removing',organization)

        organization = organization.lower()
        organization = {
            'wb': 'WB',
            'ifad': 'IFAD',
            'afdb': 'AfDB',
            'asdb': 'AsDB',
            'undp': 'UNDP',
        }[organization]

        # remove projects
        ## partner
        partner_projects_path = f'{file_path}/corpus/projects/partner'
        shutil.rmtree(f'{partner_projects_path}/{organization}')
        ## control
        control_projects_path = f'{file_path}/corpus/projects/control'
        shutil.rmtree(f'{control_projects_path}/{organization}')
        # files
        ## evaluation
        evaluation_documents = os.listdir(f'{file_path}/corpus/documents/evaluation')
        for doc in evaluation_documents:
            if doc.startswith(organization):
                os.remove(f'{file_path}/corpus/documents/evaluation/{doc}')
        ## design
        design_documents = os.listdir(f'{file_path}/corpus/documents/design')
        for doc in design_documents:
            if doc.startswith(organization):
                os.remove(f'{file_path}/corpus/documents/design/{doc}')


if __name__ == '__main__':
    args = process_args()

    # create folders if they don't exist
    if not os.path.exists(f'{file_path}/corpus'):
        os.makedirs(f'{file_path}/corpus')
    if not os.path.exists(f'{file_path}/corpus/documents'):
        os.makedirs(f'{file_path}/corpus/documents')
    if not os.path.exists(f'{file_path}/corpus/documents/design'):
        os.makedirs(f'{file_path}/corpus/documents/design')
    if not os.path.exists(f'{file_path}/corpus/documents/evaluation'):
        os.makedirs(f'{file_path}/corpus/documents/evaluation')
    if not os.path.exists(f'{file_path}/corpus/projects'):
        os.makedirs(f'{file_path}/corpus/projects')
    if not os.path.exists(f'{file_path}/corpus/projects/control'):
        os.makedirs(f'{file_path}/corpus/projects/control') 
    if not os.path.exists(f'{file_path}/corpus/projects/partner'):
        os.makedirs(f'{file_path}/corpus/projects/partner') 

    if args.partner_corpus:
        prepare_organizations_data_folders(args.debug)
        create_corpus()
        clean_corpus(args.debug)

    if args.clean_corpus and not args.partner_corpus:
        clean_corpus(args.debug)

    if args.control_corpus:
        prepare_organizations_control_corpus(args.debug)
        create_control_corpus()


    if args.analyse_corpus:
        print(analyse_corpus())

    if args.analyse_corpus_excel:
        analyse_corpus_excel()

    if args.clean_organizations:
        clean_organizations(args.clean_organizations, args.debug)


