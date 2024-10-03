import os
import json
import shutil

file_path = os.path.dirname(os.path.abspath(__file__))

def create_control_corpus():
    
    # create folder
    control_corpus_path = f'{file_path}/control_corpus'
    if not os.path.exists(control_corpus_path):
        os.makedirs(control_corpus_path)


    organizations_list = ['AsDB','IFAD','WorldBank','UNDP']

    # iterate over organizations
    for organization in organizations_list:
        print(f'Creating control corpus for {organization}')
        # check if organization folder exists
        organization_control_corpus_folder = f'{file_path}/{organization}/control_corpus'

        if os.path.isdir(organization_control_corpus_folder):
            
            # create folder for organization in control corpus
            if not os.path.exists(f'{control_corpus_path}/{organization}'):
                os.mkdir(f'{control_corpus_path}/{organization}')

            # copy organization control corpus to control corpus
            projects = os.listdir(organization_control_corpus_folder)
            for project in projects:
                shutil.copytree(f'{organization_control_corpus_folder}/{project}', f'{control_corpus_path}/{organization}/{project}')


def add_project_metadata():

    control_corpus_path = f'{file_path}/control_corpus'

    organizations_list = ['AsDB','IFAD','WorldBank','UNDP']

    # iterate over organizations
    for organization in organizations_list:
        print(f'Adding metadata to control corpus for {organization}')
        # check if organization folder exists
        organization_control_corpus_folder = f'{file_path}/control_corpus/{organization}'

        if os.path.isdir(organization_control_corpus_folder):
            

            for country_folder in os.listdir(organization_control_corpus_folder):
                if os.path.isdir(f'{organization_control_corpus_folder}/{country_folder}'):
                    country_folder_path = f'{organization_control_corpus_folder}/{country_folder}'
                    if os.path.isdir(country_folder_path):
                        # iterate over projects
                        for project_folder in os.listdir(country_folder_path):
                            project_folder_path = f'{country_folder_path}/{project_folder}'
                            if os.path.isdir(project_folder_path):
                                project_files = os.listdir(project_folder_path)
                                # read original metadata
                                data_folder = f'{file_path}/{organization}/data' if organization != 'UNDP' else f'{file_path}/{organization}/control_corpus'
                                o_metadata_file = json.load(open(f'{data_folder}/{country_folder}/{project_folder}/metadata.json', 'r', encoding='utf-8'))
                                metadata_file = json.load(open(f'{project_folder_path}/metadata.json', 'r', encoding='utf-8'))
                                metadata_file['end_date'] = o_metadata_file['end_date']
                                metadata_file['country_iso3'] = o_metadata_file['country_iso3']

                                with open(f'{project_folder_path}/metadata.json', 'w', encoding='utf-8') as f:
                                    json.dump(metadata_file, f, indent=4)



if __name__ == "__main__":
    
    create_control_corpus()
    #add_project_metadata()