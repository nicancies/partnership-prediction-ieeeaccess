import json
import re
import sys
import traceback
import fitz
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils.utils import *


current_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = f'{current_folder}/data'
projects_bar = tqdm(total=0)

# load IFI list
IFI_list_file_path = f'{current_folder}/../IFI_list.json'
IFI_dict = json.load(open(IFI_list_file_path,'r',encoding='utf-8'))

# clean IFI dict
IFI_dict = {key.lower().strip():[value.lower().strip() for value in values] for key,values in IFI_dict.items()}


header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36'}

IFI_list = []
for key,values in IFI_dict.items():
    values = [key] + [value for value in values]
    IFI_list += values



def text_partners(text:str):
    '''Check if text has partners'''
    partnerships = []
    text = text.lower().replace('\n',' ')
    for partner in IFI_list:
        pattern = r'([^\w\-‐]|^){}([^\w\-‐]|$)'.format(partner)
        if re.search(pattern, text):
            partnerships.append(partner)
    return partnerships


def map_partners(partners:set):
    '''Map partners using IFI dict'''
    mapped_partners = set()
    for partner in partners:
        # check if partner is in IFI dict
        if partner in IFI_dict:
            mapped_partners.add(partner)
        else:
            # check if partner is alternative name for an IFI
            for ifi in IFI_dict.keys():
                if partner in IFI_dict[ifi]:
                    mapped_partners.add(ifi)

    if 'asdb' in mapped_partners:
        mapped_partners.remove('asdb')
    return mapped_partners

def filter_partners(partners:list):
    '''Filter partners using IFI list'''
    partners = [partner.strip().lower() for partner in partners if partner.strip()]
    filtered_partners = set()
    for ifi in IFI_list:
        for partner in partners:
            if ifi in partner:
                filtered_partners.add(ifi)

    # specific cases
    ## african development bank and asian development bank ambiguous
    if 'adb' in filtered_partners:
        filtered_partners.remove('adb')

    return list(map_partners(filtered_partners))


def search_project_page_partnerships(project_id):
    '''Search in project page for funding partners'''
    partnerships = []
    projects_url = f'https://www.adb.org/projects/{project_id}/main'
    r = get_request(projects_url,headers=header)
    if r:
        soup = BeautifulSoup(r.text, 'lxml')
        # get all tables with
        tables = soup.find_all('table')
        # funding row
        ## look for row with 'source of Funding / amount' as first column
        funding_row = None
        for table in tables:
            row = table.find_all('tr')
            for tr in row:
                if 'source of funding / amount' in tr.text.lower().strip():
                    funding_row = tr
                    break
            if funding_row:
                break
        
        if funding_row:
            funders_table = funding_row.find_next('table')
            if funders_table:
                rows = funders_table.find_all('tr')
                for row in rows:
                    columns = row.find_all('td')
                    if columns:
                        if partners := text_partners(columns[0].text):
                            partnerships += partners

    return filter_partners(partnerships)



def report_recommendation_president_partnerships(pdf_path):
    '''Identify partnerships in report and recommendation from the president.pdf'''
    partnerships = []
    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = fitz.open(pdf_file)
        for page in pdf_reader:
            blocks = page.get_text('blocks', 
                                flags=fitz.TEXTFLAGS_BLOCKS|fitz.TEXT_DEHYPHENATE)
            page_text = ''
            for block in blocks:
                text = block[4]
                page_text += text

                # financing plan table
                if 'financing plan' in page_text.lower():
                    # word 'table' in the same line as 'financing plan'
                    financing_plan_table_page = False
                    lines = page_text.split('\n')
                    for line in lines:
                        if 'table' in line.lower() and 'financing plan' in line.lower():
                            financing_plan_table_page = True
                            break

                    if financing_plan_table_page:
                        print('financing plan table page')
                        paragraphs = page_text.split('.\n')
                        for paragraph in paragraphs:
                            found_partners = text_partners(paragraph)
                            if found_partners:
                                partnerships += found_partners

            if partnerships:
                break

    except Exception as e:
        print(f'Error reading project completion report: {traceback.format_exc()}')
    return filter_partners(partnerships)


def project_completion_report_partnerships(pdf_path):
    '''Identify project's partnerships in project completion report'''
    partnerships = []
    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = fitz.open(pdf_file)
        for page in pdf_reader:
            blocks = page.get_text('blocks', 
                                flags=fitz.TEXTFLAGS_BLOCKS|fitz.TEXT_DEHYPHENATE)
            page_text = ''
            for block in blocks:
                text = block[4]
                page_text += text
            # check if 'cofin' in page
            ## check for IFI in paragraph
            if 'cofin' in page_text.lower() or 'co-fin' in page_text.lower():
                print('cofin page')
                paragraphs = page_text.split('.\n')
                for paragraph in paragraphs:
                    if 'cofin' in paragraph.lower():
                        found_partners = text_partners(paragraph)
                        if found_partners:
                            partnerships += found_partners
            
            if not partnerships:
                # ! some pages fail with find_tables
                try:
                    # project data table
                    ## find page with 'Project Data' and tables in page
                    ## if table found and potential partner, verify if he has funded the project (cells to the right not 0)
                    tables = page.find_tables()
                    if 'project data' in page_text.lower():
                        print('project data page')
                        for table in tables:
                            table: fitz.table.Table
                            lines = table.extract()
                            for i,line in enumerate(lines):
                                found_partners = text_partners(line[0])
                                if found_partners:
                                    partnerships += found_partners
                            
                except Exception as e:
                    pass
            
            if partnerships:
                break

        
    except Exception as e:
        print(f'Error reading project completion report: {traceback.format_exc()}')

    return filter_partners(partnerships)


def identify_partnerships(project_folder_path):
    '''Identify project's partnerships

    Search in project page for funding partners.

    If none found:
        Analyze downloaded documents and identify project's partnerships in these.
    
    Priority is given to different types
    '''
    partnerships = []

    project_id = project_folder_path.split('/')[-1]
    print('Searching project page for funding partners')
    partnerships = search_project_page_partnerships(project_id) 

    if not partnerships:
        documents = os.listdir(project_folder_path)

        rrp_partnerships = []
        checked_rrp = False
        if "rrp.pdf" in documents:
            try:
                # check if doc is readable
                pdf_file = open(f'{project_folder_path}/rrp.pdf', 'rb')
                pdf_reader = fitz.open(pdf_file)
                page_text = pdf_reader[0].get_text()

                checked_rrp = True
                print(f'Searching report and recommendation from the president.pdf')
                rrp_partnerships = report_recommendation_president_partnerships(f'{project_folder_path}/rrp.pdf')

            except Exception as e:
                print(f'Error reading project completion report: {e}')

        pcr_partnerships = []
        checked_pcr = False
        if "pcr.pdf" in documents:
            try:
                # check if doc is readable
                pdf_file = open(f'{project_folder_path}/pcr.pdf', 'rb')
                pdf_reader = fitz.open(pdf_file)
                page_text = pdf_reader[0].get_text()

                checked_pcr = True
                print(f'Searching project completion report.pdf')
                pcr_partnerships = project_completion_report_partnerships(f'{project_folder_path}/pcr.pdf')
            except Exception as e:
                print(f'Error reading project completion report: {e}')
        
        # cross check rrp and pcr partners
        ## only partnerships found in both
        if checked_rrp and checked_pcr:
            for rrp_partnership in rrp_partnerships:
                if rrp_partnership in pcr_partnerships:
                    partnerships.append(rrp_partnership)
        else:
            if rrp_partnerships:
                partnerships += rrp_partnerships
            if pcr_partnerships:
                partnerships += pcr_partnerships
            

    print(f'Partnerships found: {partnerships}')
    return partnerships

def main():
    partnerships_found = 0
    # iterate over projects
    for country_folder in os.listdir(data_folder):
        if os.path.isdir(f'{data_folder}/{country_folder}'):
            for project_folder in os.listdir(f'{data_folder}/{country_folder}'):
                projects_bar.update(1)

                project_folder_path = f'{data_folder}/{country_folder}/{project_folder}'
                metadata_file_path = f'{project_folder_path}/metadata.json'
                project_files = os.listdir(project_folder_path)
                print(f'{project_folder_path}')
                if 'metadata.json' in project_files and len(project_files) > 1:
                    metadata_file = json.load(open(metadata_file_path, 'r', encoding='utf-8'))
                    partnerships = []
                    if len(project_files) > 1:
                        partnerships = identify_partnerships(project_folder_path)
                    metadata_file['partnerships'] = partnerships
                    json.dump(metadata_file, open(metadata_file_path, 'w', encoding='utf-8'), indent=4)
                    if partnerships:
                        partnerships_found += 1

    print(f'Found {partnerships_found} projects with partnerships')








if __name__ == "__main__":
    # find number of projects
    n_projects = 0
    for country_folder in os.listdir(data_folder):
        if os.path.isdir(f'{data_folder}/{country_folder}'):
            for project_folder in os.listdir(f'{data_folder}/{country_folder}'):
                if os.path.isdir(f'{data_folder}/{country_folder}/{project_folder}'):
                    n_projects += 1

    projects_bar.reset(total=n_projects)
    main()