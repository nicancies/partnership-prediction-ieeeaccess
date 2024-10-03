import json
import re
import sys
import traceback
import fitz
import os
from tqdm import tqdm


current_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = f'{current_folder}/data'
projects_bar = tqdm(total=0)

# load IFI list
IFI_list_file_path = f'{current_folder}/../IFI_list.json'
IFI_dict = json.load(open(IFI_list_file_path,'r',encoding='utf-8'))

# clean IFI dict
IFI_dict = {key.lower().strip():[value.lower().strip() for value in values] for key,values in IFI_dict.items()}

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
    
    # remove specific cases
    if 'wb' in mapped_partners:
        mapped_partners.remove('wb')

    if 'ida' in mapped_partners:
        mapped_partners.remove('ida')

    if 'ibrd' in mapped_partners:
        mapped_partners.remove('ibrd')

    if 'ifc' in mapped_partners:
        mapped_partners.remove('ifc')

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
    if ('asian development bank' in filtered_partners and 'adb' in filtered_partners) and not 'african development bank' in filtered_partners:
        filtered_partners.remove('adb')
    if ('african development bank' in filtered_partners and 'adb' in filtered_partners) and not 'asian development bank' in filtered_partners:
        filtered_partners.remove('adb')

    return list(map_partners(filtered_partners))


def project_appraisal_document_partnerships(pdf_path):
    '''Identify project's partnerships in project appraisal document'''
    partnerships = []

    found_asdb = False
    found_afdb = False

    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = fitz.open(pdf_file)
        
        for page_num in range(len(pdf_reader)):
            page = pdf_reader[page_num]
            blocks = page.get_text('blocks', 
                                flags=fitz.TEXTFLAGS_BLOCKS|fitz.TEXT_DEHYPHENATE)
            page_text = ''
            for block in blocks:
                text = block[4]
                page_text += text

            if 'african development bank' in page_text.lower():
                found_afdb = True

            if 'asian development bank' in page_text.lower():
                found_asdb = True


            # find paragraph with word financier
            ## within paragraph search for potential partners
            if not partnerships and ('financier' in page_text.lower() or 'finance' in page_text.lower() or 'cofinanc' in page_text.lower() or 'co-financ' in page_text.lower() or 'financing' in page_text.lower()):
                lines = page_text.lower().split('\n')
                for i,line in enumerate(lines):
                    found_partners = text_partners(line)
                    if found_partners:
                        big_neighborhood = lines[i-3:i+3] if i > 3 else lines[i:i+3]
                        big_neighborhood = ' '.join(big_neighborhood)
                        small_neighborhood = lines[i-1:i+1] if i > 0 else lines[i:i+1]
                        small_neighborhood += ' '.join(small_neighborhood)
                        # check if monetary reference is in paragraph
                        if ('usd' in small_neighborhood or 'us$' in small_neighborhood or '$' in small_neighborhood) and \
                            ('financier' in big_neighborhood or 'finance' in big_neighborhood or 'cofinanc' in big_neighborhood or 'co-financ' in big_neighborhood or 'financing' in big_neighborhood):
                            partnerships += filter_partners(found_partners)
            
            if partnerships:
                break

    except Exception as e:
        print(f'Error reading president\'s report: {e}')

    
    # try to remove ambiguous partners
    if not found_asdb and found_afdb and 'asdb' in partnerships:
        partnerships.remove('asdb')

    if not found_afdb and found_asdb and 'afdb' in partnerships:
        partnerships.remove('afdb')

    return filter_partners(partnerships)


def implementation_completion_results_report_partnerships(pdf_path):
    '''Identify project's partnerships in implementation completion and results report'''
    partnerships = []

    found_asdb = False
    found_afdb = False

    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = fitz.open(pdf_file)
        
        for page_num in range(len(pdf_reader)):
            page = pdf_reader[page_num]
            blocks = page.get_text('blocks', 
                                flags=fitz.TEXTFLAGS_BLOCKS|fitz.TEXT_DEHYPHENATE)
            page_text = ''
            for block in blocks:
                text = block[4]
                page_text += text

            if 'african development bank' in page_text.lower():
                found_afdb = True

            if 'asian development bank' in page_text.lower():
                found_asdb = True


            # financing or finance
            if 'financing' in page_text.lower() or 'financed' in page_text.lower():
                lines = page_text.lower().split('\n')
                for i,line in enumerate(lines):
                    found_partners = text_partners(line)
                    if found_partners:
                        neighborhood = lines[i-1:i+1] if i > 0 else lines[i:i+1]
                        neighborhood = ' '.join(neighborhood)
                        # check if monetary reference is in neighborhood
                        if re.search(r'[\d\,\.]+', neighborhood) and ('usd' in neighborhood or 'us$' in neighborhood or '$' in neighborhood):
                            partnerships += filter_partners(found_partners)
                        
            
            if partnerships:
                break

    except Exception as e:
        print(f'Error reading president\'s report: {e}')

    # try to remove ambiguous partners
    if not found_asdb and found_afdb and 'asdb' in partnerships:
        partnerships.remove('asdb')

    if not found_afdb and found_asdb and 'afdb' in partnerships:
        partnerships.remove('afdb')

    return filter_partners(partnerships)






def identify_partnerships(project_folder_path):
    '''Identify project's partnerships
    Analyze downloaded documents and identify project's partnerships in these.
    
    Priority is given to different types
    '''
    partnerships = []

    documents = os.listdir(project_folder_path)
    i = 0
    pa_partnerships = []
    icrr_partnerships = []
    checked_pa = False
    checked_icrr = False
    while i < len(documents):
        document = documents[i]
        if "Project Appraisal Document.pdf" in document:
            try:
                # check if doc is readable
                pdf_file = open(f'{project_folder_path}/{document}', 'rb')
                pdf_reader = fitz.open(pdf_file)
                page_text = pdf_reader[0].get_text()

                pa_partnerships = project_appraisal_document_partnerships(f'{project_folder_path}/{document}')
                checked_pa = True
            except Exception as e:
                print(f'Error reading project appraisal document: {e}')

        elif "Implementation Completion and Results Report.pdf" in document:
            try:
                # check if doc is readable
                pdf_file = open(f'{project_folder_path}/{document}', 'rb')
                pdf_reader = fitz.open(pdf_file)
                page_text = pdf_reader[0].get_text()
                
                icrr_partnerships = implementation_completion_results_report_partnerships(f'{project_folder_path}/{document}')
                checked_icrr = True
            except Exception as e:
                print(f'Error reading implementation completion and results report: {e}')
        i += 1

    # print(f'checked pa: {checked_pa}, checked icrr: {checked_icrr}')
    # print(f'pa_partnerships: {pa_partnerships}, icrr_partnerships: {icrr_partnerships}')

    # cross check pa and icrr partners
    ## only partnerships found in both
    if checked_pa and checked_icrr:
        for pa_partnership in pa_partnerships:
            if pa_partnership in icrr_partnerships:
                partnerships.append(pa_partnership)
    else:
        if checked_pa:
            partnerships += pa_partnerships
        if checked_icrr:
            partnerships += icrr_partnerships

    return partnerships

def main():
    partnerships_found = 0

    # iterate over projects
    for country_folder in os.listdir(data_folder):
        if os.path.isdir(f'{data_folder}/{country_folder}'):
            for project_folder in os.listdir(f'{data_folder}/{country_folder}'):
                project_folder_path = f'{data_folder}/{country_folder}/{project_folder}'
                metadata_file_path = f'{project_folder_path}/metadata.json'
                if os.path.exists(metadata_file_path):
                    metadata_file = json.load(open(metadata_file_path, 'r', encoding='utf-8'))
                    partnerships = identify_partnerships(project_folder_path)
                    metadata_file['partnerships'] = partnerships
                    json.dump(metadata_file, open(metadata_file_path, 'w', encoding='utf-8'), indent=4)
                    if partnerships:
                        partnerships_found += 1
                projects_bar.update(1)

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