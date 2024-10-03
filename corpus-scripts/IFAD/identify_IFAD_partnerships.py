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
    
    if 'ifad' in mapped_partners:
        mapped_partners.remove('ifad')
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
            # ! some pages fail with find_tables
            try:

                # actual costs table
                tables = page.find_tables()
                if 'actual costs and financing' in page_text.lower():
                    print('actual costs and financing page')
                    for table in tables:
                        table: fitz.table.Table
                        for line in table.extract():
                            text = line[0]
                            found_partners = text_partners(text)
                            if found_partners:
                                partnerships += found_partners
                        

                # financial performance by financier table
                ## search for page that says 'Financial performance by financiers'
                if not partnerships and 'financial performance by financiers' in page_text.lower():
                    print('financial performance by financiers page')
                    for table in tables:
                        table: fitz.table.Table
                        for line in table.extract():
                            text = line[0]
                            found_partners = text_partners(text)
                            if found_partners:
                                partnerships += found_partners
            except Exception as e:
                pass
            

            # project costs and financing chapter
            ## find partners in text
            ## if partner close to word "financing" (in paragraph), save partner
            if not partnerships and 'project costs and financing' in page_text.lower():
                print('project costs and financing page')
                paragraphs = page_text.split('.\n')
                for paragraph in paragraphs:
                    found_partners = text_partners(paragraph)
                    if found_partners:
                        # check if 'financing' is in neighborhood
                        if 'financing' in paragraph.lower():
                            partnerships += found_partners
            
            if partnerships:
                break

        
    except Exception as e:
        print(f'Error reading project completion report: {traceback.format_exc()}')
        
    return filter_partners(partnerships)

    


def project_design_report_partnerships(pdf_path):
    '''Identify project's partnerships in project design report'''

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
            # ! some pages fail with find_tables
            try:

                # financing plan table
                ## find page with 'Financing Plan' and tables in page
                ## if table found and potential partner, verify if he has funded the project (cells bellow not 0)
                tables = page.find_tables()
                if 'financing plan' in page_text.lower():
                    print('financing plan page')
                    for table in tables:
                        table: fitz.table.Table
                        lines = table.extract()
                        for i,line in enumerate(lines):
                            for j,cell in enumerate(line):
                                found_partners = text_partners(cell)
                                if found_partners:
                                    # verify cells below not 0
                                    funded = False
                                    for k in range(i+1,len(lines)):
                                        try:
                                            funded = float(lines[k][j]) > 0
                                            if funded:
                                                break
                                        except:
                                            pass

                                    if funded:
                                        partnerships += found_partners
                        
                # financing plan by component table
                ## find page with 'Financing Plan by Component' and tables in page
                ## if table found and potential partner, verify if he has funded the project (cells bellow not 0)
                if not partnerships and 'financing plan by component' in page_text.lower():
                    print('financing plan by component page')
                    for table in tables:
                        table: fitz.table.Table
                        lines = table.extract()
                        for i,line in enumerate(lines):
                            for j,cell in enumerate(line):
                                found_partners = text_partners(cell)
                                if found_partners:
                                    # verify cells below not 0
                                    funded = False
                                    for k in range(i+1,len(lines)):
                                        try:
                                            funded = float(lines[k][j]) > 0
                                            if funded:
                                                break
                                        except:
                                            pass

                                    if funded:
                                        partnerships += found_partners
            except Exception as e:
                pass
            

            # project financing chapter
            ## find partners in text
            ## if partner close to word "usd","us$" or "$" (in paragraph), save partner
            if not partnerships and 'project financing' in page_text.lower():
                print('project financing page')
                paragraphs = page_text.split('.\n')
                for paragraph in paragraphs:
                    found_partners = text_partners(paragraph)
                    if found_partners:
                        # check if monetary reference is in neighborhood
                        if 'usd' in paragraph.lower() or 'us$' in paragraph.lower() or '$' in paragraph.lower():
                            partnerships += found_partners
            
            if partnerships:
                break

        
    except Exception as e:
        print(f'Error reading project completion report: {traceback.format_exc()}')
        
    return filter_partners(partnerships)





def presidents_report_partnerships(pdf_path):
    '''Identify project's partnerships in president's report'''

    partnerships = []
    cofinancers_listing_pattern = r'Cofinancier(\(s\)|s)?:\d*\s*((\w|\s)*?\(\w+\))+'
    cofinancers_pattern = r'(\w|\s)*?\(\w+\)'
    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = fitz.open(pdf_file)
        

        # find cofinancers listing
        cofinancers_listing = None
        for page_num in range(len(pdf_reader)):
            page = pdf_reader[page_num]
            blocks = page.get_text('blocks', 
                                flags=fitz.TEXTFLAGS_BLOCKS|fitz.TEXT_DEHYPHENATE)
            page_text = ''
            for block in blocks:
                text = block[4]
                page_text += text
            if re.search(cofinancers_listing_pattern, page_text):
                cofinancers_listing = re.search(cofinancers_listing_pattern, page_text).group(0)
                break

        # extract cofinancers
        if cofinancers_listing:
            print(cofinancers_listing)
            cofinancers_listing = cofinancers_listing.split(':')[1]
            partnerships = re.finditer(cofinancers_pattern, cofinancers_listing)
            partnerships = [p.group(0).strip() for p in partnerships]

    except Exception as e:
        print(f'Error reading president\'s report: {e}')

    return filter_partners(partnerships)


def project_completion_report_digest_partnerships(pdf_path):
    '''Identify project's partnerships in project completion report'''
    partnerships = []

    cofinancers_listing_pattern_1 = r'Cofinanciers \(if any\)\s*(([^:]+:[^:;]+\d)+)'
    cofinancers_listing_pattern_2 = r'Cofinanciers \(if any\)\s*(([^-]+-[^-;]+\d)+)'
    pattern_used = None
    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = fitz.open(pdf_file)
        

        # find cofinancers listing
        cofinancers_listing = None
        for page_num in range(len(pdf_reader)):
            page = pdf_reader[page_num]
            blocks = page.get_text('blocks', 
                                flags=fitz.TEXTFLAGS_BLOCKS|fitz.TEXT_DEHYPHENATE)
            page_text = ''
            for block in blocks:
                text = block[4]
                page_text += text
            if re.search(cofinancers_listing_pattern_1, page_text):
                cofinancers_listing = re.search(cofinancers_listing_pattern_1, page_text).group(1)
                pattern_used = 1
                break
            elif re.search(cofinancers_listing_pattern_2, page_text):
                cofinancers_listing = re.search(cofinancers_listing_pattern_2, page_text).group(1)
                pattern_used = 2
                break

        # extract cofinancers
        if cofinancers_listing:
            cofinancers_listing = cofinancers_listing.split(';')
            if pattern_used == 1:
                partnerships = [p.split(':')[0].strip() for p in cofinancers_listing] 
            elif pattern_used == 2:
                partnerships = [p.split('-')[0].strip() for p in cofinancers_listing] 


    except Exception as e:
        print(f'Error reading project completion report: {e}')

    return filter_partners(partnerships)


def identify_partnerships(project_folder_path,metadata=None):
    '''Identify project's partnerships
    Analyze downloaded documents and identify project's partnerships in these.
    
    Priority is given to different types
    '''
    partnerships = []

    if metadata:
        cofinancers = metadata['co-financiers (international)'] if 'co-financiers (international)' in metadata else None

        if cofinancers:
            partnerships = filter_partners(text_partners(cofinancers))
            if partnerships:
                return partnerships
            
    documents = os.listdir(project_folder_path)

    partners_checks = {
        'pcr' : {
            'check' : False,
            'partners' : []
        },
        'pdr' : {
            'check' : False,
            'partners' : []
        },
        'pr' : {
            'check' : False,
            'partners' : []
        },
        'pcrd' : {
            'check' : False,
            'partners' : []
        }
    }

    if "pcr.pdf" in documents:
        try:
            # check if doc is readable
            pdf_file = open(f'{project_folder_path}/project completion report.pdf', 'rb')
            pdf_reader = fitz.open(pdf_file)
            page_text = pdf_reader[0].get_text()

            partners_checks['pcr']['check'] = True
            print(f'Searching project completion report.pdf')
            partners_checks['pcr']['partners'] = project_completion_report_partnerships(f'{project_folder_path}/project completion report.pdf')

        except Exception as e:
            print(f'Error opening project completion report: {e}')

    if "pdr.pdf" in documents or "fpdr.pdf" in documents:
        try:
            # check if doc is readable
            doc = f'{project_folder_path}/pdr.pdf' if "pdr.pdf" in documents else f'{project_folder_path}/fpdr.pdf'
            pdf_file = open(doc, 'rb')
            pdf_reader = fitz.open(pdf_file)
            page_text = pdf_reader[0].get_text()

            partners_checks['pdr']['check'] = True
            print(f'Searching project design report.pdf')
            if "fpdr.pdf" in documents:
                partners_checks['pdr']['partners'] = project_design_report_partnerships(f'{project_folder_path}/fpdr.pdf')
            else:
                partners_checks['pdr']['partners'] = project_design_report_partnerships(f'{project_folder_path}/pdr.pdf')

        except Exception as e:
            print(f'Error opening project design report: {e}')

    if "pr.pdf" in documents:
        try:
            # check if doc is readable
            pdf_file = open(f'{project_folder_path}/pr.pdf', 'rb')
            pdf_reader = fitz.open(pdf_file)
            page_text = pdf_reader[0].get_text()

            partners_checks['pr']['check'] = True
            print(f'Searching president\'s report.pdf')
            partners_checks['pr']['partners'] = presidents_report_partnerships(f'{project_folder_path}/pr.pdf')

        except Exception as e:
            print(f'Error opening president\'s report: {e}')


    if "pcrd.pdf" in documents:
        try:
            # check if doc is readable
            pdf_file = open(f'{project_folder_path}/pcrd.pdf', 'rb')
            pdf_reader = fitz.open(pdf_file)
            page_text = pdf_reader[0].get_text()

            partners_checks['pcrd']['check'] = True
            print(f'Searching project completion report digest.pdf')
            partners_checks['pcrd']['partners'] = project_completion_report_digest_partnerships(f'{project_folder_path}/pcrd.pdf')

        except Exception as e:
            print(f'Error opening project completion report digest: {e}')

    print('Partners checks:',partners_checks)
    # cross check partners
    for doc_type, doc_partners in partners_checks.items():
        if doc_partners['check']:
            for partner in doc_partners['partners']:
                appearances = 1
                n_checks = 1
                for doc_type_2, doc_partners_2 in partners_checks.items():
                    if doc_partners_2['check'] and doc_type != doc_type_2  :
                        n_checks += 1
                        if partner in doc_partners_2['partners']:
                            appearances += 1
                
                if n_checks == 1 or (n_checks > 1 and appearances > 1):
                    partnerships.append(partner)

    print(f'Partners found: {partnerships}')
    return partnerships

def main():
    partnerships_found = 0

    # iterate over projects
    for country_folder in os.listdir(data_folder):
        if os.path.isdir(f'{data_folder}/{country_folder}'):
            for project_folder in os.listdir(f'{data_folder}/{country_folder}'):
                project_folder_path = f'{data_folder}/{country_folder}/{project_folder}'
                metadata_file_path = f'{project_folder_path}/metadata.json'
                print(f'{project_folder_path}')
                if os.path.exists(metadata_file_path):
                    metadata_file = json.load(open(metadata_file_path, 'r', encoding='utf-8'))
                    partnerships = identify_partnerships(project_folder_path,metadata_file)
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