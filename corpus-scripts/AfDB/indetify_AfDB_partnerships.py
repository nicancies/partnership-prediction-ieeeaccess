import os
import re
from typing import Union
import requests
import json
import fitz
import traceback
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils.utils import *

file_path = os.path.dirname(os.path.abspath(__file__))
data_folder = f'{file_path}/data/files'

header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36'}
project_page_url = 'https://www.afdb.org/en/projects-and-operations/'

# load IFI list
IFI_list_file_path = f'{file_path}/../IFI_list.json'
IFI_dict = json.load(open(IFI_list_file_path,'r',encoding='utf-8'))

# clean IFI dict
IFI_dict = {key.lower().strip():[value.lower().strip() for value in values] for key,values in IFI_dict.items()}

IFI_list = []
for key,values in IFI_dict.items():
    values = [key] + [value for value in values]
    IFI_list += values


country_iso_codes_path = f'{file_path}/afdb_countries_iso3_mapping.json'
country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))


def process_args():
    parser = argparse.ArgumentParser(
        description="Script to identify AfDB partnerships"
    )
    parser.add_argument('-f','--full'                   , action='store_true', default=False                                            , help="Full run. Default=False")
    parser.add_argument('-gm','--get_metadata'          , action='store_true', default=False                                            , help="Get projects metadata. Default=False")
    parser.add_argument('-tpm','--toggle_page_metadata' , action='store_false',default=True                                             , help="Toggle get project page metadata. Default=True")
    parser.add_argument('-idp','--identify_partners'    , action='store_true', default=False                                            , help="Identify projects with partners. Default=False")
    parser.add_argument('-l','--logs'                   , action='store_false',default=True                                             , help="Create logs. Default=True")
    parser.add_argument("-d","--debug"                  , action='store_true', default=False                                            , help="Debug mode. Default=False")
    return parser.parse_args()


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

    if 'afdb' in mapped_partners:
        mapped_partners.remove('afdb')
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


def map_rating(rating:Union[str,int])->int:
    '''Map rating to integer'''
    r = None
    str_to_int = {
        'highly unsatisfactory' : 1,
        'hu' : 1,
        'very unsatisfactory' : 1,
        'vu' : 1,
        'unsatisfactory' : 2,
        'u' : 2,
        'satisfactory' : 3,
        's' : 3,
        'highly satisfactory' : 4,
        'hs' : 4,
        'very satisfactory' : 4,
        'vs' : 4
    }

    if str(rating) in str_to_int:
        r = str_to_int[str(rating)]
    elif isinstance(rating,int):
        if rating < 1:
            r = 1
        elif rating > 4:
            r = 4
        else:
            r = rating
    
    return r


def get_project_metadata(metadata_path:str,doc_path:str,page_metadata:bool=True,logs:bool=False,debug:bool=False):
    '''Get project metadata. First look for project id in document. If found, go to project page and get metadata.'''

    new_metadata = {}

    if not os.path.exists(metadata_path) or not os.path.exists(doc_path):
        print(f'File not found: {metadata_path} or {doc_path}')
        return
    
    # search for project id in document
    project_id = None
    project_rating = None
    project_end_date = None
    with fitz.open(doc_path) as doc:
        for page_num,page in enumerate(doc):
            text = page.get_text('text').lower()
            lines = text.splitlines()

            # if page_num < 2:
            #     print(lines)
            #     tables = page.find_tables()
            #     for table in tables:
            #         print(table.extract())

            # get project id
            ## try multiple patterns
            ## different versions of this type of file may have different patterns
            if not project_id:
                if (match:=re.search(r'project (sap\s)?number.*?:\s*([\w\d]+(-|\/)[\w\d]+(-|\/)[\w\d]+(-|\/)[\w\d]+)',text.replace('\n',''))) is not None:
                    project_id = match.group(2)
                elif (match:=re.search(r'project (sap\s)?code.*?:\s*.*?([\w\d]+(-|\/)[\w\d]+(-|\/)[\w\d]+(-|\/)[\w\d]+)',text.replace('\n',''))) is not None:
                    project_id = match.group(2)
                elif (match:=re.search(r'sap no.:\s*.*?([\w\d]+(-|\/)[\w\d]+(-|\/)[\w\d]+(-|\/)[\w\d]+)',text.replace('\n',''))) is not None:
                    project_id = match.group(1)
                elif (match:=re.search(r'programme reference:\s*.*?([\w\d]+(-|\/)[\w\d]+(-|\/)[\w\d]+(-|\/)[\w\d]+)',text.replace('\n',''))) is not None:
                    project_id = match.group(1)
                else:
                    # page may be formatted in tables
                    if 'project number' in text or 'project code' in text or 'sap no.' in text or 'programme reference' in text:
                        tables = page.find_tables()
                        if tables:
                            # search for table with reference for id and look for id
                            for table in tables:
                                rows = table.extract()
                                # clean rows
                                rows = [[col for col in row if col] for row in rows]
                                for i,row in enumerate(rows):
                                    for j,col in enumerate(row):
                                        if not col: continue

                                        lower_col = col.lower()
                                        if 'project number' in lower_col or 'project code' in lower_col or 'sap no.' in lower_col or 'programme reference' in lower_col:
                                            try:
                                                # id should be in column below
                                                bellow_col = rows[i+1][j]
                                                if not bellow_col: continue

                                                bellow_col_lower = bellow_col.lower()
                                                if (match:=re.search(r'([\w\d]+(-|/)[\w\d]+(-|/)[\w\d]+(-|/)[\w\d]+)',bellow_col_lower)) is not None:
                                                    project_id = match.group(1)
                                                    break
                                            except:
                                                pass
                                    if project_id:
                                        break
                                if project_id:
                                    break


            # get project rating
            ## try multiple patterns

            if not project_rating:
                ## "overall assessment of outcome" rating
                if not project_rating and 'overall assessment of outcome' in text:
                    # get line with rating
                    for i,line in enumerate(lines):
                        if 'overall assessment of outcome' in line:
                            search_area = ''.join(lines[i:i+2])
                            if (match:=re.search(r'overall assessment of outcome\s*(\d+(\.\d+)?)',line)) is not None:
                                project_rating = round(float(match.group(1)))
                                break

                ## "overall project outcome score" rating
                if not project_rating and 'overall project outcome score' in text:
                    # get line with rating
                    for i,line in enumerate(lines):
                        if 'overall project outcome score' in line:
                            search_area = ''.join(lines[i:i+2])
                            if (match:=re.search(r'overall project outcome score\s*.*?(\d+(\.\d+)?)',search_area)) is not None:
                                project_rating = round(float(match.group(1)))
                                break

                ## "overall project completion rating" rating
                if not project_rating and 'overall project completion rating' in text:
                    # get line with rating
                    for i,line in enumerate(lines):
                        if 'overall project completion rating' in line:
                            search_area = ''.join(lines[i:i+2])
                            if (match:=re.search(r'overall project completion rating\s*.*?(\d+(\.\d+)?)',search_area)) is not None:
                                project_rating = round(float(match.group(1)))
                            elif (match:=re.search(r'overall project completion rating\s*.*?(\w+( +\w+)*)',search_area)) is not None:
                                project_rating = match.group(1)
                                break

                ## "overall project assessment" rating
                if not project_rating and 'overall project assessment' in text:
                    # get line with rating
                    for i,line in enumerate(lines):
                        if 'overall project assessment' in line:
                            # special case: target value is last column two lines below in this table
                            ## check if there are two lines below
                            if i+2 < len(lines):
                                target_line = lines[i+2]
                                # get all number ratings
                                ratings = re.findall(r'(\d+(\.\d+)?)',target_line)
                                ## target is last one
                                if ratings:
                                    project_rating = round(float(ratings[-1][0]))
                                    break


            # get project end date (only year)
            ## try multiple patterns

            if not project_end_date:
                ## "original closing date"
                if not project_end_date and 'original closing date' in text:
                    # get line with date
                    for line in lines:
                        if 'original closing date' in line:
                            if (match:=re.search(r'original closing date.*?(\d{4})',line)) is not None:
                                project_end_date = int(match.group(1))
                                break
                            elif (match:=re.search(r'original closing date.*?\d{2}.\d{2}.(\d{2})',text)) is not None:
                                project_end_date = int(f'20{match.group(1)}')

                ## "disbursement deadline"
                if not project_end_date and 'disbursement deadline' in text:
                    # get line with date
                    for line in lines:
                        if 'disbursement deadline' in line:
                            if (match:=re.search(r'disbursement deadline.*?(\d{4})',line)) is not None:
                                project_end_date = int(match.group(1))
                                break
                            elif (match:=re.search(r'disbursement deadline.*?\d{2}.\d{2}.(\d{2})',text)) is not None:
                                project_end_date = int(f'20{match.group(1)}')

                ## "closing"
                if not project_end_date and 'closing' in text and 'actual date' in text:
                    if (match:=re.search(r'closing\s*.*?\d{4}\s*.*?(\d{4})',text)) is not None:
                        project_end_date = int(match.group(1))
                    elif (match:=re.search(r'closing\s*.*?\d{2}.\d{2}.\d{2}\s*.*?\d{2}.\d{2}.(\d{2})',text)) is not None:
                        project_end_date = int(f'20{match.group(1)}')


            if project_id and project_rating and project_end_date:
                break

    if project_id:
        if '/' in project_id:
            project_id = project_id.replace('/','-')
        if logs:
            print(f'Found project id: {project_id}')
        new_metadata['id'] = project_id


    # get project page metadata
    if page_metadata:


        project_page = f'{project_page_url}{project_id}'
        r = get_request(project_page,headers=header)
        if r:
            new_metadata['project_url'] = project_page
            soup = BeautifulSoup(r.text, 'lxml')
            # get all tables
            tables = soup.find_all('tbody')

            # project details table
            project_details_table = None
            for table in tables:
                if 'project details' in table.text.lower():
                    project_details_table = table
                    break

            if project_details_table:
                # search column (td) with
                ## "name" - project name
                ## "status" - project status
                ## "country" - project country
                ## "sector" - project sector
                ## "approval date" - project approval date
                ## "total cost" - project total cost
                ### "currency" - cost currency
                for td in project_details_table.find_all('td'):
                    col_text = td.text.lower()
                    if debug:
                        print(col_text)

                    if 'name:' in col_text:
                        new_metadata['name'] = td.text.split(':')[1].strip()
                    elif 'status:' in col_text:
                        new_metadata['status'] = td.text.split(':')[1].strip()
                    elif 'country:' in col_text:
                        new_metadata['country'] = td.text.split(':')[1].strip()
                        new_metadata['country_iso3'] = country_iso_codes[new_metadata['country']]
                    elif 'sector:' in col_text:
                        new_metadata['sector'] = td.text.split(':')[1].strip()
                    elif 'approval date:' in col_text:
                        new_metadata['approval_date'] = td.text.split(':')[1].strip()
                    elif 'total cost:' in col_text:
                        new_metadata['total_budget'] = re.search(r'total cost:\s*(\d+(\.\d+)?)',col_text).group(1).strip()
                        if 'currency:' in col_text:
                            new_metadata['total_budget_currency'] = re.search(r'currency:\s*(\w+)',col_text).group(1).strip()

    if project_rating:
        new_metadata['project_rating'] = map_rating(project_rating)

    if project_end_date:
        new_metadata['end_date'] = project_end_date

    # save metadata
    if new_metadata:
        if debug:
            print(new_metadata)

        metadata = json.load(open(metadata_path,'r',encoding='utf-8'))
        metadata.update(new_metadata)
        json.dump(metadata,open(metadata_path,'w',encoding='utf-8'),indent=4)



def identify_partners(metadata_path:str,doc_path:str,logs:bool=False,debug:bool=False)->list[str]:
    '''Identify partners. Only for projects with minimum metadata. Partnerships are found in document text.'''

    if not os.path.exists(metadata_path) or not os.path.exists(doc_path):
        print(f'File not found: {metadata_path} or {doc_path}')
        return
    
    metadata = json.load(open(metadata_path,'r',encoding='utf-8'))
    if not metadata or 'id' not in metadata or 'project_url' not in metadata:
        print(f'Metadata incomplete: {metadata_path}')
        return
    
    partnerships = []
    try:
        # search document
        with fitz.open(doc_path) as doc:
            for page in doc:
                blocks = page.get_text('blocks', 
                                        flags=fitz.TEXTFLAGS_BLOCKS|fitz.TEXT_DEHYPHENATE)
                page_text = ''
                for block in blocks:
                    text = block[4]
                    page_text += text

                page_text_lower = page_text.lower()

                # search in paragraph where "cofinanc" or "co-financ" is present
                if "co-financiers and other external partners" in page_text_lower:
                    lines = page_text_lower.split('\n')

                    for i,line in enumerate(lines):
                        # if 'co-financiers' in line:
                        #     print(line)
                        if "co-financiers and other external partners" in line:
                            neighbourhood = lines[i:i+2] if i+2 < len(lines) else lines[i:i+1] if i+1 < len(lines) else lines[i]
                            neighbourhood = ' '.join(neighbourhood)
                            # print(neighbourhood)
                            found_partners = text_partners(neighbourhood)
                            if found_partners:
                                partnerships += found_partners
                                break


    except Exception as e:
        print(f'Error reading project completion report: {traceback.format_exc()}')

    partners = filter_partners(partnerships)
    
    return partners
    
    

                
def run(args:argparse.Namespace):
    '''Run main script. For each file, gets project metadata and identifies partners'''
    files = sorted(os.listdir(data_folder))
    p_bar = tqdm(total=len(files),desc='Processing files')

    found_partners = 0
    for file in files:
        f_path = f'{data_folder}/{file}'
        metadata_path = f'{f_path}/metadata.json'
        doc_path = f'{f_path}/pcr.pdf'

        # get project metadata
        if args.get_metadata or args.full:
            metadata = json.load(open(metadata_path,'r',encoding='utf-8'))
            get_project_metadata(metadata_path,doc_path,page_metadata=args.toggle_page_metadata,logs=args.logs,debug=args.debug)

        # identify partners
        if args.identify_partners or args.full:
            partners = identify_partners(metadata_path,doc_path,logs=args.logs,debug=args.debug)
            # save partners
            metadata = json.load(open(metadata_path,'r',encoding='utf-8'))
            metadata['partnerships'] = partners
            json.dump(metadata,open(metadata_path,'w',encoding='utf-8'),indent=4)
            if partners:
                found_partners += 1

        p_bar.update(1)

    if args.identify_partners or args.full:
        print(f'Found {found_partners} projects with partners.')


if __name__ == '__main__':
    args = process_args()
    
    run(args)