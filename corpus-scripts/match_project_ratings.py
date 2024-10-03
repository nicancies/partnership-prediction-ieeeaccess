import bs4
import pandas as pd
import sys
import json, os,argparse,tqdm
from utils.utils import *
from bs4 import BeautifulSoup



file_path = os.path.dirname(os.path.abspath(__file__))
ratings_folder = f'{file_path}/../misc/project_ratings'
partner_corpus_folder = f'{file_path}/corpus/projects/partner'
control_corpus_folder = f'{file_path}/corpus/projects/control'



def process_args():
    parser = argparse.ArgumentParser(
        description="Script to obtain WB reports"
    )
    parser.add_argument("-f","--full"                           , action='store_true', default=False                                            , help="Full mode. Default=False")
    parser.add_argument("-pc","--partner_corpus"                , action='store_true', default=False                                            , help="Partner corpus. Default=False")
    parser.add_argument('-cc','--control_corpus'               , action='store_true', default=False                                            , help="Control corpus. Default=False")
    parser.add_argument('-rf','--ratings_folder'               , type=str, nargs="?", default=ratings_folder                                        , help="Ratings folder. Default=ratings")
    parser.add_argument('-pcf','--partner_corpus_folder'       , type=str, nargs="?", default=partner_corpus_folder                                , help="Partner corpus folder. Default=partner_corpus")
    parser.add_argument('-ccf','--control_corpus_folder'       , type=str, nargs="?", default=control_corpus_folder                                , help="Control corpus folder. Default=control_corpus")
    parser.add_argument('-cr','--cached_ratings'               , action='store_false', default=True                                            , help="Use cached ratings. Default=True")
    parser.add_argument('--excel'                             , action='store_true', default=False                                            , help="Output to excel. Default=False")
    parser.add_argument('--complete_IFAD_partners'             , action='store_true', default=False                                            , help="Complete IFAD partners. Default=False")
    parser.add_argument("-d","--debug"                          , action='store_true', default=False                                            , help="Debug mode. Default=False")
    parser.add_argument('--test'                              , action='store_true', default=False                                            , help="Test mode. Default=False")
    return parser.parse_args()



def load_AsDB_ratings(AsDB_ratings_path:str)->dict:
    '''Load AsDB ratings'''
    df = pd.read_excel(AsDB_ratings_path,sheet_name='AER data')
    # clean columns
    relevant_collumns = ['Project No.','PCR overall rating','PCR relevance','PCR relevance','PCR effectiveness','PCR efficiency',
                         'PCR sustainability','PCR EA Performance','PCR ADB Performance','PVR overall rating',
                         'PVR relevance','PVR relevance','PVR effectiveness','PVR efficiency','PVR sustainability',
                         'PVR EA Performance','PVR ADB  Performance','PCR Quality','PPER overall rating',
                         'PPER relevance','PPER relevance','PPER effectiveness','PPER efficiency','PPER sustainability',
                         'PPER EA Performance','PPER ADB Performance','Source of latest rating','IED overall rating',
                         'IED relevance','IED effectiveness','IED efficiency','IED sustainability','IED EA Performance',
                         'IED ADB Performance']
    df = df[relevant_collumns]

    df_dict = {}
    for index, row in df.iterrows():
        try:
            project_no = row['Project No.']
            # print(project_no,row)
            if row['IED overall rating'] not in ['-','nan']:
                if len(project_no.split('/')) > 1:
                    projects = project_no.split('/')
                else:
                    projects = [project_no]

                for project in projects:
                    df_dict[project] = { }
                    for col in relevant_collumns[1:]:
                        df_dict[project][col] = row[col] if type(row[col]) != pd.Series else row[col].iloc[-1]

        except Exception as e:
            print(e)

    return df_dict


def load_IFAD_ratings(IFAD_ratings_path:str)->dict:
    '''Load IFAD ratings'''
    df = pd.read_excel(IFAD_ratings_path,sheet_name='All Evaluations ratings',header=2)
    # clean columns
    relevant_collumns = ['Project ID','Project Overall Achievement','Relevance','Effectiveness','Efficiency','Sustainability',
                         'Average Project Performance','IFAD','Cooperating Institution','Government','NGO/Other','Co-Financiers',
                         'Combined Partner Performance','Physical Assets','Financial Assets','Food Security','Agricultural Productivity',
                         'Human Assets','Social Capital & Empowerment','Institutions & Policies','Natural resources, environment and climate change**',
                         'Markets','Rural Poverty Impact','Innovation and scaling up*','Innovation*','Scaling-up*',"Gender equality & women's empowerment",
                         "Environment & Natural Resource Management**",'Adaptation to climate change**']
    
    relevant_collumns += ['Cooperating Institution']

    df = df[relevant_collumns]
    # unique Project No.
    df = df.drop_duplicates(subset=['Project ID'])

    df_dict = {}
    for index, row in df.iterrows():
        project_no = row['Project ID']
        project_no = f'11{"0"*(8-len(str(project_no)))}{project_no}'
        df_dict[project_no] = { }
        for col in relevant_collumns[1:]:
            df_dict[project_no][col] = row[col] if type(row[col]) != pd.Series else row[col].iloc[-1]

    return df_dict


def load_UNDP_ratings(UNDP_ratings_path:str)->dict:
    '''Load UNDP ratings'''
    df = pd.read_excel(UNDP_ratings_path)
    # clean columns
    relevant_collumns = ['ERC Link','Quality Rating']
    df = df[relevant_collumns]
    # unique ERC Link
    df = df.drop_duplicates(subset=['ERC Link'])

    # for each row, if quality rating value, check link to find project ID
    df_dict = {}
    p_bar = tqdm.tqdm(total=len(df))
    total_atlas_projects = 0
    atlas_projects = []

    for index, row in df.iterrows():
        rating = str(row['Quality Rating']).strip()
        ecr_link = row['ERC Link']
        if rating != 'nan':
            r = get_request(ecr_link)
            if r:
                soup = BeautifulSoup(r.content, 'lxml')
                spans = soup.find_all('span')
                for span in spans:
                    span:bs4.element.Tag
                    if span.text.strip() == 'Atlas Project Number':
                        total_atlas_projects += 1
                        atlas_projects.append(ecr_link)
                        atlas_row = span.parent.parent
                        cols = atlas_row.find_all('div')
                        project_id = cols[1].text.strip()
                        df_dict[project_id] = {
                            'Quality Rating':rating,
                            'ERC Link':ecr_link
                        }
                        break

        p_bar.update(1)

    print(f'Total Atlas Projects: {total_atlas_projects}')
    # save atlas projects
    with open('atlas_projects.json','w',encoding='utf-8') as f:
        json.dump(atlas_projects,f,indent=4)

    return df_dict


def load_WB_ratings(WB_ratings_path:str)->dict:
    '''Load WB ratings'''
    df = pd.read_excel(WB_ratings_path)
    relevant_collumns = ['Project ID','IEG Outcome Ratings','IEG Bank Performance Ratings','IEG Quality at Entry Ratings','IEG Quality of Supervision Ratings',
                         'IEG Monitoring and Evaluation Quality Ratings']
    df = df[relevant_collumns]
    # unique Project No.
    df = df.drop_duplicates(subset=['Project ID'])

    df_dict = {}
    for index, row in df.iterrows():
        project_no = row['Project ID']
        df_dict[project_no] = { }
        for col in relevant_collumns[1:]:
            df_dict[project_no][col] = row[col] if type(row[col]) != pd.Series else row[col].iloc[-1]

    return df_dict

def load_AfDB_ratings()->dict:
    '''Load AfDB ratings'''
    ratings = {}
    control_project_path = f'{control_corpus_folder}/AfDB'
    partner_project_path = f'{partner_corpus_folder}/AfDB'

    if os.path.exists(control_project_path):
        for project_id in os.listdir(control_project_path):
            metadata = json.load(open(f'{control_project_path}/{project_id}/metadata.json'))
            if 'project_rating' in metadata:
                ratings[project_id] = metadata['project_rating']

    if os.path.exists(partner_project_path):
        for project_id in os.listdir(partner_project_path):
            metadata = json.load(open(f'{partner_project_path}/{project_id}/metadata.json'))
            if 'project_rating' in metadata:
                ratings[project_id] = metadata['project_rating']

    return ratings


def load_ratings(ratings_folder:str,logs=False)->dict:
    '''Load ratings'''
    ratings = {
        'AsDB':{},
        'IFAD':{},
        'UNDP':{},
        'WB':{},
        'AfDB':{},
    }
    AsDB_ratings_path = f'{ratings_folder}/AsDB.xlsx'
    if os.path.exists(AsDB_ratings_path):
        ratings['AsDB'] = load_AsDB_ratings(AsDB_ratings_path)
        if logs:
            print(f'AsDB loaded - {len(ratings["AsDB"])} projects')

    IFAD_ratings_path = f'{ratings_folder}/IFAD.xlsx'
    if os.path.exists(IFAD_ratings_path):
        ratings['IFAD'] = load_IFAD_ratings(IFAD_ratings_path)
        if logs:
            print(f'IFAD loaded - {len(ratings["IFAD"])} projects')

    UNDP_ratings_path = f'{ratings_folder}/UNDP.xlsx'
    if os.path.exists(UNDP_ratings_path):
        ratings['UNDP'] = load_UNDP_ratings(UNDP_ratings_path)
        if logs:
            print(f'UNDP loaded - {len(ratings["UNDP"])} projects')

    WB_ratings_path = f'{ratings_folder}/WB.xlsx'
    if os.path.exists(WB_ratings_path):
        ratings['WB'] = load_WB_ratings(WB_ratings_path)
        if logs:
            print(f'WB loaded - {len(ratings["WB"])} projects')

    ratings['AfDB'] = load_AfDB_ratings()
    if logs:
        print(f'AfDB loaded - {len(ratings["AfDB"])} projects')
    
    return ratings


def project_rating(organization:str,ratings:dict):
    '''Project rating'''
    rating = None
    if organization == 'AsDB':
        rating = ratings['IED overall rating']
    elif organization == 'IFAD':
        rating = ratings['Project Overall Achievement']
    elif organization == 'UNDP':
        rating = ratings['Quality Rating']
    elif organization in ['WB','WorldBank']:
        rating = ratings['IEG Outcome Ratings']
    elif organization == 'AfDB':
        rating = ratings

    if rating:
        rating = map_rating(organization,rating)

    return rating

def match_corpus(ratings:dict,projects_path:str,logs=False):
    '''Match corpus'''
    if logs:
        print('Matching partner corpus')
    
    matches = {}
    matched_projects = {}

    if os.path.exists(projects_path):
        projects_folder = projects_path

        for organization in os.listdir(projects_folder):
            map_o_name = {
                'UNDP':'UNDP',
                'WB':'WB',
                'IFAD':'IFAD',
                'AsDB':'AsDB',
                'AfDB':'AfDB'
            }
            matched_projects[map_o_name[organization]] = {}
            for project in os.listdir(f'{projects_folder}/{organization}'):
                project_ratings = None
                if organization == 'AsDB':
                    if project in ratings['AsDB']:
                        project_ratings = ratings['AsDB'][project]
                    elif len(project.split('-')) > 1 and project.split('-')[0] in ratings['AsDB']:
                        project_ratings = ratings['AsDB'][project.split('-')[0]]
                elif project in ratings[map_o_name[organization]]:
                    project_ratings = ratings[map_o_name[organization]][project]

                if project_ratings:
                    # update metadata
                    metadata_path = f'{projects_folder}/{organization}/{project}/metadata.json'
                    metadata = json.load(open(metadata_path,'r',encoding='utf-8'))
                    metadata['rating'] = project_ratings
                    metadata['project_rating'] = project_rating(organization,project_ratings)
                    json.dump(metadata,open(metadata_path,'w',encoding='utf-8'),indent=4)

                    matched_projects[map_o_name[organization]][project] = project_ratings
                    
                    if organization not in matches:
                        matches[organization] = 1
                    else:
                        matches[organization] += 1

    if logs:
        print(f'Matches: {matches}')

    return matched_projects




def map_rating(organization:str,rating:str)->int:
    '''Map rating according to organization'''
    if organization in ['WB','UNDP','WorldBank']:
        rating_map = {
            'highly satisfactory' : 4,
            'satisfactory' : 3,
            'moderately satisfactory' : 2,
            'mostly satisfactory' : 2,
            'moderately unsatisfactory' : 2,
            'mostly unsatisfactory' : 2,
            'unsatisfactory' : 1,
            'highly unsatisfactory' : 1
        }
    elif organization == 'AsDB':
        rating_map = {
            'highly successful' : 4,
            'successful' : 3,
            'less than successful' : 2,
            'unsuccessful' : 1
        }
    elif organization == 'IFAD':
        rating_map = {
            '1' : 1,
            '2' : 1,
            '3' : 2,
            '4' : 2,    
            '5' : 3,
            '6' : 4
        }
    elif organization == 'AfDB':
        rating_map = {
            '1' : 1,
            '2' : 2,
            '3' : 3,
            '4' : 4, 
        }

    return rating_map[str(rating).strip().lower()] if str(rating).strip().lower() in rating_map else 'NA'

def ratings_to_excel(ratings_path:str,output_path:str,logs=False):
    '''Convert ratings to excel. Only retain common overall rating column'''
    if logs:
        print(f'Converting ratings from "{ratings_path}" to excel')

    if os.path.exists(ratings_path):
        ratings = json.load(open(ratings_path,'r',encoding='utf-8'))
        df = pd.DataFrame(data=[],columns=['project','organization','rating','original_rating'])
        
        for organization in ratings:
            for project in ratings[organization]:
                if organization == 'IFAD':
                    rating = ratings[organization][project]['Project Overall Achievement']
                elif organization == 'AsDB':
                    rating = ratings[organization][project]['IED overall rating']
                elif organization == 'WB':
                    rating = ratings[organization][project]['IEG Outcome Ratings']
                elif organization == 'UNDP':
                    rating = ratings[organization][project]['Quality Rating']
                elif organization == 'AfDB':
                    rating = ratings[organization][project]

                mapped_rating = map_rating(organization,rating)
                df = df._append({'project':project,'organization':organization,'rating':mapped_rating,'original_rating':rating},ignore_index=True)


        df.to_excel(output_path)
    else:
        print(f'"{ratings_path}" not found')


def complete_IFAD_partners(ratings_path:str,logs=False):
    '''Complete IFAD partners using IFAD ratings'''
    if logs:
        print('Comparing IFAD partners')

    ifad_projects = load_IFAD_ratings(ifad_ratings_path)

    projects_with_partners = 0
    partnered_projects = {}

    from IFAD.identify_IFAD_partnerships import text_partners,filter_partners

    for project_id in ifad_projects:
        project = ifad_projects[project_id]
        cofinancers = project['Cooperating Institution']
        partners = text_partners(cofinancers)
        partners = filter_partners(partners)
        # print(project_id,partners)
        if len(partners) > 0:
            projects_with_partners += 1
            partnered_projects[project_id] = partners

    print('Projects with partners:',projects_with_partners)

    ifad_data_path = f'{file_path}/IFAD/data'
    not_found = 0
    for country in os.listdir(ifad_data_path):
        country_data_path = f'{ifad_data_path}/{country}'
        for project_id in os.listdir(country_data_path):
            project_path = f'{country_data_path}/{project_id}'
            metadata = json.load(open(f'{project_path}/metadata.json','r',encoding='utf-8'))
            if project_id in partnered_projects:
                if not metadata['partnerships']:
                    not_found += 1
                    metadata['partnership_from_ratings'] = partnered_projects[project_id]
                    metadata['partnerships'] = partnered_projects[project_id]
                else:
                    metadata['partnership_from_ratings'] = partnered_projects[project_id]
                    for partner in partnered_projects[project_id]:
                        if partner not in metadata['partnerships']:
                            metadata['partnerships'].append(partner)
                            not_found += 1
            else:
                metadata['partnership_from_ratings'] = 'None'

            json.dump(metadata,open(f'{project_path}/metadata.json','w',encoding='utf-8'))



    print('Not found:',not_found)



if __name__ == "__main__":
    args = process_args()
    
    cached_rating_path = f'{file_path}/ratings.json'

    if args.control_corpus or args.full or args.partner_corpus:
        if not args.cached_ratings or not os.path.exists(cached_rating_path):
            ratings = load_ratings(args.ratings_folder,args.debug)

            # save rating to json
            with open('ratings.json','w',encoding='utf-8') as f:
                json.dump(ratings,f)
        else:
            ratings = json.load(open(cached_rating_path,'r',encoding='utf-8'))


        if args.partner_corpus or args.full:
            print("Partner corpus")
            matched_projects = match_corpus(ratings,args.partner_corpus_folder,args.debug)

            # save matched projects to json
            with open('matched_partner_projects.json','w',encoding='utf-8') as f:
                json.dump(matched_projects,f,indent=4)

        if args.control_corpus or args.full:
            print("Control corpus")
            matched_projects = match_corpus(ratings,args.control_corpus_folder,args.debug)

            # save matched projects to json
            with open('matched_control_projects.json','w',encoding='utf-8') as f:
                json.dump(matched_projects,f,indent=4)

    if args.excel:
        ratings_to_excel('matched_partner_projects.json','partnered.xlsx',args.debug)
        ratings_to_excel('matched_control_projects.json','controlled.xlsx',args.debug)

    if args.complete_IFAD_partners:
        ifad_ratings_path = f'{args.ratings_folder}/IFAD.xlsx'
        complete_IFAD_partners(ifad_ratings_path,args.debug)

    if args.test:
        print('Test')
        
        load_UNDP_ratings(f'{args.ratings_folder}/UNDP.xlsx')
        

