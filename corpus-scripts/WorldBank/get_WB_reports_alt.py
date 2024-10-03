import re
import sys
import requests, json, os,argparse,tqdm
from utils.utils import *




docrel= ["Project Appraisal Document",
         "Implementation Completion Report Review",
         "Implementation Completion and Results Report",
         "Project Performance Assessment Report",
        ]

wb="https://search.worldbank.org/api/v2/wds"   ## base WB api
p_wb = "https://search.worldbank.org/api/v2/projects" ## projects WB api

run_path = os.getcwd()
progress_bar = tqdm.tqdm(total=0)




def process_args():
    parser = argparse.ArgumentParser(
        description="Script to obtain WB reports"
    )
    parser.add_argument("-t","--threads"                        , type=int, nargs="?", default=1                                                , help="Number of threads to use. Default=1")
    parser.add_argument("-d","--debug"                          , action='store_true', default=False                                            , help="Debug mode. Default=False")
    return parser.parse_args()



def get_save_pdf(url,file,debug=False):            ## get url and save it to file
    if not os.path.exists(file):           ## if not already exists
        if debug:                 ## if option debug
            print(f"geting document {file} | {url}")
        r= get_request(url)
        if r:
            with open(file, 'wb') as f:        ## and save it
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)


def get_country_projects(country,debug=False):
    '''Get all projects for a country using the WB API'''
    projects = {}
    if debug:
        print(f"Geting projects for {country}")
    country_projects = {}
    params = {
        "format": "json",
        "countrycode": country,
        "projectstatusdisplay": "Closed",
        "rows": 500,
        "os": 0
    }
    # get all projects
    ## iterate over pages if needed
    total_projects = 0
    while True:
        response = get_request(p_wb,params=params)
        if response:
            response = response.json()
            total = int(response['total'])
            projects = response['projects']
            # add projects after 2004
            country_projects.update({project_id:project for project_id,project in projects.items() if 'approvalfy' in project and int(project['approvalfy']) > 2004})
            total_projects += len(projects)
            if total_projects >= total:
                break
            params['os'] += 1
        else:
            break


    return projects



def get_project_documents(project_id,country,debug=False):
    '''Get all documents for a project using the WB API'''
    if debug:
        print(f"Geting documents for {project_id}")
    params = {
        "format": "json",
        "projectid": project_id,
        "rows": 500,
        "os": 0
    }
    seen_documents = 0
    while True:
        try:
            response = get_request(wb,params=params)
            if response:
                    response = response.json()
                    # print(response)
                    total = response['total']
                    documents = response['documents']
                    for doc_id,document in documents.items():
                        if 'docty' in document and document['docty'] in docrel:
                            url = document['pdfurl']
                            file = f"{run_path}/data/{country}/{project_id}/{doc_id}_{document['docty']}.pdf"
                            get_save_pdf(url,file,debug)


                    seen_documents += len(documents)
                    if seen_documents >= total:
                        break 
            else:
                break    
        except:
            break


def get_documents(country_list,debug=False):
    ''' 
    downloads Report from WB
    related to a country
    '''
    for country in country_list:
        dir = f"{run_path}/data/{country}"

        # create country folder
        if not os.path.exists(dir):
            os.makedirs(dir)


        projects = get_country_projects(country,debug)

        for project_id,project_metada in projects.items():

            # clean metadata
            project_country = country_iso_codes[project_metada['countryshortname']]
            project_metada['country_iso3'] = project_country
            project_metada['end_date'] = re.search(r'\d{4}',project_metada['closingdate']).group(0) if 'closingdate' in project_metada else None
            project_metada['total_budget'] = int(project_metada['totalamt'].replace(',','')) if 'totalamt' in project_metada else 0
            sector = project_metada['sector'] if 'sector' in project_metada else None
            # turn list of sector dict into single item or list of items (not dicts)
            if type(sector) == list and len(sector) > 0 and type(sector[0]) == dict:
                new_sector = []
                for s in sector:
                    new_sector.append(s['Name'])
                if len(new_sector) == 1:
                    new_sector = new_sector[0]
                sector = new_sector
            project_metada['sector'] = sector

            # save project metadata
            if not os.path.exists(f"{dir}/{project_id}"):
                os.makedirs(f"{dir}/{project_id}")
                with open(f"{dir}/{project_id}/metadata.json","w",encoding="utf-8") as f:
                    json.dump(project_metada, f,indent=4)

            get_project_documents(project_id,country,debug)

        progress_bar.update(1)


if __name__ == '__main__':

    args = process_args()

    # country codes
    if os.path.exists(f"{run_path}/countries"):
        countries = open(f"{run_path}/countries").read().splitlines()[1:]
        progress_bar.reset(len(countries))
    else:
        print("File with country codes not found.")
        sys.exit(0)


    # create data folder if not exists
    if not os.path.exists("data"):
        os.makedirs("data")

    country_iso_codes_path = f'{run_path}/wb_countries_iso3_mapping.json'
    country_iso_codes = json.load(open(country_iso_codes_path,'r',encoding='utf-8'))

    # get documents
    run_threaded_for(get_documents,countries,[country_iso_codes,args.debug],log=args.debug,threads=args.threads)
