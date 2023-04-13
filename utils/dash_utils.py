from urllib.parse import urlparse, parse_qs


def get_query_params(url):
    params = parse_qs(urlparse(url).query)
    return params


def get_study_name_from_params_or_default(parsed_url):
    params = parse_qs(parsed_url.query)
    return params['study_config'][0] if 'study_config' in params else "stage-study"


def get_study_name_from_url(url):
    # url = 'https://stage-program.nrel.gov.com/e-mission/em-public-dashboard/blob/main/frontend?study_config=stage'
    study_name = None
    parsed_url = urlparse(url)
    host_name = str(parsed_url.hostname)
    if host_name == 'localhost' or host_name == '127.0.0.1':
        return get_study_name_from_params_or_default(parsed_url)
    first_domain = host_name.split(".")[0]
    if first_domain == "openpath-stage":
        return get_study_name_from_params_or_default(parsed_url)
    openpath_index = first_domain.index("-openpath")
    if openpath_index != -1:
        study_name = first_domain[:openpath_index]
    return study_name
