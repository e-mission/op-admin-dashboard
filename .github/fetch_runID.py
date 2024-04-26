
import json
import logging
import requests
import sys

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Just checking to see if workflows are being fetched; it works!!!
    # download_url = "https://api.github.com/repos/MukuFlash03/e-mission-server/actions/workflows"
    # logging.debug("About to fetch workflows present in e-mission-server from %s" % download_url)
    # r = requests.get(download_url)
    # if r.status_code != 200:
    #     logging.debug(f"Unable to fetch workflows, status code: {r.status_code}")
    #     sys.exit(1)
    # else:
    #     workflows_json = json.loads(r.text)
    #     logging.debug(f"Successfully fetched workflows")
    #     print(workflows_json)

        # for workflow_entry in workflows_json["workflows"]:
        #     print("Workflow name: %s ; id: %s " % (workflow_entry["name"], workflow_entry["id"]))
    


    '''
        Workflow "docker image" uses image_build_push.yml
        From above commented out code, and checked via terminal as well,
        workflow id for the "docker image" can be fetched using:
        https://api.github.com/repos/MukuFlash03/e-mission-server/actions/workflows
        https://api.github.com/repos/e-mission/e-mission-server/actions/workflows

        For MukuFlash03: id = 75506902 
        For e-mission-server: id = 35580278 
    '''
    download_url = "https://api.github.com/repos/MukuFlash03/e-mission-server/actions/workflows/75506902/runs"
    # download_url = "https://api.github.com/repos/e-mission/e-mission-server/actions/workflows/35580278/runs"
    logging.debug("About to fetch workflow runs present in docker image workflow present in e-mission-server from %s" % download_url)
    r = requests.get(download_url)
    if r.status_code != 200:
        logging.debug(f"Unable to fetch workflow runs, status code: {r.status_code}")
        sys.exit(1)
    else:
        workflow_runs_json = json.loads(r.text)
        logging.debug(f"Successfully fetched workflow runs")
        # print(workflow_runs_json)

        workflow_runs = workflow_runs_json["workflow_runs"]
        if workflow_runs:
            successful_runs = [run for run in workflow_runs \
                                if run["status"] == "completed" and \
                                run["conclusion"] == "success" and \
                                run["head_branch"] == "tags-artifact"
                               ]
            # print(successful_runs)
            if successful_runs:
                sorted_runs = successful_runs.sort(reverse=True, key=lambda x: x["updated_at"])
                sorted_runs = sorted(successful_runs, reverse=True, key=lambda x: x["updated_at"])
                # print(sorted_runs)
                latest_run_id = sorted_runs[0]["id"]
                # print(latest_run_id)
                print(f"::set-output name=run_id::{latest_run_id}")
            # else:
            #     print("No successful runs")
        # else:
        #     print("No workflow runs found")
