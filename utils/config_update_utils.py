import os
import logging
from jose import jwt
import arrow
import requests


CONFIG_REPO = os.getenv("CONFIG_REPO")
WORKFLOW = os.getenv("CONFIG_UDPATES_WORKFLOW")
GH_APP_ID = os.getenv("CONFIG_UPDATES_GH_APP_ID")
GH_APP_INSTALLATION_ID = os.getenv("CONFIG_UPDATES_GH_APP_INSTALLATION_ID")
GH_APP_PRIVATE_KEY = os.getenv("CONFIG_UPDATES_GH_APP_PRIVATE_KEY")


def generate_jwt():
    if not GH_APP_ID or not GH_APP_PRIVATE_KEY:
        raise ValueError("GH app credentials not set in env")

    now = int(arrow.utcnow().timestamp())
    payload = {
        "iat": now,
        "exp": now + 600,  # 10 min
        "iss": GH_APP_ID,
    }

    encoded_jwt = jwt.encode(payload, GH_APP_PRIVATE_KEY, algorithm="RS256")
    return encoded_jwt


def get_installation_access_token():
    if not GH_APP_INSTALLATION_ID:
        raise ValueError("GH app installation ID not set in env")

    jwt_token = generate_jwt()

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json"
    }

    url = f"https://api.github.com/app/installations/{GH_APP_INSTALLATION_ID}/access_tokens"
    response = requests.post(url, headers=headers)
    response.raise_for_status()

    return response.json()["token"]


def trigger_config_update_workflow(inputs):
    try:
        token = get_installation_access_token()
    except ValueError as e:
        logging.warning(f"Failed to get installation access token: {e}")
        return 500

    inputs['token'] = token

    response = requests.post(
        f"https://api.github.com/repos/{CONFIG_REPO}/actions/workflows/{WORKFLOW}/dispatches",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        },
        json={
            "ref": 'main',
            "inputs": inputs,
        })
    response.raise_for_status()
    return response.status_code


def get_workflow_run_status(run_id):
    url = f"https://api.github.com/repos/{CONFIG_REPO}/actions/runs/{run_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_pr_status(pr_number):
    url = f"https://api.github.com/repos/{CONFIG_REPO}/pulls/{pr_number}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_recent_workflow_run(threshold_seconds=60):
    now = arrow.utcnow().timestamp()
    threshold_time = now - threshold_seconds

    url = f"https://api.github.com/repos/{CONFIG_REPO}/actions/workflows/{WORKFLOW}/runs"
    response = requests.get(url)
    response.raise_for_status()
    runs = response.json().get("workflow_runs", [])

    for run in runs:
        if run["event"] != "workflow_dispatch":
            continue
        run_created_ts = arrow.get(run["created_at"]).timestamp()
        if run["head_branch"] == "main" and run_created_ts > threshold_time:
            return run
    return None


def get_recent_pr(run_id):
    url = f"https://api.github.com/repos/{CONFIG_REPO}/pulls?state=all"
    response = requests.get(url)
    response.raise_for_status()
    pulls = response.json()
    for pull in pulls:
        if pull["head"]["ref"].endswith(str(run_id)):
            return pull
    return None
