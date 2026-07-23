"""Handles OAuth2 flow for Gmail + Calendar. Run once interactively to
generate token.json; after that it refreshes silently."""
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import config


def get_google_credentials() -> Credentials:
    creds = None
    if os.path.exists(config.GOOGLE_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(
            config.GOOGLE_TOKEN_PATH, config.GOOGLE_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(config.GOOGLE_CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Missing {config.GOOGLE_CREDENTIALS_PATH}. "
                    "Download OAuth client JSON from Google Cloud Console first "
                    "(see README section 3)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                config.GOOGLE_CREDENTIALS_PATH, config.GOOGLE_SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(config.GOOGLE_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return creds


def gmail_service():
    return build("gmail", "v1", credentials=get_google_credentials())


def calendar_service():
    return build("calendar", "v3", credentials=get_google_credentials())