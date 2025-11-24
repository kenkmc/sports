import os
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


class SheetsUploader:
    def __init__(self, sheet_id=None, creds_path=None):
        self.sheet_id = sheet_id
        self.creds_path = creds_path or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        self.client = None
        self.sheet = None
        if self.creds_path and self.sheet_id:
            self._init_client()

    def _init_client(self):
        creds = Credentials.from_service_account_file(self.creds_path, scopes=SCOPES)
        self.client = gspread.authorize(creds)
        try:
            self.sheet = self.client.open_by_key(self.sheet_id).sheet1
        except Exception:
            self.sheet = None

    def upload_record(self, record: dict):
        # record should be a dict that matches header order
        if not self.sheet:
            print('Sheets not configured. Skipping upload.')
            return False
        try:
            headers = self.sheet.row_values(1)
            row = [record.get(h, '') for h in headers]
            self.sheet.append_row(row)
            return True
        except Exception as e:
            print('Sheets upload failed:', e)
            return False
