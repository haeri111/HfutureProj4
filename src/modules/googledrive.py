#############################################################################################################
# DB 연동 방식(구글 드라이브 API 활용, chromaDB의 경우 파일단위로 DB 관리)
# 1. 구글 공유 드라이브내 파일 로컬 or 기동중인 서버 디렉토리 내 다운
# 2. 저장한 DB 파일을 활용하여 데이터 조회
# 3. DB 데이터 수정 시 collection.add 등 chroma DB 함수 활용 시 자동으로 불러온 chroma.sqlite3 파일에 수정
# 4. DB 데이터 수정 후 chroma.sqlite3 파일을 구글 공유 드라이브에 업로드 하여 DB 데이터 연동
#############################################################################################################
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaFileUpload
import streamlit as st

class GoogleDrive: 

    def __init__(self, service_account_file, scopes):
        self.credentials = Credentials.from_service_account_file(service_account_file, scopes=scopes)
        self.drive_service = build('drive', 'v3', credentials=self.credentials)

    # Google Drive API로 폴더 내 파일 가져오기
    def list_files_in_folder(self, folder_id):
        results = self.drive_service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name, mimeType)"
        ).execute()
        return results.get('files', [])

    # 특정 파일 다운로드
    def download_file(self, file_id, destination):
        request = self.drive_service.files().get_media(fileId=file_id)
        with open(destination, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}% complete.")

    def download_chroma_db(self, folder_id, db_filename="chroma.sqlite3", local_path="./chroma_db2/chroma.sqlite3"):
        
        print(f"♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡ {st.session_state}")
        if 'googledrive' not in st.session_state:
            # 구글 드라이브에서 chroma.sqlite3 파일을 찾고 다운로드
            files = self.list_files_in_folder(folder_id)
            db_file_id = None
            for file in files:
                print(f"File Name: {file['name']}, File ID: {file['id']}, MimeType: {file['mimeType']}")
                # chroma.sqlite3 파일 찾기
                if file['name'] == db_filename:
                    db_file_id = file['id']

            # chroma.sqlite3 파일 다운로드
            if db_file_id:
                print(f"Downloading '{db_filename}' with File ID: {db_file_id}")
                self.download_file(db_file_id, local_path) # 로컬에 저장
                print(f"File '{db_filename}' has been downloaded successfully.")
            else:
                print(f"No {db_filename} file found in the folder.")
            
            # 세션에 db, retriever 저장
            st.session_state["googledrive"] = True
        else:
            print(f"File '{db_filename}' has already been downloaded previously. Skipping download.")

        print(f"♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡2222 {st.session_state}")


    #############################################################################################################
    # 1.DB 데이터 변경 시 upload_to_drive() 함수 호출
    # 2. 공유 드라이브 내 파일 업로드
    # ex) upload_to_drive("/content/chroma.sqlite3", "chroma.sqlite3", "1sU-NhDxUm4U5c3IVPmfiHmO_1pY4bjLg") -> chroma.sqlite3가 저장된 경로(수정),파일명(고정),업로드 폴더ID(고정)
    #############################################################################################################
    def upload_to_drive(self, file_path, file_name, folder_id):

        if not os.path.exists(file_path):
            print(f"Local file not found: {file_path}")
            return None

        local_file_size = os.path.getsize(file_path)
        print(f"Local file size: {local_file_size} bytes")

        # 기존 파일 검색
        query = f"'{folder_id}' in parents and name = '{file_name}' and trashed = false"
        results = self.drive_service.files().list(q=query, fields="files(id, name, size)").execute()
        files = results.get("files", [])

        # 기존 파일 삭제
        if files:
            for file in files:
                self.drive_service.files().delete(fileId=file["id"]).execute()
                print(f"Deleted existing file: {file['name']} (ID: {file['id']})")

        # 새 파일 업로드
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        uploaded_file = self.drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, size'
        ).execute()

        uploaded_file_size = int(uploaded_file.get('size', 0))
        print(f"Uploaded {file_name} to Google Drive with ID: {uploaded_file['id']}")
        print(f"Uploaded file size: {uploaded_file_size} bytes")

        # 크기 검증
        if local_file_size == uploaded_file_size:
            print("Upload verification successful: File sizes match!")
        else:
            print("Upload verification failed: File sizes do not match!")

        return uploaded_file['id']