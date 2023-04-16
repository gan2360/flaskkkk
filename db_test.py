from app import db, UserInfo, FileInfo
user = UserInfo.query.get(1)
files = user.files
print(files.len())



