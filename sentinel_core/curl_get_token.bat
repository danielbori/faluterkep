@ECHO OFF
curl -d client_id=cdse-public -d username=your_username -d password=your_password -d grant_type=password https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token

