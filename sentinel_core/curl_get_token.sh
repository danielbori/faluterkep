#!/bin/bash

curl -X POST https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token -d '{"client_id"="cdse-public", "username"="your_username", "password"="your_password", "grant_type"="password"}'


